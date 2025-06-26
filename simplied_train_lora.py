##########################################################
# This script is simplied from https://github.com/huggingface/diffusers/blob/f161e277d0ec534afa4dfc461bc5baacffd7278b/examples/advanced_diffusion_training/train_dreambooth_lora_sdxl_advanced.py
##########################################################
import gc
from pathlib import Path
import os
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from diffusers.utils import load_image
from torchvision import transforms
from torchvision.transforms.functional import crop
from transformers import AutoTokenizer, PretrainedConfig
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTextModelWithProjection

##########################################################
################## init accelerator ######################
##########################################################
output_dir = "./exp"
logging_dir = "./exp/logs"

logging_dir = Path(output_dir, logging_dir)
accelerator_project_config = ProjectConfiguration(project_dir=output_dir, logging_dir=logging_dir)
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    gradient_accumulation_steps=1,
    mixed_precision="fp16",
    log_with="tensorboard",
    project_config=accelerator_project_config,
    kwargs_handlers=[kwargs],
)

##########################################################
################## load modules ##########################
##########################################################
pretrained_model_name_or_path = "cagliostrolab/animagine-xl-3.1"
weight_dtype = torch.float16

noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

tokenizer_one = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer_2",
    use_fast=False,
)

text_encoder_one = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder", #revision=args.revision, variant=args.variant
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder_2", #revision=args.revision, variant=args.variant
)

vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="vae",
)
latents_mean = latents_std = None
if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, 4, 1, 1)
if hasattr(vae.config, "latents_std") and vae.config.latents_std is not None:
    latents_std = torch.tensor(vae.config.latents_std).view(1, 4, 1, 1)

unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="unet", #revision=args.revision, variant=args.variant
)

vae.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
unet.requires_grad_(False)

unet.to(accelerator.device, dtype=weight_dtype)
vae.to(accelerator.device, dtype=torch.float32)
text_encoder_one.to(accelerator.device, dtype=weight_dtype)
text_encoder_two.to(accelerator.device, dtype=weight_dtype)

##########################################################
################## create lora layers ####################
##########################################################
target_modules = ["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out", "conv1", "conv2"]
unet_lora_config = LoraConfig(
    r=4,
    use_dora=False,
    lora_alpha=8,
    lora_dropout=0.0,
    init_lora_weights="gaussian",
    target_modules=target_modules,
)
unet.add_adapter(unet_lora_config)
cast_training_params([unet], dtype=torch.float32)

##########################################################
################## init training helpers #################
##########################################################
max_train_steps = 10
learning_rate = 1e-4
unet_lora_parameters = list(filter(lambda p: p.requires_grad, unet.parameters()))
unet_lora_parameters_with_lr = {"params": unet_lora_parameters, "lr": learning_rate}

optimizer = torch.optim.AdamW(
    [unet_lora_parameters_with_lr],
    betas=(0.9, 0.999),
    weight_decay=1e-04,
    eps=1e-08,
)

lr_scheduler = get_scheduler(
    "constant",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=max_train_steps * accelerator.num_processes,
    num_cycles=1,
    power=1.0,
)

unet, optimizer, lr_scheduler = accelerator.prepare(
    unet, optimizer, lr_scheduler
)

##########################################################
################## preprocess training data ##############
##########################################################
prompts = ["a photo of a cup in sand"]

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None, clip_skip=None):

    def tokenize_prompt(tokenizer, prompt, add_special_tokens=False):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        return text_input_ids

    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device), output_hidden_states=True, return_dict=False
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        if clip_skip is None:
            prompt_embeds = prompt_embeds[-1][-2]
        else:
            # "2" because SDXL always indexes from the penultimate layer.
            prompt_embeds = prompt_embeds[-1][-(clip_skip + 2)]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

prompt_embeds, pooled_prompt_embeds = encode_prompt(
    text_encoders=[text_encoder_one, text_encoder_two],
    tokenizers=[tokenizer_one, tokenizer_two],
    prompt=prompts,
    text_input_ids_list=None,
    clip_skip=None,
)

image_pil = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup.png")
# image_pil = load_image("cup.png")
original_sizes = [(image_pil.height, image_pil.width)]
size = resolution = 1024
interpolation = getattr(transforms.InterpolationMode, "LANCZOS", None)
train_resize = transforms.Resize(size, interpolation=interpolation)
train_crop = transforms.CenterCrop(size)
train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)
image = train_resize(image_pil)
y1 = max(0, int(round((image.height - size) / 2.0)))
x1 = max(0, int(round((image.width - size) / 2.0)))
image = train_crop(image)
crop_top_lefts = [(y1, x1)]
image = train_transforms(image)
pixel_values = torch.stack([image])
pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float().to(accelerator.device, dtype=vae.dtype)

##########################################################
################## training ##############################
##########################################################
for step in tqdm(range(max_train_steps)):
    unet.train()
    # vae encode
    model_input = vae.encode(pixel_values.clone()).latent_dist.sample()
    model_input = model_input * vae.config.scaling_factor
    # add noise
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
    )
    timesteps = timesteps.long()

    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

    # time ids
    def compute_time_ids(crops_coords_top_left, original_size=None):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        target_size = (resolution, resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids

    add_time_ids = torch.cat(
        [
            compute_time_ids(original_size=s, crops_coords_top_left=c)
            for s, c in zip(original_sizes, crop_top_lefts)
        ]
    )

    elems_to_repeat_text_embeds = 1

    unet_added_conditions = {
        "time_ids": add_time_ids,
        "text_embeds": pooled_prompt_embeds.repeat(elems_to_repeat_text_embeds, 1)
    }
    prompt_embeds_input = prompt_embeds.repeat(elems_to_repeat_text_embeds, 1, 1)

    # uent forward
    model_pred = unet(
        noisy_model_input,
        timesteps,
        prompt_embeds_input,
        added_cond_kwargs=unet_added_conditions,
        return_dict=False,
    )[0]

    # cal training loss
    target = noise
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    # update params
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(unet_lora_parameters, 1.0)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

##########################################################
################## save lora weights #####################
##########################################################
unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(accelerator.unwrap_model(unet)))
StableDiffusionXLPipeline.save_lora_weights(
    output_dir,
    unet_lora_layers=unet_lora_layers_to_save,
    text_encoder_lora_layers=None,
    text_encoder_2_lora_layers=None,
)

##########################################################
################## in-training evaluation ################
##########################################################
pipeline = StableDiffusionXLPipeline.from_pretrained(
    pretrained_model_name_or_path,
    vae=vae,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    text_encoder=accelerator.unwrap_model(text_encoder_one),
    text_encoder_2=accelerator.unwrap_model(text_encoder_two),
    unet=accelerator.unwrap_model(unet),
    # torch_dtype=weight_dtype,
).to(accelerator.device, dtype=torch.float32)

pipeline_args = {"prompt": prompts[0]}
generator = torch.Generator(device=accelerator.device).manual_seed(0)
# a normal image though it may look not perfect
pipeline(**pipeline_args, generator=generator).images[0].save("test_in_train_gen.jpg")
del pipeline
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

##########################################################
## inference by loading the pre-trained lora weights #####
##########################################################
pipeline2 = StableDiffusionXLPipeline.from_pretrained(
    pretrained_model_name_or_path,
    vae=vae,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    text_encoder=accelerator.unwrap_model(text_encoder_one),
    text_encoder_2=accelerator.unwrap_model(text_encoder_two),
).to(accelerator.device, dtype=torch.float32)
# warnings from here: missing params
pipeline2.load_lora_weights(output_dir, weight_name="pytorch_lora_weights.safetensors")
generator = torch.Generator(device=accelerator.device).manual_seed(0)
# a broken image
pipeline2(**pipeline_args, generator=generator).images[0].save("test_load_gen.jpg")



    