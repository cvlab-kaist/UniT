import re
import torch 
from diffusers.utils.torch_utils import is_compiled_module
from PIL import Image, ImageDraw, ImageFont

# Copied from dreambooth sd3 example
def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


# Copied from dreambooth sd3 example
def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


# Copied from dreambooth sd3 example
def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    # prompt_embeds = clip_prompt_embeds

    return prompt_embeds, pooled_prompt_embeds



# Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def compute_text_embeddings(args, accelerator, batch, text_encoders, tokenizers):
    with torch.no_grad():
        prompt = batch["prompts"]
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, args.max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(accelerator.device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}


# def get_sigmas(timesteps, accelerator, noise_scheduler_copy, n_dim=4, dtype=torch.float32):
#         sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
#         schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
#         timesteps = timesteps.to(accelerator.device)
#         step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

#         sigma = sigmas[step_indices].flatten()
#         while len(sigma.shape) < n_dim:
#             sigma = sigma.unsqueeze(-1)
#         return sigma

def get_sigmas(timesteps, accelerator, noise_scheduler_copy, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.cpu().long()  # keep on CPU for safe search
    timesteps_cpu = timesteps.detach().cpu().long().flatten()

    step_indices = []
    for t in timesteps_cpu.tolist():
        matches = (schedule_timesteps == t).nonzero(as_tuple=False)
        if matches.numel() == 0:
            raise ValueError(
                f"[get_sigmas] Timestep {t} not found in scheduler timesteps. "
                f"Valid range: {int(schedule_timesteps.min().item())}–{int(schedule_timesteps.max().item())}"
            )
        step_indices.append(int(matches[0].item()))

    step_indices = torch.tensor(step_indices, device=accelerator.device, dtype=torch.long)
    sigma = sigmas[step_indices].flatten()

    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)

    return sigma

def remove_focus_sentences(text):
    # 使用正则表达式按照 . ? ! 分割，并且保留分隔符本身
    # re.split(pattern, string) 如果 pattern 中带有捕获组()，分隔符也会保留在结果列表中
    prohibited_words = ['focus', 'focal', 'prominent', 'close-up', 'black and white', 'blur', 'depth', 'dense', 'locate', 'position']
    parts = re.split(r'([.?!])', text)
    
    filtered_sentences = []
    i = 0
    while i < len(parts):
        # sentence 可能是句子主体，punctuation 是该句子结尾的标点
        sentence = parts[i]
        punctuation = parts[i+1] if (i+1 < len(parts)) else ''

        # 组合为完整句子，避免漏掉结尾标点
        full_sentence = sentence + punctuation
        
        full_sentence_lower = full_sentence.lower()
        skip = False
        for word in prohibited_words:
            if word.lower() in full_sentence_lower:
                skip = True
                break
        
        # 如果该句子不包含任何禁用词，则保留
        if not skip:
            filtered_sentences.append(full_sentence)
        
        # 跳过已经处理的句子和标点
        i += 2
    
    # 根据需要选择如何重新拼接；这里去掉多余空格并直接拼接
    return "".join(filtered_sentences).strip()


def text_file_to_image(txt_file, img_file=None, font_size=16, img_width=1000, padding=10):
    """
    Converts a text file to an image and saves it.

    Args:
        txt_file (str): Path to the text file.
        img_file (str, optional): Path to save the image. 
                                  If None, saves in the same location as txt_file with .png extension.
        font_size (int, optional): Font size for text. Default=16.
        img_width (int, optional): Width of the image. Default=1000.
        padding (int, optional): Padding around text. Default=10.
    Returns:
        str: Path to the saved image file.
    """
    # Set output image file if not provided
    if img_file is None:
        img_file = txt_file.replace('.txt', '.png')

    # Read text
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # Load default font
    font = ImageFont.load_default()

    # Calculate image height
    img_height = font_size * len(lines) + 2 * padding
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)

    # Draw each line
    y = padding
    for line in lines:
        draw.text((padding, y), line.strip(), fill='black', font=font)
        y += font_size

    # Save image
    img.save(img_file)
    return img_file