import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

def run_inference(model, tokenizer, image_processor, image_file, question, inference_args):
    # Prepare the image input
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    image_size = image.size

    # Process the image
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [img.to(inference_args.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(inference_args.device, dtype=torch.float16)

    # Determine conversation mode
    model_name = get_model_name_from_path(inference_args.model_path)
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    # Prepare the prompt
    if hasattr(model.config, 'mm_use_im_start_end') and model.config.mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + question

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize the prompt with special image tokens
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors='pt'
    ).unsqueeze(0).to(inference_args.device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Set model to evaluation mode
    model.eval()

    # Run inference
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True if inference_args.temperature > 0 else False,
            temperature=inference_args.temperature,
            max_new_tokens=inference_args.max_new_tokens,
            streamer=streamer,
            use_cache=True
        )

    # Decode the outputs
    outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    conv.messages[-1][-1] = outputs

    # Set model back to training mode
    model.train()

    return outputs


import argparse
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model

# Inference arguments
inference_args = argparse.Namespace(
    device='cuda',
    temperature=0,
    max_new_tokens=512,
    model_path='/content/MyLLaVA/LLaVA/checkpoints/llava-v1.5-7b-pretrain'
)

# Disable torch init to save memory
disable_torch_init()

# Load the model
tokenizer, model, image_processor, context_len = load_pretrained_model(
    inference_args.model_path,
    'lmsys/vicuna-7b-v1.5', 
    get_model_name_from_path(inference_args.model_path),
    load_8bit=False,
    load_4bit=False,
    device=inference_args.device
)

image_file = ''
question = "How many circles in the picture?"

outputs = run_inference(model, tokenizer, image_processor, image_file, question, inference_args)
print(outputs)
