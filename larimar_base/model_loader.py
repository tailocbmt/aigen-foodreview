import os

import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration

from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import (
    AutoPipelineForText2Image,
    GGUFQuantizationConfig,
    StableDiffusion3Pipeline,
    FluxPipeline,
    QwenImageTransformer2DModel,
    QwenImagePipeline,
    ZImagePipeline,
    FluxTransformer2DModel,
    SD3Transformer2DModel
)

QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LLAMA_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MISTRAL_MODEL_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
SD_IMAGE_MODEL_ID = "https://huggingface.co/city96/stable-diffusion-3.5-large-turbo-gguf/blob/main/sd3.5_large_turbo-Q8_0.gguf"
FLUX_IMAGE_MODEL_ID = "https://huggingface.co/city96/FLUX.1-schnell-gguf/blob/main/flux1-schnell-Q8_0.gguf"
Z_IMAGE_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
QWEN_IMAGE_MODEL_ID = "https://huggingface.co/city96/Qwen-Image-gguf/blob/main/qwen-image-Q6_K.gguf"


def initialize_llama3_model(MODEL_ID: str = LLAMA_MODEL_ID):
    print("Loading LLaMA 3...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return tokenizer, model


def initialize_mixtral_model(MODEL_ID: str = MISTRAL_MODEL_ID):
    print("Loading Mixtral 8x7B...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    offload_dir = "./offload_mixtral"
    os.makedirs(offload_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        offload_folder=offload_dir,
        offload_state_dict=True,
    )

    return tokenizer, model


def initialize_qwen_model(MODEL_ID: str = QWEN_MODEL_ID):
    """Initializes a local Qwen model for caption rewriting."""
    print("Loading Qwen model locally...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return tokenizer, model


def initialize_sd_models(IMAGE_MODEL_ID: str = SD_IMAGE_MODEL_ID):
    """Initializes the SD3.5 Turbo pipeline using the Q8_0 GGUF model."""
    print("Loading the SD3.5 Large Turbo GGUF transformer...")

    transformer = SD3Transformer2DModel.from_single_file(
        IMAGE_MODEL_ID,
        quantization_config=GGUFQuantizationConfig(
            compute_dtype=torch.bfloat16
        ),
        torch_dtype=torch.bfloat16,
    )

    print("Loading the rest of the SD3.5 pipeline...")

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large-turbo",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    print("Moving model to GPU...")
    pipe.to("cuda")

    return pipe


def initialize_flux_models(IMAGE_MODEL_ID: str = FLUX_IMAGE_MODEL_ID):
    """Initializes the FLUX pipeline using the Q8_0 GGUF model."""
    print("Loading the 12.7GB GGUF transformer...")

    transformer = FluxTransformer2DModel.from_single_file(
        IMAGE_MODEL_ID,
        quantization_config=GGUFQuantizationConfig(
            compute_dtype=torch.bfloat16
        ),
        torch_dtype=torch.bfloat16,
    )

    print("Loading the rest of the FLUX pipeline...")

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    pipe.to("cuda")
    return pipe


def initialize_qwen_image_pipeline(IMAGE_MODEL_ID: str = QWEN_IMAGE_MODEL_ID):
    """Initializes Qwen Image Lightning pipeline."""

    transformer_quant_config = DiffusersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        "Qwen/Qwen-Image",
        subfolder="transformer",
        quantization_config=transformer_quant_config,
        torch_dtype=torch.bfloat16,
    )
    transformer = transformer.to("cpu")

    # Quantize the text encoder
    text_encoder_quant_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen-Image",
        subfolder="text_encoder",
        quantization_config=text_encoder_quant_config,
        torch_dtype=torch.bfloat16,
    )
    text_encoder = text_encoder.to("cpu")
    # Build the generation pipeline
    pipe = QwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image",
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
        cache_dir="/media/t2-503-3090-3/data112/hf_cache",
    )

    # Optional Lightning LoRA for faster generation
    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors",
    )

    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()

    if hasattr(pipe, "vae") and pipe.vae is not None:
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
    return pipe


def initialize_zimage_pipeline(IMAGE_MODEL_ID: str = Z_IMAGE_MODEL_ID):
    """Initializes Z-Image Turbo pipeline."""

    pipe = ZImagePipeline.from_pretrained(
        IMAGE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )

    pipe.to("cuda")
    return pipe


def initialize_sdxl_pipeline(IMAGE_MODEL_ID: str = Z_IMAGE_MODEL_ID):
    """Initializes SDXL Turbo pipeline."""

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")

    return pipe


def initialize_text_model(MODEL_NAME):
    if MODEL_NAME == "qwen":
        return initialize_qwen_model()
    elif MODEL_NAME == "llama3":
        return initialize_llama3_model()
    elif MODEL_NAME == "mixtral":
        return initialize_mixtral_model()
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")


def initialize_image_pipeline(MODEL_NAME):
    """Initializes the correct image pipeline based on MODEL_NAME."""

    if MODEL_NAME == "sd":
        return initialize_sd_models()
    elif MODEL_NAME == "flux":
        return initialize_flux_models()
    elif MODEL_NAME == "z_image":
        return initialize_zimage_pipeline()
    elif MODEL_NAME == "qwen_image":
        return initialize_qwen_image_pipeline()
    elif MODEL_NAME == "sdxl":
        return initialize_sdxl_pipeline()
    else:
        raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")
