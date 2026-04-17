import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import (
    GGUFQuantizationConfig,
    StableDiffusion3Pipeline,
    FluxPipeline,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
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


def initialize_qwen_image_pipeline():
    """Initializes Qwen Image Lightning pipeline."""

    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    pipe = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image",
        cache_dir="/media/t2-503-3090-3/data112/hf_cache",
        scheduler=scheduler,
        torch_dtype=torch.float16,
    )

    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning",
        weight_name="Qwen-Image-Lightning-8steps-V1.0.safetensors",
    )

    pipe.to("cuda")
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
    else:
        raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")
