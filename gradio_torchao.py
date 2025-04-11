import torch
import argparse
import math
import einops
import inspect
import logging
import sys
import os
import gc
import time
from typing import Any, Callable, Dict, List, Optional, Union
from contextlib import contextmanager

# --- Gradio and Multiprocessing Imports ---
import gradio as gr
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

progress = gr.Progress()

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# --- TorchAo / Diffusers Imports ---
try:
    import torchao
    from diffusers import TorchAoConfig
    logger.info(f"Successfully imported torchao version {torchao.__version__} and diffusers.TorchAoConfig.")
    TORCHAO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"torchao or TorchAoConfig import failed: {e}")
    TORCHAO_AVAILABLE = False
except Exception as e:
    logger.warning(f"Unexpected error during torchao/TorchAoConfig import: {e}")
    TORCHAO_AVAILABLE = False

# --- Diffusers/Transformers Imports ---
try:
    from hi_diffusers import HiDreamImagePipeline as BaseHiDreamImagePipeline
    from hi_diffusers import HiDreamImageTransformer2DModel
    from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
    from hi_diffusers.pipelines.hidream_image.pipeline_output import HiDreamImagePipelineOutput

    from transformers import (
        CLIPTextModelWithProjection,
        CLIPTokenizer,
        T5EncoderModel,
        T5Tokenizer,
        LlamaForCausalLM,
        LlamaConfig,
        PreTrainedTokenizerFast
    )
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.models.autoencoders import AutoencoderKL
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from diffusers.utils import is_torch_xla_available
    from diffusers.utils.torch_utils import randn_tensor
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    logger.info("Successfully imported Diffusers and Transformers components.")
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import Diffusers/Transformers components: {e}")
    sys.exit(1)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
    logger.info("Torch XLA is available.")
else:
    XLA_AVAILABLE = False
    logger.info("Torch XLA is not available.")
    
parser = argparse.ArgumentParser(description="Generate images using HiDream with optional FP8 quantization.")
parser.add_argument("--llama_path", type=str, default="RichardErkhov/NaniDAO_-_Meta-Llama-3.1-8B-Instruct-ablated-v1-8bits", help="Huggingface or local path to the Llama model, ideally INT8 quantized as FP8 will not work. Also try the non-ablated model at fsaudm/Meta-Llama-3.1-8B-Instruct-INT8")
parser.add_argument("--quantize_llama", action="store_true", help="Quantize llama when loading, in case you're loading a non-quantized model.")
parser.add_argument("--listen", action="store_true", help="Open to LAN (run on 0.0.0.0).")
parser.add_argument("--port", type=int, default=7860, help="Port for Gradio server.")
parser.add_argument("--negative_prompt_scale", type=float, default=1.0, help="Scale for negative prompt.")
args = parser.parse_args()

# quantize_enabled depends on both the flag AND successful torchao/TorchAoConfig import
negative_prompt_scale = args.negative_prompt_scale
quantize_enabled = True
quantize_llama = args.quantize_llama
listen = args.listen
listen_port = args.port
LLAMA_MODEL_NAME = args.llama_path

# --- Global Settings and Model Configurations ---
time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
MODEL_PREFIX = "HiDream-ai"  # Local path prefix or model hub prefix
MODEL_CONFIGS = {
    "dev": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Dev",
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    },
    "full": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Full",
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": FlowUniPCMultistepScheduler
    },
    "fast": {
        "path": f"{MODEL_PREFIX}/HiDream-I1-Fast",
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": FlashFlowMatchEulerDiscreteScheduler
    }
}

# --- Helper Functions ---
def calculate_shift(image_seq_len, base_seq_len: int = 256, max_seq_len: int = 4096,
                    base_shift: float = 0.5, max_shift: float = 1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b

def retrieve_timesteps(scheduler, num_inference_steps: Optional[int] = None,
                       device: Optional[Union[str, torch.device]] = None,
                       timesteps: Optional[List[int]] = None,
                       sigmas: Optional[List[float]] = None, **kwargs):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"Scheduler {scheduler.__class__} does not support `timesteps`.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_sigmas:
            raise ValueError(f"Scheduler {scheduler.__class__} does not support `sigmas`.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def parse_resolution(resolution_str):
    try:
        resolution_str = resolution_str.replace("×", "x").replace(" ", "").replace("(", "").replace(")", "")
        width, height = map(int, resolution_str.split("x"))
        return height, width  # HiDream expects (height, width)
    except Exception as e:
        logger.warning(f"Could not parse resolution '{resolution_str}': {e}. Falling back to 1024x1024.")
        return 1024, 1024

# --- Pipeline Classes ---
class HiDreamImagePipelineFP8(BaseHiDreamImagePipeline):
    # Added optional progress parameter to __call__
    @torch.no_grad()
    def __call__(self,
                 prompt: Union[str, List[str]] = None,
                 prompt_2: Optional[Union[str, List[str]]] = None,
                 prompt_3: Optional[Union[str, List[str]]] = None,
                 prompt_4: Optional[Union[str, List[str]]] = None,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 num_inference_steps: int = 50,
                 sigmas: Optional[List[float]] = None,
                 guidance_scale: float = 5.0,
                 negative_prompt: Optional[Union[str, List[str]]] = None,
                 negative_prompt_2: Optional[Union[str, List[str]]] = None,
                 negative_prompt_3: Optional[Union[str, List[str]]] = None,
                 negative_prompt_4: Optional[Union[str, List[str]]] = None,
                 num_images_per_prompt: Optional[int] = 1,
                 generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                 latents: Optional[torch.FloatTensor] = None,
                 prompt_embeds: Optional[List[torch.FloatTensor]] = None,
                 negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
                 pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
                 negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
                 output_type: Optional[str] = "pil",
                 return_dict: bool = True,
                 joint_attention_kwargs: Optional[Dict[str, Any]] = None,
                 callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
                 callback_on_step_end_tensor_inputs: List[str] = ["latents"],
                 max_sequence_length: int = 128,
                 ):
        logger.info("--- Entering Pipeline __call__ ---")
        start_call_time = time.time()
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        division = self.vae_scale_factor * 2
        height = (height // division) * division
        width = (width // division) * division

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        if prompt is not None:
            batch_size = len(prompt) if isinstance(prompt, list) else 1
        elif prompt_embeds is not None:
            if not isinstance(prompt_embeds, list) or len(prompt_embeds) == 0:
                raise ValueError("prompt_embeds must be a non-empty list for HiDream.")
            example_embed = prompt_embeds[0]
            bs_dim = 1 if len(example_embed.shape) == 4 else 0
            batch_size = example_embed.shape[bs_dim] // num_images_per_prompt
        else:
            raise ValueError("Either `prompt` or `prompt_embeds` must be provided.")

        device = self._execution_device
        text_encoder_device = device
        transformer_device = device
        vae_device = device
        logger.info(f"Target execution device: {device}")

        lora_scale = (self.joint_attention_kwargs.get("scale", None)
                      if self.joint_attention_kwargs is not None else None)
        do_classifier_free_guidance = self.guidance_scale > 0
        logger.info(f"Classifier-Free Guidance: {'Enabled' if do_classifier_free_guidance else 'Disabled'} (Scale: {guidance_scale})")

        # --- Encoding Phase ---
        logger.info("--- Starting Encoding Phase ---")
        encode_start_time = time.time()
        final_prompt_embeds = None
        final_pooled_prompt_embeds = None
        try:
            logger.info("Encoding prompts...")
            progress(0, desc="Encoding prompts...")
            prompt_embeds_tuple = self.encode_prompt(
                prompt=prompt, prompt_2=prompt_2, prompt_3=prompt_3, prompt_4=prompt_4,
                negative_prompt=negative_prompt, negative_prompt_2=negative_prompt_2,
                negative_prompt_3=negative_prompt_3, negative_prompt_4=negative_prompt_4,
                do_classifier_free_guidance=do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=text_encoder_device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            prompt_embeds_list, negative_prompt_embeds_list, pooled_prompt_embeds_out, negative_pooled_prompt_embeds_out = prompt_embeds_tuple
            logger.info("Prompt encoding complete.")
            prompt_embeds_list = [p.to(torch.bfloat16) for p in prompt_embeds_list] if prompt_embeds_list is not None else None
            negative_prompt_embeds_list = [n.to(torch.bfloat16) for n in negative_prompt_embeds_list] if negative_prompt_embeds_list is not None else None
            if pooled_prompt_embeds_out is not None:
                pooled_prompt_embeds_out = pooled_prompt_embeds_out.to(torch.bfloat16)
            if negative_pooled_prompt_embeds_out is not None:
                negative_pooled_prompt_embeds_out = negative_pooled_prompt_embeds_out.to(torch.bfloat16)
            if do_classifier_free_guidance:
                logger.info("Performing Classifier-Free Guidance embedding concatenation...")
                final_prompt_embeds_components = []
                if negative_prompt_embeds_list is None:
                    raise ValueError("Negative prompt embeddings required for CFG but not provided.")

                # Multiply all negative prompts by negative_prompt_scale
                if negative_prompt_scale != 1:
                    negative_prompt_embeds_list = [
                        n * negative_prompt_scale for n in negative_prompt_embeds_list
                    ]
                    
                for i, (neg, pos) in enumerate(zip(negative_prompt_embeds_list, prompt_embeds_list)):
                    if len(neg.shape) == 4:
                        final_prompt_embeds_components.append(torch.cat([neg, pos], dim=1))
                    elif len(neg.shape) == 3:
                        final_prompt_embeds_components.append(torch.cat([neg, pos], dim=0))
                    else:
                        raise ValueError(f"Unexpected embedding shape during CFG concat: {neg.shape}")
                
                final_prompt_embeds = final_prompt_embeds_components
                final_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds_out, pooled_prompt_embeds_out], dim=0)
            else:
                final_prompt_embeds = prompt_embeds_list
                final_pooled_prompt_embeds = pooled_prompt_embeds_out
            logger.info("Final prompt embeddings prepared.")
            encode_end_time = time.time()
            logger.info(f"--- Encoding Phase finished in {encode_end_time - encode_start_time:.2f} seconds ---")
        except Exception as e:
            logger.error(f"Error during encoding: {e}", exc_info=True)
            raise e
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.text_encoder = None
            self.text_encoder_2 = None
            self.text_encoder_3 = None
            self.text_encoder_4 = None  # Unload encoder 4 as well
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5)

        # --- Diffusion Phase ---
        self.transformer = load_models(self.model_type, load_transformer=True,
                                         quantize=True)
        logger.info("--- Starting Diffusion Phase ---")
        diffusion_start_time = time.time()
        image = None
        try:
            num_channels_latents = self.transformer.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt, num_channels_latents,
                height, width, final_pooled_prompt_embeds.dtype, transformer_device,
                generator, latents,
            )
            latent_dtype = latents.dtype
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, transformer_device)
            # --- Progress update in diffusion phase ---
            if progress is not None:
                # Use the provided Gradio progress callback for updates.
                for i, t in enumerate(timesteps):
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timesteps=t,
                        encoder_hidden_states=final_prompt_embeds,
                        pooled_embeds=final_pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
                    noise_pred = -noise_pred
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    step_output = self.scheduler.step(noise_pred, t, latents, return_dict=False)
                    latents = step_output[0]
                    if latents.dtype != latent_dtype:
                        latents = latents.to(latent_dtype)
                    progress((i + 1) / num_inference_steps, desc=f"Diffusion step {i+1} of {num_inference_steps}")
            else:
                with self.progress_bar(total=num_inference_steps) as progress_bar:
                    for i, t in enumerate(timesteps):
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timesteps=t,
                            encoder_hidden_states=final_prompt_embeds,
                            pooled_embeds=final_pooled_prompt_embeds,
                            return_dict=False,
                        )[0]
                        noise_pred = -noise_pred
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                        step_output = self.scheduler.step(noise_pred, t, latents, return_dict=False)
                        latents = step_output[0]
                        if latents.dtype != latent_dtype:
                            latents = latents.to(latent_dtype)
                        progress_bar.update()
            logger.info(f"Diffusion Phase finished in {time.time() - diffusion_start_time:.2f} seconds.")
            logger.info("--- Starting Decoding Phase ---")
            if output_type != "latent":
                self.vae.to("cuda")
                scaling_factor = getattr(self.vae.config, 'scaling_factor', 1.0)
                shift_factor = getattr(self.vae.config, 'shift_factor', 0.0)
                latents = (latents / scaling_factor) + shift_factor
                if latents.dtype != self.vae.dtype:
                    latents = latents.to(self.vae.dtype)
                    
                self.transformer = None
                logger.info("Transformer unloaded.")
                logger.info("Clearing CUDA cache...")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.info(f"CUDA memory allocated after clearing cache: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
                logger.info(f"CUDA memory reserved after clearing cache: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
                image = self.vae.decode(latents, return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)
                
                self.vae = None
                logger.info("VAE unloaded.")
                logger.info("Clearing CUDA cache...")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.info(f"CUDA memory allocated after clearing cache: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
                logger.info(f"CUDA memory reserved after clearing cache: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
            else:
                image = latents
        except Exception as e:
            logger.error(f"Error during diffusion/decoding: {e}", exc_info=True)
            raise e
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return HiDreamImagePipelineOutput(images=image)

# --- Additional Pipeline Class for Dynamic Encoder Offloading ---
class HiDreamImagePipelineDynamicLoad(HiDreamImagePipelineFP8):
    @contextmanager
    def dynamic_text_encoder_offload(self, device):
        try:
            if self.text_encoder is not None:
                self.text_encoder.to(device)
            if self.text_encoder_2 is not None:
                self.text_encoder_2.to(device)
            if self.text_encoder_3 is not None:
                self.text_encoder_3.to(device)
            yield
        except Exception as e:
            logger.error(f"Error during dynamic offload: {e}", exc_info=True)
            raise e
        finally:
            if self.text_encoder is not None:
                self.text_encoder.to("cpu")
            if self.text_encoder_2 is not None:
                self.text_encoder_2.to("cpu")
            if self.text_encoder_3 is not None:
                self.text_encoder_3.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
    def encode_prompt(self, prompt, prompt_2, prompt_3, prompt_4, device: Optional[torch.device] = None,
                      dtype: Optional[torch.dtype] = None, num_images_per_prompt: int = 1,
                      do_classifier_free_guidance: bool = True, negative_prompt: Optional[Union[str, List[str]]] = None,
                      negative_prompt_2: Optional[Union[str, List[str]]] = None, negative_prompt_3: Optional[Union[str, List[str]]] = None,
                      negative_prompt_4: Optional[Union[str, List[str]]] = None,
                      prompt_embeds: Optional[List[torch.FloatTensor]] = None,
                      negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                      pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
                      negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
                      max_sequence_length: int = 128, lora_scale: Optional[float] = None):
        if device is None:
            device = self._execution_device
        with self.dynamic_text_encoder_offload(device):
            (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds,
             negative_pooled_prompt_embeds) = super().encode_prompt(
                 prompt=prompt, prompt_2=prompt_2, prompt_3=prompt_3, prompt_4=prompt_4,
                 device=device, dtype=dtype, num_images_per_prompt=num_images_per_prompt,
                 do_classifier_free_guidance=do_classifier_free_guidance,
                 negative_prompt=negative_prompt, negative_prompt_2=negative_prompt_2,
                 negative_prompt_3=negative_prompt_3, negative_prompt_4=negative_prompt_4,
                 prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                 pooled_prompt_embeds=pooled_prompt_embeds,
                 negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                 max_sequence_length=max_sequence_length, lora_scale=lora_scale,
             )
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

def load_models(model_type, load_transformer, quantize=False):
    """Loads models, applying quantization via TorchAoConfig if enabled."""
    config = MODEL_CONFIGS[model_type]
    pretrained_model_name_or_path = config["path"]
    logger.info(f"--- Starting Model Loading for '{model_type}' from '{pretrained_model_name_or_path}' ---")
    load_start_time = time.time()

    logger.info("Loading tokenizers...")
    start_time = time.time()
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
    tokenizer_3 = T5Tokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_3")
    try:
        tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_NAME, use_fast=False)
        tokenizer_4.pad_token = tokenizer_4.eos_token
    except OSError:
        logger.error(f"Cannot find Llama tokenizer files at '{LLAMA_MODEL_NAME}'. Please check the path.")
        raise
    logger.info(f"Tokenizers loaded in {time.time() - start_time:.2f}s")

    dtype = torch.bfloat16
    logger.info(f"Using base dtype: {dtype}")

    # --- Define Quantization Config (if enabled) ---
    quantization_config = None
    transformer_quant_status = "without quantization"
    if quantize:
        if not TORCHAO_AVAILABLE:
             logger.warning("Quantization requested but torchao/TorchAoConfig unavailable. Skipping quantization for Transformer.")
        else:
            fp8_quant_type = "float8wo_e4m3"  # Default to weight-only E4M3
            logger.info(f"Defining TorchAoConfig for Transformer quantization type: {fp8_quant_type}")
            try:
                quantization_config = TorchAoConfig(fp8_quant_type)
                transformer_quant_status = f"with TorchAoConfig '{fp8_quant_type}' quantization"
                logger.info("TorchAoConfig created successfully for Transformer.")
            except Exception as e:
                 logger.error(f"Failed to create TorchAoConfig: {e}. Disabling quantization for Transformer.", exc_info=True)
                 quantization_config = None
                 global quantize_enabled
                 quantize_enabled = False

    if not load_transformer:
        progress(0, desc="Loading text encoders...")
        logger.info("--- Loading VAE and Text Encoders ---")
        start_time = time.time()
        logger.info("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae", torch_dtype=dtype
        )
        logger.info(f"VAE loaded in {time.time() - start_time:.2f}s.")

        start_time = time.time()
        logger.info("Loading Text Encoder 1 (CLIP)...")
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=dtype
        )
        logger.info(f"Text Encoder 1 (CLIP) loaded in {time.time() - start_time:.2f}s.")

        start_time = time.time()
        logger.info("Loading Text Encoder 2 (CLIP)...")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=dtype
        )
        logger.info(f"Text Encoder 2 (CLIP) loaded in {time.time() - start_time:.2f}s.")

        start_time = time.time()
        logger.info("Loading Text Encoder 3 (T5)...")
        text_encoder_3 = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder_3", torch_dtype=dtype
        )
        logger.info(f"Text Encoder 3 (T5) loaded in {time.time() - start_time:.2f}s.")

        start_time = time.time()
        logger.info(f"Loading Text Encoder 4 (Llama) from {LLAMA_MODEL_NAME} without explicit quantization config...")
        if quantize_llama:
            text_encoder_4 = LlamaForCausalLM.from_pretrained(
                LLAMA_MODEL_NAME, 
                output_hidden_states=True,
                output_attentions=True,
                torch_dtype=dtype,
                device_map="auto",
                load_in_8bit=True, 
            )
        else:
            text_encoder_4 = LlamaForCausalLM.from_pretrained(
                LLAMA_MODEL_NAME, 
                output_hidden_states=True,
                output_attentions=True,
                torch_dtype=dtype,
                device_map="auto",
            )
            
    else:
        start_time = time.time()
        logger.info(f"--- Loading Transformer {transformer_quant_status} ---")
        progress(0, desc="Loading transformer...")

        cache_dir = "model_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"transformer_{model_type}.safetensors")
        
        if not os.path.exists(cache_file):
            progress(0, desc="Cached transformer not found... Downloading and quantizing...")
            logger.info(f"Cached quantized transformer at {cache_file} not found... Loading and quantizing...")
            transformer = HiDreamImageTransformer2DModel.from_pretrained(
                "HiDream-ai/HiDream-I1-Dev", 
                subfolder="transformer", 
                torch_dtype=torch.bfloat16,
                quantization_config=TorchAoConfig("float8wo_e4m3")).to("cuda")
            
            torch.save(transformer.state_dict(), cache_file)
        else:
            logger.info(f"Loading quantized transformer from cache at {cache_file}...")
            with torch.device("meta"):
                transformer = HiDreamImageTransformer2DModel.from_config("HiDream-ai/HiDream-I1-Dev", subfolder="transformer").eval().to(torch.bfloat16)
                state_dict = torch.load(cache_file)
                transformer.load_state_dict(state_dict, assign=True)

        logger.info(f"Transformer loaded in {time.time() - start_time:.2f}s.")
        return transformer

    start_time = time.time()
    logger.info("Initializing Scheduler...")
    progress(0, desc="Initializing Scheduler...")
    scheduler_class = config["scheduler"]
    scheduler_config_params = {}
    if issubclass(scheduler_class, FlowUniPCMultistepScheduler):
         scheduler_config_params = {"num_train_timesteps": 1000, "shift": config["shift"], "use_dynamic_shifting": False}
         logger.info(f"Configuring {scheduler_class.__name__} with: {scheduler_config_params}")

    scheduler = scheduler_class(**scheduler_config_params)
    logger.info(f"Scheduler {scheduler_class.__name__} initialized in {time.time() - start_time:.2f}s.")

    start_time = time.time()
    logger.info("Initializing HiDreamImagePipelineFP8 (memory-managed pipeline)...")
    progress(0, desc="Initializing Pipeline...")
    try:
        pipe = HiDreamImagePipelineDynamicLoad(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            text_encoder_3=text_encoder_3,
            tokenizer_3=tokenizer_3,
            text_encoder_4=text_encoder_4,
            tokenizer_4=tokenizer_4,
        )
        text_encoder_4 = None
        pipe.model_type = model_type

        pipe.to("cuda")  # This sets pipe._execution_device, doesn't move models yet

        logger.info(f"Pipeline initialized in {time.time() - start_time:.2f}s.")
    except Exception as e:
        logger.error(f"Error initializing pipeline: {e}", exc_info=True)
        raise

    logger.info(f"--- Model Loading finished in {time.time() - load_start_time:.2f} seconds ---")
    return pipe, config

# --- Inference Worker Function ---
def inference_worker(prompt: str, negative_prompt: str, resolution: str, seed: int, model_type: str, progress=None):
    """
    Runs inference in a child process so that GPU memory is freed after each request.
    Loads the pipeline, runs image generation, and returns the generated image as PNG bytes.
    """
    quantize_enabled = torch.cuda.is_available() and TORCHAO_AVAILABLE
    pipe, config = load_models(model_type, load_transformer=False, quantize=quantize_enabled)
    height, width = parse_resolution(resolution)
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        logger.info(f"[Worker] Generated random seed: {seed}")
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(generator_device).manual_seed(int(seed))
    logger.info(f"[Worker] Running inference with prompt: '{prompt}', seed: {seed}, resolution: ({height}, {width})")
    # Pass along the progress callback to the pipeline call.
    image_output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt,
        negative_prompt_3=negative_prompt,
        negative_prompt_4=negative_prompt,
        height=height,
        width=width,
        guidance_scale=config["guidance_scale"],
        num_inference_steps=config["num_inference_steps"],
        num_images_per_prompt=1,
        generator=generator,
        output_type="pil",
        return_dict=True,
    )
    image = image_output.images[0]
    from io import BytesIO
    buf = BytesIO()
    image.save(buf, format="PNG")
    os.makedirs("gradio_output", exist_ok=True)
    image.save(f"gradio_output/{time_str}_{seed}.png", format="PNG")
    return buf.getvalue()
  
def gradio_generate(prompt: str, negative_prompt: str, resolution: str, seed: int, model_type: str):
    """
    Gradio generation function runs inference in the main process.
    """
    image_bytes = inference_worker(prompt, negative_prompt, resolution, seed, model_type)
    from io import BytesIO
    from PIL import Image
    image = Image.open(BytesIO(image_bytes))
    return image

# --- Gradio Interface Setup ---
iface = gr.Interface(
    fn=gradio_generate,
    inputs=[
        gr.Textbox(lines=2, label="Prompt", placeholder="Enter your prompt here..."),
        gr.Textbox(lines=2, label="Negative Prompt", placeholder="Optional negative prompt..."),
        gr.Textbox(lines=1, label="Resolution", value="1024x1024"),
        gr.Number(label="Seed (-1 for random)", value=-1),
        gr.Dropdown(choices=["dev", "full", "fast"], label="Model Type", value="full")
    ],
    outputs=gr.Image(label="Generated Image"),
    title="HiDream Image Generator",
    description="Generate images using the HiDream pipeline with FP8 quantization. Inference runs in a separate process to free GPU memory between requests."
)

if __name__ == "__main__":
    if listen:
        iface.launch(server_name="0.0.0.0", server_port=listen_port)
    else:
        iface.launch(server_port=listen_port)
    logger.info("Gradio interface launched.")
