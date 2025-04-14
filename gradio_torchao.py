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
import hashlib
from io import BytesIO
import json
from PIL.PngImagePlugin import PngInfo
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

# Global variable to save transformer between pipeline calls
# This is a workaround to avoid reloading the transformer model every time.
cached_tranmsformer = None

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
parser.add_argument("--t5_path", type=str, default="", help="Huggingface or local path to the T5 model.")
parser.add_argument("--quantize_llama", action="store_true", help="Quantize llama when loading, in case you're loading a non-quantized model.")
parser.add_argument("--listen", action="store_true", help="Open to LAN (run on 0.0.0.0).")
parser.add_argument("--port", type=int, default=7860, help="Port for Gradio server.")
# parser.add_argument("--disable_clip", action="store_true", help="EXPERIMENTAL: Send a blank prompt to the CLIP text encoder.")
# parser.add_argument("--disable_t5", action="store_true", help="EXPERIMENTAL: Send a blank prompt to the T5 text encoder.")
# parser.add_argument("--disable_llama", action="store_true", help="EXPERIMENTAL: Send a blank prompt to the Llama text encoder.")
parser.add_argument("--default_seed", type=int, default=-1, help="Default seed for random number generation (defaults to -1, which is random).")
parser.add_argument("--unload_transformer_before_vae", action="store_true", help="Unload transformer before VAE decoding to save memory.")
args = parser.parse_args()

# quantize_enabled depends on both the flag AND successful torchao/TorchAoConfig import
unload_transformer_before_vae = args.unload_transformer_before_vae
quantize_enabled = True
quantize_llama = args.quantize_llama
listen = args.listen
listen_port = args.port
t5_path = args.t5_path
default_seed = args.default_seed
disable_clip = True
disable_t5 = True
disable_llama = False
LLAMA_MODEL_NAME = args.llama_path

# --- Global Settings and Model Configurations ---
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

# def get_cached_encoding(model, prompt):
#     """Check if the prompt is already cached to disk."""
#     cache_dir = "prompt_cache"
#     os.makedirs(cache_dir, exist_ok=True)
#     prompt_hash = hash(prompt)
#     cache_file = os.path.join(cache_dir, f"{model}_{prompt_hash}.json")
    
#     if os.path.exists(cache_file):
#         with open(cache_file, 'r') as f:
#             cached_encoding = json.load(f)
#         return cached_encoding
#     else:
#         return None

def hash_string(string):
    if isinstance(string, list):
        string = string[0]
    elif isinstance(string, dict):
        string = json.dumps(string, sort_keys=True)
    elif isinstance(string, bytes):
        string = string.decode('utf-8')
    encoded_string = string.encode('utf-8')
    hash_object = hashlib.sha256(encoded_string)
    hex_digest = hash_object.hexdigest()
    return hex_digest

def get_cached_encoding(model, prompt):
    """Check if the prompt is already cached to disk."""
    cache_dir = "prompt_cache"
    os.makedirs(cache_dir, exist_ok=True)
    prompt_hash = hash_string(prompt)
    cache_file = os.path.join(cache_dir, f"{model}_{prompt_hash}.pt")
    print(f"Loading encoding from {cache_file}")
    
    if os.path.exists(cache_file):
        return torch.load(cache_file)
        # with open(cache_file, 'r') as f:
        #     # The encoding is a tensor, so load it with torch
        #     cached_encoding = torch.load(f)
        #     # Unpickle encoding
        #     cached_encoding = pickle.loads(cached_encoding).to("cuda")
        # return cached_encoding
    else:
        return None

def cached_encoding_exists(model, prompt):
    """Check if the prompt is already cached to disk."""
    cache_dir = "prompt_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # If propmt is a list, join it to create a unique hash
    if isinstance(prompt, list):
        prompt = " ".join(prompt)
    prompt_hash = hash_string(prompt)
    
    cache_file = os.path.join(cache_dir, f"{model}_{prompt_hash}.pt")
    if os.path.exists(cache_file):
        print(f"Cache file exists: {cache_file}")
    else:
        print(f"Cache file does not exist: {cache_file}")
    return os.path.exists(cache_file)


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
                 negative_prompt_scale: float = 1.0,
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
        global cached_tranmsformer
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
            logger.info(f"Prompt: {prompt}\nPrompt 2: {prompt_2}\nPrompt 3: {prompt_3}\nPrompt 4: {prompt_4}")
            logger.info(f"Negative Prompt: {negative_prompt}\nNegative Prompt 2: {negative_prompt_2}\nNegative Prompt 3: {negative_prompt_3}\nNegative Prompt 4: {negative_prompt_4}")
            
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
                
            # Multiply all negative prompts by negative_prompt_scale
            if negative_prompt_scale != 1 and do_classifier_free_guidance:
                # Get an embedding of a blank negative prompt
                prompt_embeds_tuple_temp = self.encode_prompt(
                    prompt="", prompt_2="", prompt_3="", prompt_4="",
                    negative_prompt="", negative_prompt_2="", negative_prompt_3="", negative_prompt_4="",
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    device=text_encoder_device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                )
                _, blank_negative_prompt_embeds_list, _, blank_negative_pooled_prompt_embeds_out = prompt_embeds_tuple_temp
              
              
                logger.info(f"Scaling negative prompt embeddings by {negative_prompt_scale}...")                
                # Iterate through negative prompt embeddings and scale them
                for i, n in enumerate(negative_prompt_embeds_list):
                    blank_n = blank_negative_prompt_embeds_list[i]
                    negative_prompt_embeds_list[i] = n * negative_prompt_scale + (1 - negative_prompt_scale) * blank_n
 
                # negative_prompt_embeds_list = [
                #     n * negative_prompt_scale + (1 - negative_prompt_scale) * blank_n
                #     for n in negative_prompt_embeds_list
                # ]
                negative_pooled_prompt_embeds_out = (
                    negative_pooled_prompt_embeds_out * negative_prompt_scale + blank_negative_pooled_prompt_embeds_out * (1 - negative_prompt_scale)
                ) if negative_pooled_prompt_embeds_out is not None else None
            else:
                logger.info("No scaling applied to negative prompt embeddings.")
                
            if do_classifier_free_guidance:
                logger.info("Performing Classifier-Free Guidance embedding concatenation...")
                final_prompt_embeds_components = []
                if negative_prompt_embeds_list is None:
                    raise ValueError("Negative prompt embeddings required for CFG but not provided.")
        
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
                if negative_prompt != "":
                    # Get an embedding of a blank negative prompt
                    logger.info(f"Negative prompt: {negative_prompt} at scale {negative_prompt_scale}...")                    
                    prompt_embeds_tuple_temp = self.encode_prompt(
                        prompt="", prompt_2="", prompt_3="", prompt_4="",
                        negative_prompt="", negative_prompt_2="", negative_prompt_3="", negative_prompt_4="",
                        do_classifier_free_guidance=do_classifier_free_guidance,
                        device=text_encoder_device,
                        num_images_per_prompt=num_images_per_prompt,
                        max_sequence_length=max_sequence_length,
                    )
                    _, blank_negative_prompt_embeds_list, _, blank_negative_pooled_prompt_embeds_out = prompt_embeds_tuple_temp
                    
                    # Subtract empty negative prompt embeddings from the actual negative prompt embeddings
                    for i, n in enumerate(negative_prompt_embeds_list):
                        blank_n = blank_negative_prompt_embeds_list[i]
                        negative_prompt_embeds_list[i] = (n - blank_n) * negative_prompt_scale
                    negative_pooled_prompt_embeds_out = (
                        negative_pooled_prompt_embeds_out - blank_negative_pooled_prompt_embeds_out) * negative_prompt_scale if negative_pooled_prompt_embeds_out is not None else None
                    
                    logger.info("Combining negative and positive prompt embeddings...")
                    final_prompt_embeds = []
                    for i, (neg, pos) in enumerate(zip(negative_prompt_embeds_list, prompt_embeds_list)):
                        final_prompt_embeds.append(pos - neg)   
                    final_pooled_prompt_embeds = (pooled_prompt_embeds_out - negative_pooled_prompt_embeds_out) if pooled_prompt_embeds_out is not None else None                                   
                            
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
        self.transformer = load_models(self.model_type, load_transformer=True, quantize=True)
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
                    
                if unload_transformer_before_vae:
                    self.transformer = None
                    cached_tranmsformer = None
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
            
        # We have an opportunity here to intercept the prompt and cache the embeddings (on a per encoder basis) and hash those prompts
        # to save time on future calls.
        
        # Check if the prompt is already cached to disk
        
        
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

def load_models(model_type, load_transformer, quantize=False, load_clip=True, load_t5=True, load_llama=True):
    """Loads models, applying quantization via TorchAoConfig if enabled."""
    global cached_tranmsformer
    config = MODEL_CONFIGS[model_type]
    pretrained_model_name_or_path = config["path"]
    logger.info(f"--- Starting Model Loading for '{model_type}' from '{pretrained_model_name_or_path}' ---")
    load_start_time = time.time()



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
        if load_t5 or load_llama:
            cached_transformer = None
            logger.info("Transformer unloaded.")
            logger.info("Clearing CUDA cache...")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
      
        logger.info("Loading tokenizers...")
        start_time = time.time()
        if load_clip:
            tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
            tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
        else:
            tokenizer = None
            tokenizer_2 = None
            
        if load_t5:
            tokenizer_3 = T5Tokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_3")
        else:
            tokenizer_3 = None
            
        if load_llama:
            try:
                tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_NAME, use_fast=False)
                tokenizer_4.pad_token = tokenizer_4.eos_token
            except OSError:
                logger.error(f"Cannot find Llama tokenizer files at '{LLAMA_MODEL_NAME}'. Please check the path.")
                raise
        else:
            tokenizer_4 = None
            
        logger.info(f"Tokenizers loaded in {time.time() - start_time:.2f}s")  
         
        progress(0, desc="Loading text encoders...")
        logger.info("--- Loading VAE and Text Encoders ---")
        start_time = time.time()
        logger.info("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae", torch_dtype=dtype
        )
        logger.info(f"VAE loaded in {time.time() - start_time:.2f}s.")

        if load_clip:
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
        else:
            text_encoder = None
            text_encoder_2 = None

        if load_t5:
            start_time = time.time()
            logger.info("Loading Text Encoder 3 (T5)...")
            if t5_path == "":
                text_encoder_3 = T5EncoderModel.from_pretrained(
                    pretrained_model_name_or_path, subfolder="text_encoder_3", torch_dtype=dtype
                )
            else:
                text_encoder_3 = T5EncoderModel.from_pretrained(
                    t5_path, torch_dtype=dtype
                )
            logger.info(f"Text Encoder 3 (T5) loaded in {time.time() - start_time:.2f}s.")
        else:
            text_encoder_3 = None

        if load_llama:
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
            text_encoder_4 = None
            
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
        cached_transformer = transformer
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

def inference_worker(prompt: str, negative_prompt: str, negative_prompt_scale: float, resolution: str, seed: int, model_type: str):
    """
    Runs inference in a child process so that GPU memory is freed after each request.
    Loads the pipeline, runs image generation, and returns the generated image as PNG bytes.
    """
    # Determine quantization flag (models must remain on GPU)
    metadata_prompt = prompt
    metadata_negative_prompt = negative_prompt
    quantize_enabled = torch.cuda.is_available() and TORCHAO_AVAILABLE

    progress(0, desc="Encoding prompts...")
    
    prompt_2 = prompt
    prompt_3 = prompt
    prompt_4 = prompt
    negative_prompt_2 = negative_prompt
    negative_prompt_3 = negative_prompt
    negative_prompt_4 = negative_prompt
    load_clip = True
    load_t5 = True
    load_llama = True
    
    if disable_clip:
        logger.info("Disabling CLIP text encoder...")
        prompt = ""
        prompt_2 = ""
        negative_prompt = ""
        negative_prompt_2 = ""

    if disable_t5:
        logger.info("Disabling T5 text encoder...")
        prompt_3 = ""
        negative_prompt_3 = ""

    if disable_llama:
        logger.info("Disabling Llama text encoder...")
        prompt_4 = ""
        negative_prompt_4 = ""

    if cached_encoding_exists("clip1", prompt) and cached_encoding_exists("clip2", prompt_2) and cached_encoding_exists("clip1", negative_prompt) and cached_encoding_exists("clip2", negative_prompt_2):
        load_clip = False
        
    if cached_encoding_exists("t5", prompt_3) and cached_encoding_exists("t5", negative_prompt_3):
        load_t5 = False
        
    if cached_encoding_exists("llama", prompt_4) and cached_encoding_exists("llama", negative_prompt_4):
        load_llama = False
    
    pipe, config = load_models(model_type, load_transformer=False, quantize=quantize_enabled, load_clip=load_clip, load_t5=load_t5, load_llama=load_llama)
    height, width = parse_resolution(resolution)
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        logger.info(f"[Worker] Generated random seed: {seed}")
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(generator_device).manual_seed(int(seed))
    logger.info(f"[Worker] Running inference with prompt: '{prompt}', seed: {seed}, resolution: ({height}, {width})")
    
    # Run the pipeline; pass along the progress callback if provided.
    image_output = pipe(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        prompt_4=prompt_4,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        negative_prompt_4=negative_prompt_4,
        negative_prompt_scale=negative_prompt_scale,
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
    
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    
    # Create metadata dictionary and embed it as JSON in the PNG
    metadata = {
        "prompt": metadata_prompt,
        "negative_prompt": metadata_negative_prompt,
        "seed": seed,
        "model": model_type,
        "llama_model": LLAMA_MODEL_NAME,
        "resolution": resolution,
        "timestamp": time_str,
    }
    png_info = PngInfo()
    png_info.add_text("parameters", json.dumps(metadata))
    
    # Save image to a BytesIO buffer and also to disk with metadata.
    buf = BytesIO()
    image.save(buf, format="PNG", pnginfo=png_info)
    os.makedirs("gradio_output", exist_ok=True)
    image.save(f"gradio_output/{time_str}_{seed}.png", format="PNG", pnginfo=png_info)
    return buf.getvalue()

  
def gradio_generate(prompt: str, negative_prompt: str, negative_prompt_scale: float, resolution: str, seed: int, model_type: str, gen_disable_clip: bool, gen_disable_t5: bool, gen_disable_llama: bool):
    """
    Gradio generation function runs inference in the main process.
    """
    global disable_clip, disable_t5, disable_llama
    disable_clip = gen_disable_clip
    disable_t5 = gen_disable_t5
    disable_llama = gen_disable_llama
    image_bytes = inference_worker(prompt, negative_prompt, negative_prompt_scale, resolution, seed, model_type)
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
        gr.Number(label="Negative Prompt Scale", value=0.6, step=0.05),
        gr.Textbox(lines=1, label="Resolution", value="1024x1024"),
        gr.Number(label="Seed (-1 for random)", value=default_seed),
        gr.Dropdown(choices=["dev", "full", "fast"], label="Model Type", value="full"),
        # Checkboxes for disabling CLIP and T5
        gr.Checkbox(label="Disable CLIP Text Encoder", value=True),
        gr.Checkbox(label="Disable T5 Text Encoder", value=True),
        gr.Checkbox(label="Disable Llama Text Encoder", value=False),
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

