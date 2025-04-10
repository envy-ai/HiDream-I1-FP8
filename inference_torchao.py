import torch
import argparse
import math
import einops
import inspect
import logging
import sys
import gc
import time # Import time for more granular timing if needed
from typing import Any, Callable, Dict, List, Optional, Union
from contextlib import contextmanager

# --- Configure Logging ---
# Moved logging config to the top to be available immediately
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Config ---


# --- torchao / Diffusers Imports ---
try:
    # Import torchao itself to check version/availability if needed
    import torchao
    # The key import for the new method:
    from diffusers import TorchAoConfig
    logger.info(f"Successfully imported torchao version {torchao.__version__} and diffusers.TorchAoConfig.")
    TORCHAO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"torchao or TorchAoConfig import failed: {e}")
    # Don't exit yet, but log the error and disable quantization
    import traceback
    traceback.print_exc()
    logger.warning("torchao/TorchAoConfig not found or failed import. FP8 quantization will be skipped.")
    TORCHAO_AVAILABLE = False
except Exception as e: # Catch other potential errors during import
     logger.warning(f"An unexpected error occurred during torchao/TorchAoConfig import: {e}")
     import traceback
     traceback.print_exc()
     logger.warning("torchao/TorchAoConfig failed import. FP8 quantization will be skipped.")
     TORCHAO_AVAILABLE = False
# --- End Imports ---


# --- Diffusers/Transformers Imports ---
# Wrap these in try-except as well, since the flash-attn error can occur here
try:
    from hi_diffusers import HiDreamImagePipeline as BaseHiDreamImagePipeline # Renamed to avoid clash
    from hi_diffusers import HiDreamImageTransformer2DModel
    from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
    from hi_diffusers.pipelines.hidream_image.pipeline_output import HiDreamImagePipelineOutput # Import the output class

    from transformers import (
        CLIPTextModelWithProjection,
        CLIPTokenizer,
        T5EncoderModel,
        T5Tokenizer,
        LlamaForCausalLM,
        PreTrainedTokenizerFast
    )
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.loaders import FromSingleFileMixin
    from diffusers.models.autoencoders import AutoencoderKL
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler # Base class for typing
    from diffusers.utils import (
        USE_PEFT_BACKEND,
        is_torch_xla_available,
        # Use diffusers internal logging if desired, or stick to Python's logging
        # logging as diffusers_logging,
    )
    from diffusers.utils.torch_utils import randn_tensor
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    logger.info("Successfully imported Diffusers and Transformers components.")
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import Diffusers/Transformers components: {e}")
    logger.error("This might be due to the flash-attn/PyTorch conflict mentioned earlier.")
    logger.error("Please ensure your environment is correctly set up (PyTorch nightly + reinstalled flash-attn).")
    import traceback
    traceback.print_exc()
    DIFFUSERS_AVAILABLE = False
    # Exit if core components are missing
    logger.critical("Core Diffusers/Transformers components failed to import. Exiting.")
    import sys
    sys.exit(1)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
    logger.info("Torch XLA is available.")
else:
    XLA_AVAILABLE = False
    logger.info("Torch XLA is not available.")
# --- End Diffusers/Transformers Imports ---


# --- Argument Parsing and Configs ---
parser = argparse.ArgumentParser(description="Generate images using HiDream with optional FP8 quantization.")
parser.add_argument("--model_type", type=str, default="full", choices=["dev", "full", "fast"], help="Specify HiDream model variant.")
parser.add_argument("--llama_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Path to the Llama model directory.")
parser.add_argument("--output_file", type=str, default="output_fp8.png", help="Filename for the generated output image.")
parser.add_argument("--prompt", type=str, default="A majestic rainbow-colored llama surfing on a cosmic wave, digital art, high detail.", help="Text prompt for image generation.")
parser.add_argument("--resolution", type=str, default="1024x1024", help="Target image resolution (1024x1024 by default).")
parser.add_argument("--seed", type=int, default=12345, help="Seed for random number generator. Use -1 for random.")
args = parser.parse_args()

model_type = args.model_type
# quantize_enabled depends on both the flag AND successful torchao/TorchAoConfig import
quantize_enabled = True
LLAMA_MODEL_NAME = args.llama_path
OUTPUT_FILENAME = args.output_file
# --- End Argument Parsing ---


MODEL_PREFIX = "HiDream-ai" # Or your local path prefix

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
# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len, base_seq_len: int = 256, max_seq_len: int = 4096,
    base_shift: float = 0.5, max_shift: float = 1.15,
):
    """Calculates shift parameter for schedulers based on sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler, num_inference_steps: Optional[int] = None, device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None, sigmas: Optional[List[float]] = None, **kwargs,
):
    """Retrieves timesteps from the scheduler based on arguments."""
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
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(f"Scheduler {scheduler.__class__} does not support `sigmas`.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def parse_resolution(resolution_str):
    """Parses resolution string like '1024×1024' into (height, width)."""
    try:
        # Normalize resolution string to remove non-standard characters
        resolution_str = resolution_str.replace("×", "x").replace(" ", "").replace("(", "").replace(")", "")
        # Split by 'x' and convert to integers
        width, height = map(int, resolution_str.split("x"))

        # HiDream pipeline expects (height, width)
        # The resolution strings seem to be WxH format.
        # Let's return (height, width) as expected by the pipeline's prepare_latents.
        return height, width
    except Exception as e:
        logger.warning(f"Could not parse resolution '{resolution_str}': {e}. Falling back to 1024x1024.", exc_info=True)
        return 1024, 1024
# --- End Helper Functions ---


# --- Modified HiDreamImagePipeline with Memory Management ---
# This class inherits from the original and overrides __call__ to
# manage moving models between CPU and GPU during generation.
class HiDreamImagePipelineFP8(BaseHiDreamImagePipeline):

    # Assuming the __init__ from BaseHiDreamImagePipeline is compatible
    # with receiving potentially quantized models loaded via TorchAoConfig.
    # If BaseHiDreamImagePipeline needs specific initialization for quantized models,
    # override __init__ here as well.

    @torch.no_grad()
    def __call__(
        self,
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
        prompt_embeds: Optional[List[torch.FloatTensor]] = None, # HiDream uses a list
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None, # HiDream uses a list
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
        # --- Start Memory Management & Setup ---
        start_call_time = time.time()
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # Resolution scaling logic (as in original pipeline)
        division = self.vae_scale_factor * 2 # Patch size * VAE factor
        default_pixel_height = self.default_sample_size * self.vae_scale_factor
        default_pixel_width = self.default_sample_size * self.vae_scale_factor
        S_max = default_pixel_height * default_pixel_width

        if width * height == 0:
            logger.warning(f"Input width or height is zero ({width=}, {height=}). Resetting to default.")
            width = default_pixel_width
            height = default_pixel_height

        scale = math.sqrt(S_max / (width * height))
        target_h = int(height * scale)
        target_w = int(width * scale)
        height = (target_h // division) * division
        width = (target_w // division) * division
        if height == 0 or width == 0:
             logger.warning(f"Calculated zero dimension after scaling ({width=}, {height=}). Check input resolution and VAE scale factor. Resetting to default.")
             height = default_pixel_height
             width = default_pixel_width
        logger.info(f"Pipeline calculating image size: {width}x{height} (after scaling/alignment)")


        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # Determine batch size
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
        logger.info(f"Determined batch size: {batch_size}")


        device = self._execution_device
        # Assuming all text encoders run on the same device temporarily
        text_encoder_device = device
        transformer_device = device
        vae_device = device # VAE also needs to be moved
        logger.info(f"Target execution device: {device}")

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        do_classifier_free_guidance = self.guidance_scale > 0 # Simplified CFG check
        logger.info(f"Classifier-Free Guidance: {'Enabled' if do_classifier_free_guidance else 'Disabled'} (Scale: {guidance_scale})")
        # --- End Setup ---

        # --- Encoding Phase ---
        logger.info("--- Starting Encoding Phase ---")
        encode_start_time = time.time()
        final_prompt_embeds = None
        final_pooled_prompt_embeds = None
        try:
            '''
            logger.info(f"Moving text encoders to target device: {text_encoder_device} for prompt encoding...")
            # Move necessary encoders to GPU
            if hasattr(self, "text_encoder") and self.text_encoder is not None:
                logger.info("  Moving Text Encoder 1 (CLIP)...")
                self.text_encoder.to(text_encoder_device)
                logger.info("  Text Encoder 1 (CLIP) moved.")
            if hasattr(self, "text_encoder_2") and self.text_encoder_2 is not None:
                logger.info("  Moving Text Encoder 2 (CLIP)...")
                self.text_encoder_2.to(text_encoder_device)
                logger.info("  Text Encoder 2 (CLIP) moved.")
            if hasattr(self, "text_encoder_3") and self.text_encoder_3 is not None:
                logger.info("  Moving Text Encoder 3 (T5)...")
                self.text_encoder_3.to(text_encoder_device)
                logger.info("  Text Encoder 3 (T5) moved.")
            if hasattr(self, "text_encoder_4") and self.text_encoder_4 is not None:
                logger.info("  Moving Text Encoder 4 (Llama)...")
                self.text_encoder_4.to(text_encoder_device)
                logger.info("  Text Encoder 4 (Llama) moved.")

            if torch.cuda.is_available():
                logger.info(f"CUDA memory allocated after moving encoders: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
                logger.info(f"CUDA memory reserved after moving encoders: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
            '''
            #logger.info("Offloading transformer from GPU to CPU for prompt encoding...")
            #self.transformer.to("cpu")
            #torch.cuda.empty_cache()
            logger.info("Encoding prompts...")

            # Call encode_prompt only if embeds aren't pre-provided
            prompt_embeds_tuple = self.encode_prompt(
                prompt=prompt, prompt_2=prompt_2, prompt_3=prompt_3, prompt_4=prompt_4,
                negative_prompt=negative_prompt, negative_prompt_2=negative_prompt_2,
                negative_prompt_3=negative_prompt_3, negative_prompt_4=negative_prompt_4,
                do_classifier_free_guidance=do_classifier_free_guidance,
                prompt_embeds=prompt_embeds, # Pass provided embeds if any
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=text_encoder_device, # Specify device for encoding
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            prompt_embeds_list, negative_prompt_embeds_list, pooled_prompt_embeds_out, negative_pooled_prompt_embeds_out = prompt_embeds_tuple
            logger.info("Prompt encoding calculation complete.")

            # Combine for CFG if enabled
            if do_classifier_free_guidance:
                logger.info("Performing Classifier-Free Guidance embedding concatenation...")
                final_prompt_embeds_components = []
                if negative_prompt_embeds_list is None:
                     raise ValueError("Negative prompt embeddings are required for CFG but were not generated/provided.")

                for i, (neg, pos) in enumerate(zip(negative_prompt_embeds_list, prompt_embeds_list)):
                    logger.debug(f"  Concatenating embeds component {i+1}: neg shape {neg.shape}, pos shape {pos.shape}")
                    if len(neg.shape) == 4: # Llama-like (layers, batch*num_img, seq, dim)
                         final_prompt_embeds_components.append(torch.cat([neg, pos], dim=1))
                    elif len(neg.shape) == 3: # T5-like (batch*num_img, seq, dim)
                         final_prompt_embeds_components.append(torch.cat([neg, pos], dim=0))
                    else:
                         raise ValueError(f"Unexpected embedding shape during CFG concat: {neg.shape}")
                final_prompt_embeds = final_prompt_embeds_components

                if negative_pooled_prompt_embeds_out is None:
                     raise ValueError("Negative pooled prompt embeddings are required for CFG but were not generated/provided.")
                logger.debug(f"  Concatenating pooled embeds: neg shape {negative_pooled_prompt_embeds_out.shape}, pos shape {pooled_prompt_embeds_out.shape}")
                final_pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds_out, pooled_prompt_embeds_out], dim=0)
                logger.info("CFG concatenation finished.")
            else:
                final_prompt_embeds = prompt_embeds_list
                final_pooled_prompt_embeds = pooled_prompt_embeds_out

            # Ensure final embeds are on the transformer's target device before clearing encoders
            #logger.info(f"Moving final prompt embeddings to target device: {transformer_device}...")
            #final_prompt_embeds = [p.to(transformer_device) for p in final_prompt_embeds]
            #final_pooled_prompt_embeds = final_pooled_prompt_embeds.to(transformer_device)
            logger.info("Final prompt embeddings moved.")

            encode_end_time = time.time()
            logger.info(f"--- Encoding Phase finished in {encode_end_time - encode_start_time:.2f} seconds ---")
        except Exception as e:
            logger.error(f"Error during encoding phase: {e}", exc_info=True)
            # Optionally re-raise or handle the exception as needed
            raise e
        finally:
            # Move encoders back to CPU regardless of success/failure
            # logger.info("Moving text encoders back to CPU...")
            # if hasattr(self, "text_encoder") and self.text_encoder is not None:
            #     logger.info("  Moving Text Encoder 1 (CLIP) to CPU...")
            #     self.text_encoder.to("cpu")
            #     logger.info("  Text Encoder 1 (CLIP) moved to CPU.")
            # if hasattr(self, "text_encoder_2") and self.text_encoder_2 is not None:
            #     logger.info("  Moving Text Encoder 2 (CLIP) to CPU...")
            #     self.text_encoder_2.to("cpu")
            #     logger.info("  Text Encoder 2 (CLIP) moved to CPU.")
            # if hasattr(self, "text_encoder_3") and self.text_encoder_3 is not None:
            #     logger.info("  Moving Text Encoder 3 (T5) to CPU...")
            #     self.text_encoder_3.to("cpu")
            #     logger.info("  Text Encoder 3 (T5) moved to CPU.")
            # if hasattr(self, "text_encoder_4") and self.text_encoder_4 is not None:
            #     logger.info("  Moving Text Encoder 4 (Llama) to CPU...")
            #     self.text_encoder_4.to("cpu")
            #     logger.info("  Text Encoder 4 (Llama) moved to CPU.")
            
            # Clear VRAM potentially occupied by encoders
            if torch.cuda.is_available():
                logger.info("Clearing CUDA cache after encoding...")
                torch.cuda.empty_cache()
                logger.info(f"CUDA memory allocated after clearing cache: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
                logger.info(f"CUDA memory reserved after clearing cache: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

            logger.info("Skipping manual move.")
            # Unload all text encoders
            self.text_encoder = None
            self.text_encoder_2 = None
            self.text_encoder_3 = None
            self.text_encoder_4 = None
            logger.info("Text encoders unloaded.")
            # Clear cache, etc
            if torch.cuda.is_available():
                logger.info("Clearing CUDA cache after encoding...")
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.info(f"CUDA memory allocated after clearing cache: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
                logger.info(f"CUDA memory reserved after clearing cache: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
            # Sleep 5 seconds
            time.sleep(5) # Optional sleep for debugging
            
            
        # --- End Encoding Phase ---

        # --- Diffusion Phase ---
        # Load transformer
        self.transformer = load_models(model_type, load_transformer=True, quantize=quantize_enabled)

        logger.info("--- Starting Diffusion Phase ---")
        diffusion_start_time = time.time()
        image = None
        try:
            logger.info(f"Moving transformer to target device: {transformer_device}...")
            if not hasattr(self, 'transformer') or self.transformer is None: raise ValueError("Transformer not loaded")
            #self.transformer.to(transformer_device)
            logger.info("Transformer moved.")
            if torch.cuda.is_available():
                logger.info(f"CUDA memory allocated after moving transformer: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
                logger.info(f"CUDA memory reserved after moving transformer: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

            # VAE is needed for decoding, move it later, just before decoding

            # 4. Prepare latent variables
            logger.info("Preparing initial latents...")
            num_channels_latents = self.transformer.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt, num_channels_latents, height, width,
                final_pooled_prompt_embeds.dtype, transformer_device, generator, latents,
            )
            latent_dtype = latents.dtype # Store original dtype
            logger.info(f"Prepared initial latents on {latents.device} with shape {latents.shape} and dtype {latents.dtype}")

            # Aspect ratio embedding prep (as in original pipeline)
            img_sizes = img_ids = None
            latent_height, latent_width = latents.shape[-2:]
            if latent_height != latent_width:
                logger.info("Preparing aspect ratio embeddings...")
                actual_batch_size_for_ar = (batch_size * num_images_per_prompt) * 2 if do_classifier_free_guidance else (batch_size * num_images_per_prompt)

                patch_size = self.transformer.config.patch_size
                pH, pW = latent_height // patch_size, latent_width // patch_size

                img_sizes_single = torch.tensor([pH, pW], dtype=torch.long, device=transformer_device)
                img_ids_single = torch.zeros(pH, pW, 3, device=transformer_device, dtype=torch.long) # Use long for IDs? Check transformer req.
                img_ids_single[..., 1] += torch.arange(pH, device=transformer_device)[:, None]
                img_ids_single[..., 2] += torch.arange(pW, device=transformer_device)[None, :]
                img_ids_single = img_ids_single.reshape(pH * pW, -1)

                # Pad if needed
                max_seq = getattr(self.transformer.config, 'max_seq_len', getattr(self.transformer, 'max_seq', 4096))
                if pH * pW < max_seq:
                    img_ids_pad = torch.zeros(max_seq, 3, device=transformer_device, dtype=img_ids_single.dtype)
                    img_ids_pad[:pH*pW, :] = img_ids_single
                    img_ids_single = img_ids_pad
                elif pH * pW > max_seq:
                    logger.warning(f"Latent patch count {pH*pW} exceeds transformer max sequence {max_seq}. Truncation might occur.")
                    img_ids_single = img_ids_single[:max_seq, :]


                # Repeat for batch size
                img_sizes = img_sizes_single.unsqueeze(0).repeat(actual_batch_size_for_ar, 1)
                img_ids = img_ids_single.unsqueeze(0).repeat(actual_batch_size_for_ar, 1, 1)
                logger.info(f"Prepared aspect ratio embeddings for size {latent_height}x{latent_width} (patched: {pH}x{pW}), repeated to batch {actual_batch_size_for_ar}")
            else:
                logger.info("Using square aspect ratio, skipping aspect ratio embedding preparation.")


            # 5. Prepare timesteps
            logger.info("Preparing scheduler and timesteps...")
            max_seq_for_shift = getattr(self.transformer.config, 'max_seq_len', getattr(self.transformer, 'max_seq', 4096))
            mu = calculate_shift(max_seq_for_shift)
            logger.info(f"Calculated scheduler shift parameter (mu): {mu:.4f} for max_seq_len {max_seq_for_shift}")
            scheduler_kwargs = {}
            if isinstance(self.scheduler, FlowUniPCMultistepScheduler):
                 effective_shift = math.exp(mu) if hasattr(self.scheduler, 'shift') else None
                 if effective_shift is not None and hasattr(self.scheduler, 'shift') and self.scheduler.shift != effective_shift:
                     logger.warning(f"Overriding pre-configured scheduler shift {self.scheduler.shift} with calculated {effective_shift}")
                     self.scheduler.shift = effective_shift
                 self.scheduler.set_timesteps(num_inference_steps, device=transformer_device)
                 timesteps = self.scheduler.timesteps
            elif isinstance(self.scheduler, (FlowMatchEulerDiscreteScheduler, FlashFlowMatchEulerDiscreteScheduler)):
                 scheduler_kwargs["mu"] = mu
                 timesteps, num_inference_steps = retrieve_timesteps(
                     self.scheduler, num_inference_steps, transformer_device, sigmas=sigmas, **scheduler_kwargs,
                 )
            else: # Generic fallback
                 timesteps, num_inference_steps = retrieve_timesteps(
                     self.scheduler, num_inference_steps, transformer_device, sigmas=sigmas
                 )

            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0) if hasattr(self.scheduler, 'order') else 0
            self._num_timesteps = len(timesteps)
            logger.info(f"Using scheduler {self.scheduler.__class__.__name__} with {num_inference_steps} steps. Timesteps range: [{timesteps[0]}, ..., {timesteps[-1]}] on device {timesteps.device}")


            # 6. Denoising loop
            logger.info("Starting diffusion loop...")
            loop_start_time = time.time()
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    step_start_time = time.time()
                    if self.interrupt:
                        logger.warning(f"Interrupt flag set at step {i}. Stopping diffusion loop.")
                        continue

                    # Expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    #timestep = t.expand(latent_model_input.shape[0]).to(transformer_device) # Ensure timestep on device

                    # Prepare transformer input
                    current_latent_input = latent_model_input
                    if img_sizes is not None and img_ids is not None:
                        B, C, H, W = current_latent_input.shape
                        patch_size = self.transformer.config.patch_size
                        pH, pW = H // patch_size, W // patch_size
                        current_latent_input = einops.rearrange(current_latent_input, 'B C (H p1) (W p2) -> B C (H W) (p1 p2)', p1=patch_size, p2=patch_size)
                        max_seq_current = getattr(self.transformer.config, 'max_seq_len', getattr(self.transformer, 'max_seq', 4096))
                        if pH * pW < max_seq_current:
                           padded_input = torch.zeros(
                               (B, C, max_seq_current, patch_size * patch_size),
                               dtype=current_latent_input.dtype, device=current_latent_input.device
                           )
                           padded_input[:, :, :pH*pW] = current_latent_input
                           current_latent_input = padded_input
                        elif pH * pW > max_seq_current:
                             logger.warning(f"Input sequence length {pH*pW} > {max_seq_current}. Truncating input for transformer.")
                             current_latent_input = current_latent_input[:, :, :max_seq_current, :]

                    # Predict noise/flow
                    logger.debug(f"Step {i}, Time {t:.4f}, Input shape: {current_latent_input.shape}, Embed shape: {[e.shape for e in final_prompt_embeds]}, Pooled shape: {final_pooled_prompt_embeds.shape}")
                    noise_pred = self.transformer(
                        hidden_states = current_latent_input,
                        timesteps = t,
                        encoder_hidden_states = final_prompt_embeds, # Use the CFG-combined embeds
                        pooled_embeds = final_pooled_prompt_embeds, # Use the CFG-combined embeds
                        img_sizes = img_sizes, # Pass AR embeds
                        img_ids = img_ids,     # Pass AR embeds
                        return_dict = False,
                    )[0]
                    noise_pred = -noise_pred # HiDream predicts velocity

                    # Unpad the prediction if padding was applied
                    if img_sizes is not None and img_ids is not None:
                        max_seq_pred = getattr(self.transformer.config, 'max_seq_len', getattr(self.transformer, 'max_seq', 4096))
                        if noise_pred.shape[2] == max_seq_pred: # Check if output sequence length matches max_seq
                            pH_orig = img_sizes[0, 0].item()
                            pW_orig = img_sizes[0, 1].item()
                            actual_seq_len = pH_orig * pW_orig
                            if actual_seq_len < max_seq_pred:
                                noise_pred = noise_pred[:, :, :actual_seq_len, :]

                    # Perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # Reshape back to image latent format
                    if len(noise_pred.shape) == 4 and noise_pred.shape[2] != latent_height:
                         patch_size = self.transformer.config.patch_size
                         pH_expect = latent_height // patch_size
                         pW_expect = latent_width // patch_size
                         expected_seq_len = pH_expect * pW_expect
                         if noise_pred.shape[2] == expected_seq_len:
                             noise_pred = einops.rearrange(noise_pred, 'B C (H W) (p1 p2) -> B C (H p1) (W p2)',
                                                           H=pH_expect, W=pW_expect, p1=patch_size, p2=patch_size)
                         else:
                              logger.warning(f"Noise prediction sequence length {noise_pred.shape[2]} doesn't match expected {expected_seq_len} for reshape. Skipping reshape.")

                    # Scheduler step
                    step_output = self.scheduler.step(noise_pred, t, latents, return_dict=False)
                    latents = step_output[0]

                    # Cast latents back to original dtype if necessary
                    if latents.dtype != latent_dtype:
                        latents = latents.to(latent_dtype)

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_on_step_end(self, i, t, callback_kwargs)

                    step_end_time = time.time()
                    logger.debug(f"Step {i+1}/{num_inference_steps} completed in {step_end_time - step_start_time:.3f}s (Timestep: {t:.4f})")

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                         progress_bar.update()
                    if XLA_AVAILABLE: xm.mark_step()

            loop_end_time = time.time()
            logger.info(f"Diffusion loop finished in {loop_end_time - loop_start_time:.2f} seconds.")

            # --- Decoding ---
            logger.info("--- Starting Decoding Phase ---")
            decode_start_time = time.time()

            logger.info(f"Moving VAE to target device: {vae_device} for decoding...")
            if not hasattr(self, 'vae') or self.vae is None: raise ValueError("VAE not loaded")
            #self.vae.to(vae_device)
            logger.info("VAE moved.")
            if torch.cuda.is_available():
                logger.info(f"CUDA memory allocated after moving VAE: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
                logger.info(f"CUDA memory reserved after moving VAE: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

            logger.info("Decoding latents...")
            if output_type != "latent":
                #latents = latents.to(vae_device) # Ensure latents are on VAE device
                scaling_factor = getattr(self.vae.config, 'scaling_factor', 1.0)
                shift_factor = getattr(self.vae.config, 'shift_factor', 0.0)
                logger.debug(f"VAE scaling_factor: {scaling_factor}, shift_factor: {shift_factor}")
                latents = (latents / scaling_factor) + shift_factor

                if latents.dtype != self.vae.dtype:
                    logger.warning(f"Casting latents from {latents.dtype} to VAE dtype {self.vae.dtype} for decoding.")
                    latents = latents.to(self.vae.dtype)

                # Add memory log before decode
                if torch.cuda.is_available():
                    logger.info(f"CUDA memory allocated before VAE decode: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
                    logger.info(f"CUDA memory reserved before VAE decode: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

                image = self.vae.decode(latents, return_dict=False)[0]

                 # Add memory log after decode
                if torch.cuda.is_available():
                    logger.info(f"CUDA memory allocated after VAE decode: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
                    logger.info(f"CUDA memory reserved after VAE decode: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")


                logger.info("Postprocessing image...")
                image = self.image_processor.postprocess(image, output_type=output_type)
            else:
                logger.info("Output type is 'latent', skipping VAE decoding.")
                image = latents # Return raw latents (potentially on GPU)

            decode_end_time = time.time()
            logger.info(f"--- Decoding Phase finished in {decode_end_time - decode_start_time:.2f} seconds ---")
        except Exception as e:
            logger.error(f"Error during decoding phase: {e}", exc_info=True)
            # Optionally re-raise or handle the exception as needed
            raise e

        finally:
            # Move transformer and VAE back to CPU
            logger.info("Moving transformer and VAE back to CPU...")
            if hasattr(self, 'transformer') and self.transformer is not None:
                logger.info("  Moving Transformer to CPU...")
                #self.transformer.to("cpu")
                logger.info("  Transformer moved to CPU.")
            if hasattr(self, 'vae') and self.vae is not None:
                logger.info("  Moving VAE to CPU...")
                #self.vae.to("cpu")
                logger.info("  VAE moved to CPU.")

            diffusion_end_time = time.time()
            logger.info(f"--- Diffusion Phase (including model offload) finished in {diffusion_end_time - diffusion_start_time:.2f} seconds ---")

            # Clear VRAM
            if torch.cuda.is_available():
                logger.info("Clearing CUDA cache after diffusion and decoding...")
                torch.cuda.empty_cache()
                logger.info(f"CUDA memory allocated after final clear: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
                logger.info(f"CUDA memory reserved after final clear: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")
        # --- End Diffusion Phase ---

        # Offload hooks might be managed by accelerate, but call just in case
        if hasattr(self, "maybe_free_model_hooks"):
             logger.info("Calling maybe_free_model_hooks...")
             self.maybe_free_model_hooks()

        if not return_dict:
            # If returning latents, ensure they are on CPU if requested implicitly
            if output_type == "latent" and isinstance(image, torch.Tensor):
                 logger.info("Moving latent output to CPU...")
                 image = image.cpu()
            logger.info(f"--- Pipeline __call__ finished in {time.time() - start_call_time:.2f} seconds (returning tuple) ---")
            return (image,)

        # If returning dict with PIL image, it's already on CPU
        # If returning dict with latents, ensure they are on CPU
        if output_type == "latent" and isinstance(image, torch.Tensor):
            logger.info("Moving latent output to CPU for dictionary return...")
            image = image.cpu()
        logger.info(f"--- Pipeline __call__ finished in {time.time() - start_call_time:.2f} seconds (returning dict) ---")
        return HiDreamImagePipelineOutput(images=image)

# --- End Modified HiDreamImagePipelineFP8 Class ---



class HiDreamImagePipelineDynamicLoad(HiDreamImagePipelineFP8):
    @contextmanager
    def dynamic_text_encoder_offload(self, device):
        """
        Context manager that moves all text encoder models to the specified device
        (typically a GPU) before use, and then offloads them back to CPU after use.
        """
        try:
            logger.info(f"Loading text encoders to device {device} for prompt encoding...")
            if self.text_encoder is not None:
                self.text_encoder.to(device)
            if self.text_encoder_2 is not None:
                self.text_encoder_2.to(device)
            if self.text_encoder_3 is not None:
                self.text_encoder_3.to(device)
            #if self.text_encoder_4 is not None:
                #self.text_encoder_4.to(device)
            yield  # run prompt encoding while the models are on GPU
        except Exception as e:
            logger.error(f"Error during dynamic text encoder offload: {e}", exc_info=True)
            raise e
        finally:
            logger.info("Offloading text encoders back to CPU to free GPU memory...")
            if self.text_encoder is not None:
                self.text_encoder.to("cpu")
            if self.text_encoder_2 is not None:
                self.text_encoder_2.to("cpu")
            if self.text_encoder_3 is not None:
                self.text_encoder_3.to("cpu")
            #if self.text_encoder_4 is not None:
                #self.text_encoder_4.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Text encoders offloaded; GPU cache cleared.")

    def encode_prompt(
        self,
        prompt,
        prompt_2,
        prompt_3,
        prompt_4,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        negative_prompt_4: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 128,
        lora_scale: Optional[float] = None,
    ):
        # Determine the target device (usually your execution device, e.g. "cuda")
        if device is None:
            device = self._execution_device

        # Wrap the base encode_prompt functionality in the dynamic text encoder offload context
        with self.dynamic_text_encoder_offload(device):
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = super().encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_3=prompt_3,
                prompt_4=prompt_4,
                device=device,
                dtype=dtype,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                negative_prompt_3=negative_prompt_3,
                negative_prompt_4=negative_prompt_4,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
        logger.info("Prompt encoding complete.")
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

# --- Model Loading Function (Updated for TorchAoConfig) ---
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
            # Choose FP8 type: 'float8wo_e4m3', 'float8dq_e4m3', etc.
            fp8_quant_type = "float8wo_e4m3" # Default to weight-only E4M3
            logger.info(f"Defining TorchAoConfig for Transformer quantization type: {fp8_quant_type}")
            try:
                quantization_config = TorchAoConfig(fp8_quant_type)
                transformer_quant_status = f"with TorchAoConfig '{fp8_quant_type}' quantization"
                logger.info("TorchAoConfig created successfully for Transformer.")
            except Exception as e:
                 logger.error(f"Failed to create TorchAoConfig: {e}. Disabling quantization for Transformer.", exc_info=True)
                 quantization_config = None # Ensure it's None if creation fails
                 global quantize_enabled # Make sure the global flag is updated
                 quantize_enabled = False # This might need adjustment if only part fails

    if not load_transformer:
        # --- Load Models to CPU (Applying Quantization via config if defined) ---
        # Load non-transformer models first
        logger.info("--- Loading VAE and Text Encoders to CPU ---")
        start_time = time.time()
        logger.info("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae", torch_dtype=dtype
        )#.to("cpu")
        logger.info(f"VAE loaded to CPU in {time.time() - start_time:.2f}s.")

        start_time = time.time()
        logger.info("Loading Text Encoder 1 (CLIP)...")
        text_encoder = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=dtype
        )#.to("cpu")
        logger.info(f"Text Encoder 1 (CLIP) loaded to CPU in {time.time() - start_time:.2f}s.")

        start_time = time.time()
        logger.info("Loading Text Encoder 2 (CLIP)...")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=dtype
        )#.to("cpu")
        logger.info(f"Text Encoder 2 (CLIP) loaded to CPU in {time.time() - start_time:.2f}s.")

        start_time = time.time()
        logger.info("Loading Text Encoder 3 (T5)...")
        text_encoder_3 = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder_3", torch_dtype=dtype
        )#.to("cpu")
        logger.info(f"Text Encoder 3 (T5) loaded to CPU in {time.time() - start_time:.2f}s.")

        # Load Llama separately (Transformers quantization is handled differently)
        # Note: Transformers quantization (like int8) happens during loading if config is passed.
        # The user script defined `llama_quantization_config` but didn't pass it.
        # We'll keep it unloaded for now as per the original logic, but log clearly.
        start_time = time.time()
        logger.info(f"Loading Text Encoder 4 (Llama) from {LLAMA_MODEL_NAME} to CPU without explicit quantization config...")
        try:
            text_encoder_4 = LlamaForCausalLM.from_pretrained(
                LLAMA_MODEL_NAME,
                output_hidden_states=True,
                output_attentions=True,
                torch_dtype=dtype,
                #low_cpu_mem_usage=True,
                load_in_8bit=True,             # This activates 8-bit quantization
                device_map="auto",             # Optionally, let the model be mapped appropriately
                #quantization_config=TorchAoConfig(fp8_quant_type)
            )#.to("cpu")
            logger.info(f"Text Encoder 4 (Llama) loaded to CPU in {time.time() - start_time:.2f}s.")
        except OSError:
            logger.error(f"Cannot find Llama model files at '{LLAMA_MODEL_NAME}'. Please check the path.")
            raise
        except Exception as e:
            logger.error(f"Error loading Llama model: {e}", exc_info=True)
            raise
    else:
        # Load Transformer last (potentially largest and subject to TorchAo quantization)
        start_time = time.time()
        logger.info(f"--- Loading Transformer to CPU {transformer_quant_status} ---")
        try:
            # Log GPU ram usage
            if torch.cuda.is_available():
                logger.info(f"CUDA memory allocated before loading transformer: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
                logger.info(f"CUDA memory reserved before loading transformer: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
                
            transformer = HiDreamImageTransformer2DModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=dtype,
                quantization_config=quantization_config # Pass the TorchAoConfig here!
            )
            logger.info(f"Transformer loaded to CPU in {time.time() - start_time:.2f}s.")
            # Log GPU ram usage
            if torch.cuda.is_available():
                logger.info(f"CUDA memory allocated after loading transformer: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
                logger.info(f"CUDA memory reserved after loading transformer: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
            
            if quantization_config:
                logger.info("TorchAo quantization should be applied to the Transformer during loading.")
                # You might need further checks here to confirm quantization actually happened,
                # e.g., by inspecting layer types or checking for specific attributes added by torchao.
            
            return transformer
          
        except Exception as e:
            logger.error(f"Error loading Transformer model: {e}", exc_info=True)
            raise

    # --- Scheduler Setup ---
    start_time = time.time()
    logger.info("Initializing Scheduler...")
    scheduler_class = config["scheduler"]
    scheduler_config_params = {}
    if issubclass(scheduler_class, FlowUniPCMultistepScheduler):
         scheduler_config_params = {"num_train_timesteps": 1000, "shift": config["shift"], "use_dynamic_shifting": False}
         logger.info(f"Configuring {scheduler_class.__name__} with: {scheduler_config_params}")
    # Add elif for other schedulers if they need specific init args

    scheduler = scheduler_class(**scheduler_config_params)
    logger.info(f"Scheduler {scheduler_class.__name__} initialized in {time.time() - start_time:.2f}s.")


    # --- Create Pipeline ---
    start_time = time.time()
    logger.info("Initializing HiDreamImagePipelineFP8 (memory-managed pipeline)...")
    try:
        # Instantiate the pipeline that handles memory management
        # All models are passed as CPU objects here.
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
        pipe.enable_model_cpu_offload()
        #pipe.transformer = transformer
        

        # The pipeline's internal device is set, but models remain on CPU
        #target_device = "cuda" if torch.cuda.is_available() else "cpu"
        #pipe.to(target_device) # This sets pipe._execution_device, doesn't move models yet
        #logger.info(f"Pipeline initialized. Target execution device set to '{target_device}'. Models remain on CPU until use.")
        logger.info(f"Pipeline initialized in {time.time() - start_time:.2f}s.")
    except Exception as e:
        logger.error(f"Error initializing pipeline: {e}", exc_info=True)
        raise

    logger.info(f"--- Model Loading finished in {time.time() - load_start_time:.2f} seconds ---")
    return pipe, config
# --- End Model Loading Function ---


# --- Main Execution ---
if __name__ == "__main__":

    # Log initial setup
    logger.info(f"--- Starting HiDream Generation Script ---")
    script_start_time = time.time()
    logger.info(f"Selected model type: {args.model_type}")
    # Update quantize_enabled based on availability check *after* imports
    quantize_enabled = True
    logger.info(f"FP8 Quantization (Transformer): {'Requested and Available (using TorchAoConfig)' if quantize_enabled else ('Requested but Unavailable' if args.quantize else 'Disabled')}")
    logger.info(f"Using Llama from: {args.llama_path}")
    logger.info(f"Output file: {args.output_file}")

    # Check CUDA and FP8 capability
    if torch.cuda.is_available():
        capability_tuple = torch.cuda.get_device_capability()
        capability = capability_tuple[0] + capability_tuple[1] / 10.0
        logger.info(f"CUDA available: Device: {torch.cuda.get_device_name(0)}, Capability: {capability_tuple}")
        if quantize_enabled and capability < 8.9: # FP8 needs Hopper (SM 8.9+) ideally >= 9.0
             logger.warning(f"Current GPU compute capability {capability_tuple} may have limited or no native FP8 support. Requires >= (8, 9). Quantization performance/support may vary.")
        # Log initial memory
        logger.info(f"Initial CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        logger.info(f"Initial CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        logger.warning("CUDA is not available. Running on CPU will be extremely slow and is not recommended.")
        if quantize_enabled:
            logger.warning("Quantization requested but CUDA not found. Disabling quantization as it typically requires GPU acceleration.")
            quantize_enabled = False # Force disable if no CUDA

    # Load models
    try:
        pipe, model_specific_config = load_models(args.model_type, load_transformer=False, quantize=quantize_enabled)
    except Exception as e:
        logger.critical(f"Failed to load models: {e}", exc_info=True)
        sys.exit(1) # Exit if models can't load

    # Generation parameters from args and config
    prompt = args.prompt
    resolution_str = args.resolution
    seed = args.seed
    guidance_scale = model_specific_config["guidance_scale"]
    num_inference_steps = model_specific_config["num_inference_steps"]

    logger.info(f"\n--- Generation Settings ---")
    logger.info(f"  Prompt: \"{prompt}\"")
    logger.info(f"  Model: {args.model_type}")
    logger.info(f"  Resolution String: {resolution_str}")
    logger.info(f"  Steps: {num_inference_steps}")
    logger.info(f"  CFG Scale: {guidance_scale}")
    logger.info(f"  Seed: {seed if seed != -1 else 'Random'}")

    # Parse resolution
    height, width = parse_resolution(resolution_str)
    logger.info(f"  Parsed Resolution (H, W): ({height}, {width})")

    # Handle seed
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item() # Use a larger range for seed
        logger.info(f"Generated random seed: {seed}")

    # Setup generator (use CUDA if available, otherwise CPU)
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(generator_device).manual_seed(seed)
    logger.info(f"Using generator on device '{generator_device}' with seed {seed}")

    # Generate image
    try:
        logger.info("--- Starting Image Generation Pipeline Execution ---")
        gen_start_time = time.time()
        # Use CUDA events for more accurate GPU timing if available
        start_event, end_event = None, None
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

        image_output = pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=1,
            generator=generator,
            output_type="pil", # Get PIL image
            return_dict=True
        )

        elapsed_time_sec = -1.0
        if torch.cuda.is_available() and start_event and end_event:
            end_event.record()
            torch.cuda.synchronize() # Wait for GPU operations to complete
            elapsed_time_ms = start_event.elapsed_time(end_event)
            elapsed_time_sec = elapsed_time_ms / 1000.0
            logger.info(f"Image generation finished (GPU time): {elapsed_time_sec:.2f} seconds.")
        else:
            gen_end_time = time.time()
            elapsed_time_sec = gen_end_time - gen_start_time
            logger.info(f"Image generation finished (CPU time): {elapsed_time_sec:.2f} seconds.")


        image = image_output.images[0]

        # Save the image
        start_save_time = time.time()
        image.save(args.output_file)
        logger.info(f"Image saved successfully to {args.output_file} in {time.time() - start_save_time:.2f}s")

    except Exception as e:
        logger.critical(f"Error during image generation pipeline execution: {e}", exc_info=True)
        logger.critical("Check VRAM usage (nvidia-smi), model paths, prompt/resolution, and full traceback for details.")
        if torch.cuda.is_available():
             logger.info(f"CUDA memory allocated at error: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
             logger.info(f"CUDA memory reserved at error: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        sys.exit(1) # Exit on generation error

    logger.info(f"--- Script finished successfully in {time.time() - script_start_time:.2f} seconds ---")
# --- End Main Execution ---