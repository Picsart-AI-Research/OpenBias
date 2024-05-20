from torch import autocast
import sys
import torch
from diffusers import DiffusionPipeline
from utils.config import GEN_SETTING
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import numpy as np

# stylegan3 imports
import utils.stylegan3.dnnlib as dnnlib_sg3
from utils.stylegan3 import legacy

def dummy_checker(images, **kwargs):
    return images, [False]*len(images)

# stylegan3 model class
class StyleGAN3():
    def __init__(
        self,
        **kwargs
    ):
        self.device = kwargs['device']
        with dnnlib_sg3.util.open_url(kwargs['gen_info']['checkpoint_path']) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)
        self.rotate = kwargs['gen_info']['rotate']
        self.translate = legacy.parse_vec2(kwargs['gen_info']['translate'])
        self.noise_mode = kwargs['gen_info']['noise_mode']
        self.truncation_psi = kwargs['gen_info']['truncation_psi']
    
    @torch.no_grad()
    def generate_images(self, seed):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(self.device)
        label = torch.zeros([1, self.G.c_dim], device=self.device)

        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(self.G.synthesis, 'input'):
            m = legacy.make_transform(self.translate, self.rotate)
            m = np.linalg.inv(m)
            self.G.synthesis.input.transform.copy_(torch.from_numpy(m))

        img = self.G(z, label, truncation_psi=self.truncation_psi, noise_mode=self.noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img[0].cpu()

# diffusion xl model class
class Stable_Diffusion_XL():
    def __init__(
        self,
        gen_info,
        device,
        n_images,
        safe_checker = False
    ):
        # base diffusion xl model
        self.base = DiffusionPipeline.from_pretrained(
            gen_info['version'], 
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(device)
        # self.base.enable_model_cpu_offload()
        self.base.set_progress_bar_config(disable=True)
        if not safe_checker:
            self.base.safety_checker = dummy_checker
        
        # refiner
        self.refiner = DiffusionPipeline.from_pretrained(
            gen_info['refiner'],
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(device)
        # self.refiner.enable_model_cpu_offload()
        self.refiner.set_progress_bar_config(disable=True)
        if not safe_checker:
            self.refiner.safety_checker = dummy_checker
        
        self.high_noise_frac = 0.8
        self.inference_steps = GEN_SETTING['inference_steps']
        self.n_images = n_images
        self.neg_prompt = [GEN_SETTING['neg_prompt']]*GEN_SETTING['batch_size']
            
    @torch.no_grad()
    def generate_images(self, prompt):
        torch.cuda.empty_cache()
        images = self.base(
            prompt, 
            negative_prompt=self.neg_prompt,
            num_inference_steps = self.inference_steps,
            num_images_per_prompt=self.n_images,
            denoising_end=self.high_noise_frac,
            output_type="latent",
        ).images
        images = self.refiner(
            prompt=prompt,
            negative_prompt=self.neg_prompt,
            num_inference_steps=self.inference_steps,
            num_images_per_prompt=self.n_images,
            denoising_start=self.high_noise_frac,
            image=images, 
        ).images
        return images

# diffusion model class (for 1.5 and 2)
class Stable_Diffusion():
    def __init__(
        self,
        gen_info,
        device,
        n_images,
        safe_checker = False
    ):
        if gen_info['version'] == 'stabilityai/stable-diffusion-2':
            # Use the Euler scheduler here instead
            scheduler = EulerDiscreteScheduler.from_pretrained(gen_info['version'], subfolder="scheduler")
            # diffusion model
            self.dm = StableDiffusionPipeline.from_pretrained(
                gen_info['version'], 
                scheduler=scheduler,
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to(device)
        else:
            # diffusion model
            self.dm = StableDiffusionPipeline.from_pretrained(
                gen_info['version'], 
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to(device)
        # self.dm.enable_model_cpu_offload()
        self.dm.set_progress_bar_config(disable=True)
        if not safe_checker:
            self.dm.safety_checker = dummy_checker
        
        self.inference_steps = GEN_SETTING['inference_steps']
        self.n_images = n_images
        self.neg_prompt = [GEN_SETTING['neg_prompt']]*GEN_SETTING['batch_size']
            
    @torch.no_grad()
    def generate_images(self, prompt):
        torch.cuda.empty_cache()
        images = self.dm(
            prompt, 
            negative_prompt=self.neg_prompt,
            num_inference_steps = self.inference_steps,
            num_images_per_prompt=self.n_images,
        ).images
        return images


