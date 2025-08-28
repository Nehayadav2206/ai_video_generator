import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image
import numpy as np
from typing import List, Optional, Union
#from ..utils.logger import Logger
from utils.logger import Logger

class VideoGenerator:
    def __init__(self, model_id="stabilityai/stable-video-diffusion-img2vid-xt"):
        self.logger = Logger("VideoGenerator")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Load pipeline
        self.logger.info("Loading Stable Video Diffusion pipeline...")
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            variant="fp16" if self.device.type == "cuda" else None
        )
        
        if self.device.type == "cuda":
            self.pipeline = self.pipeline.to(self.device)
            self.pipeline.enable_model_cpu_offload()  # Save GPU memory
        
        self.logger.info("Pipeline loaded successfully")
    
    def generate_video(self, 
                      image: Union[str, Image.Image, np.ndarray],
                      prompt: Optional[str] = None,
                      negative_prompt: Optional[str] = None,
                      num_frames: int = 25,
                      fps: int = 7,
                      motion_bucket_id: int = 127,
                      noise_aug_strength: float = 0.1,
                      decode_chunk_size: int = 8,
                      num_inference_steps: int = 25,
                      guidance_scale: float = 7.5,
                      seed: Optional[int] = None) -> List[Image.Image]:
        """
        Generate video from image and optional text prompt
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            prompt: Text description (optional for SVD)
            negative_prompt: Negative text description
            num_frames: Number of frames to generate
            fps: Frames per second
            motion_bucket_id: Motion intensity (higher = more motion)
            noise_aug_strength: Noise augmentation strength
            decode_chunk_size: Batch size for decoding (lower = less GPU memory)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
        
        Returns:
            List of PIL Images representing video frames
        """
        self.logger.info("Starting video generation...")
        
        # Preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Ensure image dimensions are divisible by 8
        width, height = image.size
        width = (width // 8) * 8
        height = (height // 8) * 8
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Generate video
        try:
            with torch.inference_mode():
                video_frames = self.pipeline(
                    image=image,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    decode_chunk_size=decode_chunk_size,
                    motion_bucket_id=motion_bucket_id,
                    fps=fps,
                    noise_aug_strength=noise_aug_strength,
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator().manual_seed(seed) if seed else None,
                ).frames[0]
            
            self.logger.info(f"Successfully generated {len(video_frames)} frames")
            return video_frames
            
        except Exception as e:
            self.logger.error(f"Error during video generation: {e}")
            raise
    
    def batch_generate(self, 
                      images: List[Union[str, Image.Image]], 
                      prompts: Optional[List[str]] = None,
                      **kwargs) -> List[List[Image.Image]]:
        """Generate videos for multiple images"""
        results = []
        
        for i, image in enumerate(images):
            prompt = prompts[i] if prompts and i < len(prompts) else None
            self.logger.info(f"Processing image {i+1}/{len(images)}")
            
            video_frames = self.generate_video(image=image, prompt=prompt, **kwargs)
            results.append(video_frames)
        
        return results
    
    def interpolate_images(self, 
                          image1: Union[str, Image.Image], 
                          image2: Union[str, Image.Image],
                          num_frames: int = 25,
                          **kwargs) -> List[Image.Image]:
        """Generate interpolation video between two images"""
        # This is a simplified approach - for better results, you'd need 
        # specialized interpolation models
        self.logger.info("Generating interpolation video...")
        
        # Use first image as base and generate towards second
        video_frames = self.generate_video(image=image1, num_frames=num_frames, **kwargs)
        
        # Simple blending towards second image (basic implementation)
        if isinstance(image2, str):
            image2 = Image.open(image2).convert('RGB')
        
        # Blend last few frames towards target image
        blend_frames = min(5, len(video_frames))
        for i in range(blend_frames):
            blend_ratio = (i + 1) / blend_frames
            frame = video_frames[-(blend_frames-i)]
            frame_array = np.array(frame)
            image2_array = np.array(image2.resize(frame.size))
            
            blended = (1 - blend_ratio) * frame_array + blend_ratio * image2_array
            video_frames[-(blend_frames-i)] = Image.fromarray(blended.astype(np.uint8))
        
        return video_frames
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()