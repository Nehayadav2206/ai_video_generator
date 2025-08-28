import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Union, List

class ImageProcessor:
    def __init__(self, target_size=(1024, 576)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
        
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess single image for model input"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("Unsupported image type")
        
        # Apply transforms
        tensor = self.transform(image)
        return tensor
    
    def preprocess_batch(self, images: List[Union[str, Image.Image, np.ndarray]]) -> torch.Tensor:
        """Preprocess batch of images"""
        processed_images = []
        for img in images:
            processed_img = self.preprocess_image(img)
            processed_images.append(processed_img)
        
        return torch.stack(processed_images)
    
    def postprocess_frames(self, frames: torch.Tensor) -> List[Image.Image]:
        """Convert tensor frames back to PIL Images"""
        # Ensure frames are in [0, 1] range
        frames = torch.clamp(frames, 0, 1)
        
        pil_frames = []
        for frame in frames:
            # Convert from tensor to PIL
            frame_np = (frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_frame = Image.fromarray(frame_np)
            pil_frames.append(pil_frame)
        
        return pil_frames
    
    def create_image_grid(self, images: List[Image.Image], grid_size=(2, 2)) -> Image.Image:
        """Create a grid of images"""
        rows, cols = grid_size
        w, h = images[0].size
        
        grid_img = Image.new('RGB', (cols * w, rows * h))
        
        for i, img in enumerate(images[:rows * cols]):
            row = i // cols
            col = i % cols
            grid_img.paste(img, (col * w, row * h))
        
        return grid_img
