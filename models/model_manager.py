#import torch
#from typing import Dict, Any, Optional
#from ..utils.logger import Logger
#from .image_processor import ImageProcessor
#from .text_encoder import TextEncoder
#from .video_generator import VideoGenerator

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import Logger
from models.image_processor import ImageProcessor
from models.text_encoder import TextEncoder
from models.video_generator import VideoGenerator


class ModelManager:
    """Centralized model management"""
    
    def __init__(self):
        self.logger = Logger("ModelManager")
        self.models: Dict[str, Any] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models"""
        try:
            self.logger.info("Initializing models...")
            
            # Image processor
            self.models['image_processor'] = ImageProcessor()
            
            # Text encoder
            self.models['text_encoder'] = TextEncoder()
            
            # Video generator
            self.models['video_generator'] = VideoGenerator()
            
            self.logger.info("All models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise
    
    def get_model(self, model_name: str):
        """Get specific model"""
        return self.models.get(model_name)
    
    def generate_video_pipeline(self, 
                              image_input,
                              text_prompt: str,
                              **generation_kwargs) -> list:
        """Complete video generation pipeline"""
        self.logger.info("Starting complete video generation pipeline...")
        
        try:
            # Process image
            image_processor = self.get_model('image_processor')
            processed_image = image_processor.preprocess_image(image_input)
            
            # Generate video
            video_generator = self.get_model('video_generator')
            video_frames = video_generator.generate_video(
                image=image_input,
                prompt=text_prompt,
                **generation_kwargs
            )
            
            self.logger.info("Pipeline completed successfully")
            return video_frames
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            raise
    
    def cleanup_all(self):
        """Cleanup all models"""
        for model_name, model in self.models.items():
            if hasattr(model, 'cleanup_memory'):
                model.cleanup_memory()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()