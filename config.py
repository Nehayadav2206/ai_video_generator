import os
from pathlib import Path

class Config:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.absolute()
    OUTPUT_DIR = PROJECT_ROOT / "output" / "generated_videos"
    TEMP_DIR = PROJECT_ROOT / "temp"
    
    # Model settings
    MODEL_CACHE_DIR = PROJECT_ROOT / "model_cache"
    VIDEO_MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
    TEXT_ENCODER_MODEL_ID = "openai/clip-vit-large-patch14"
    
    # Video generation settings
    DEFAULT_NUM_FRAMES = 25
    DEFAULT_FPS = 7
    DEFAULT_VIDEO_LENGTH = 3  # seconds
    MAX_VIDEO_LENGTH = 10
    VIDEO_WIDTH = 1024
    VIDEO_HEIGHT = 576
    
    # Processing settings
    BATCH_SIZE = 1
    MAX_SEQUENCE_LENGTH = 77
    GUIDANCE_SCALE = 7.5
    NUM_INFERENCE_STEPS = 25
    
    # Hardware settings
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    MIXED_PRECISION = True
    ENABLE_MEMORY_EFFICIENT_ATTENTION = True
    
    # UI settings
    GRADIO_SHARE = False
    GRADIO_PORT = 7860
    STREAMLIT_PORT = 8501
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)