import cv2
import numpy as np
import imageio
from PIL import Image
from pathlib import Path
from typing import List, Union
import torch

class VideoUtils:
    @staticmethod
    def frames_to_video(frames: List[Union[np.ndarray, Image.Image]], 
                       output_path: str, 
                       fps: int = 7) -> str:
        """Convert frames to video file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert PIL images to numpy arrays if needed
        processed_frames = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            processed_frames.append(frame)
        
        # Write video using imageio
        with imageio.get_writer(output_path, fps=fps, codec='libx264') as writer:
            for frame in processed_frames:
                writer.append_data(frame)
        
        return str(output_path)
    
    @staticmethod
    def video_to_frames(video_path: str) -> List[np.ndarray]:
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    @staticmethod
    def resize_frames(frames: List[np.ndarray], 
                     width: int, height: int) -> List[np.ndarray]:
        """Resize frames to target dimensions"""
        resized_frames = []
        for frame in frames:
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
            resized_frames.append(resized)
        return resized_frames
