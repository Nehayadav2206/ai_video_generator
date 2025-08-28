import gradio as gr
import torch
from PIL import Image
from pathlib import Path
import datetime
from models.model_manager import ModelManager
from utils.video_utils import VideoUtils
from utils.file_utils import FileUtils
from config import Config
#from ..models.model_manager import ModelManager
#from ..utils.video_utils import VideoUtils
#from ..utils.file_utils import FileUtils
#from ..config import Config

class GradioApp:
    def __init__(self):
        self.model_manager = ModelManager()
        self.video_utils = VideoUtils()
        Config.create_directories()
    
    def generate_video_interface(self, 
                               image_input,
                               text_prompt,
                               negative_prompt,
                               num_frames,
                               fps,
                               motion_intensity,
                               noise_strength,
                               inference_steps,
                               seed,
                               progress=gr.Progress()):
        """Main video generation interface"""
        
        if image_input is None:
            return None, "Please upload an image first."
        
        try:
            progress(0.1, desc="Initializing...")
            
            # Generate video frames
            progress(0.2, desc="Generating video frames...")
            video_frames = self.model_manager.generate_video_pipeline(
                image_input=image_input,
                text_prompt=text_prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_frames=int(num_frames),
                fps=int(fps),
                motion_bucket_id=int(motion_intensity),
                noise_aug_strength=noise_strength,
                num_inference_steps=int(inference_steps),
                seed=int(seed) if seed != -1 else None
            )
            
            progress(0.8, desc="Saving video...")
            
            # Save video
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_prompt = FileUtils.clean_filename(text_prompt[:30] if text_prompt else "generated")
            filename = f"{clean_prompt}_{timestamp}.mp4"
            output_path = Config.OUTPUT_DIR / filename
            
            video_path = self.video_utils.frames_to_video(
                video_frames, 
                str(output_path), 
                fps=int(fps)
            )
            
            progress(1.0, desc="Complete!")
            
            return video_path, f"Video generated successfully! Saved as: {filename}"
            
        except Exception as e:
            error_msg = f"Error generating video: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def create_interface(self):
        """Create Gradio interface"""
        
        with gr.Blocks(
            title="AI Video Generator", 
            theme=gr.themes.Soft(),
            css=".gradio-container {max-width: 1200px; margin: auto;}"
        ) as interface:
            
            gr.Markdown(
                """
                # 🎬 AI Video Generator
                
                Generate amazing videos from images using AI! Upload an image and optionally provide a text prompt 
                to guide the video generation process.
                
                **Tips:**
                - Use high-quality images for best results
                - Text prompts can help guide the motion and style
                - Experiment with different motion intensity settings
                - Higher frame counts = longer videos but more processing time
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Input controls
                    gr.Markdown("### 📥 Input")
                    
                    image_input = gr.Image(
                        label="Upload Image",
                        type="pil",
                        format="RGB"
                    )
                    
                    text_prompt = gr.Textbox(
                        label="Text Prompt (Optional)",
                        placeholder="Describe the motion or style you want...",
                        lines=3
                    )
                    
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt (Optional)",
                        placeholder="What you don't want in the video...",
                        lines=2
                    )
                    
                    # Advanced settings
                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        with gr.Row():
                            num_frames = gr.Slider(
                                minimum=10,
                                maximum=50,
                                value=25,
                                step=1,
                                label="Number of Frames"
                            )
                            
                            fps = gr.Slider(
                                minimum=5,
                                maximum=30,
                                value=7,
                                step=1,
                                label="FPS"
                            )
                        
                        with gr.Row():
                            motion_intensity = gr.Slider(
                                minimum=1,
                                maximum=255,
                                value=127,
                                step=1,
                                label="Motion Intensity"
                            )
                            
                            noise_strength = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.1,
                                step=0.01,
                                label="Noise Strength"
                            )
                        
                        with gr.Row():
                            inference_steps = gr.Slider(
                                minimum=10,
                                maximum=50,
                                value=25,
                                step=1,
                                label="Inference Steps"
                            )
                            
                            seed = gr.Number(
                                label="Seed (-1 for random)",
                                value=-1,
                                precision=0
                            )
                    
                    generate_btn = gr.Button(
                        "🎬 Generate Video", 
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    # Output
                    gr.Markdown("### 📤 Output")
                    
                    output_video = gr.Video(
                        label="Generated Video",
                        format="mp4"
                    )
                    
                    output_message = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
                    
                    # Example gallery
                    gr.Markdown("### 🎨 Examples")
                    gr.Examples(
                        examples=[
                            [
                                "examples/landscape.jpg",
                                "A serene landscape with gentle camera movement",
                                "",
                                25,
                                7,
                                100,
                                0.1,
                                25,
                                42
                            ],
                        ],
                        inputs=[
                            image_input, text_prompt, negative_prompt,
                            num_frames, fps, motion_intensity, 
                            noise_strength, inference_steps, seed
                        ]
                    )
            
            # Event handlers
            generate_btn.click(
                fn=self.generate_video_interface,
                inputs=[
                    image_input, text_prompt, negative_prompt,
                    num_frames, fps, motion_intensity,
                    noise_strength, inference_steps, seed
                ],
                outputs=[output_video, output_message],
                show_progress=True
            )
            
            # Footer
            gr.Markdown(
                """
                ---
                
                **Note:** Video generation can take several minutes depending on your hardware. 
                GPU acceleration is recommended for faster processing.
                
                Generated videos are saved in the `output/generated_videos/` directory.
                """
            )
        
        return interface
    
    def launch(self, share=False, debug=False):
        """Launch the Gradio interface"""
        interface = self.create_interface()
        interface.launch(
            share=share,
            debug=debug,
            server_port=Config.GRADIO_PORT,
            server_name="0.0.0.0" if share else "127.0.0.1"
        )

        #RUN the APP

        if __name__ == "__main__":
            app = GradioApp()
            app.launch(share=False, debug=True)

        #RUN the APP

    
    

