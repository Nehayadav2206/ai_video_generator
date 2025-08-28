import streamlit as st
import torch
from PIL import Image
import datetime
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing your modules
try:
    from models.model_manager import ModelManager
    from utils.video_utils import VideoUtils
    from utils.file_utils import FileUtils
    from config import Config
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    import_error = str(e)

@st.cache_resource
def load_models_static():
    #def load_models():
    """Load AI models with caching"""
    if not DEPENDENCIES_AVAILABLE:
       raise ImportError("Dependencies not available")
    Config.create_directories()
    return ModelManager()

class StreamlitApp:
    def __init__(self):
        self.init_session_state()

    def init_session_state(self):
        if 'model_manager' not in st.session_state:
            st.session_state.model_manager = None
        if 'generated_video' not in st.session_state:
            st.session_state.generated_video = None
        if 'generation_status' not in st.session_state:
            st.session_state.generation_status = ""




    def main(self):
        st.set_page_config(page_title="AI Video Generator", layout="wide")
        st.title("🎬 AI Video Generator")

        if not DEPENDENCIES_AVAILABLE:
            st.error(f"Missing dependencies: {import_error}")
            return

        # Sidebar
        with st.sidebar:
            st.header("⚙️ Settings")
            if st.session_state.model_manager is None:
                if st.button("🚀 Load Models"):
                    try:
                        #st.session_state.model_manager = StreamlitApp.load_models(self)
                        #st.session_state.model_manager = self.load_models()
                        st.session_state.model_manager = load_models_static()
                        st.success("Models loaded successfully!")
                    except Exception as e:
                        st.error(f"Failed to load models: {e}")
                        return
                else:
                    st.warning("Click 'Load Models' to start")
                    return
            else:
                st.success("✅ Models Ready")

            st.subheader("📹 Video Settings")
            num_frames = st.slider("Number of Frames", 10, 50, 25)
            fps = st.slider("FPS", 5, 30, 7)

            st.subheader("🎨 Style Settings")
            motion_intensity = st.slider("Motion Intensity", 1, 255, 127)
            noise_strength = st.slider("Noise Strength", 0.0, 1.0, 0.1)

            st.subheader("🔧 Technical Settings")
            inference_steps = st.slider("Inference Steps", 10, 50, 25)
            seed = st.number_input("Seed (-1 for random)", value=-1, step=1)

            device = "GPU" if torch.cuda.is_available() else "CPU"
            st.info(f"Device: {device}")
            if torch.cuda.is_available():
                st.info(f"GPU: {torch.cuda.get_device_name()}")

        # Main content
        col1, col2 = st.columns(2)
        with col1:
            st.header("📥 Input")
            uploaded_image = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Input Image")

            text_prompt = st.text_area("Text Prompt (Optional)")
            negative_prompt = st.text_area("Negative Prompt (Optional)")

            if st.button("🎬 Generate Video"):
                if uploaded_image is None:
                    st.error("Please upload an image first!")
                else:
                    self.generate_video(
                        image, text_prompt, negative_prompt,
                        num_frames, fps, motion_intensity,
                        noise_strength, inference_steps, seed
                    )

        with col2:
            st.header("📤 Output")
            if st.session_state.generation_status:
                st.info(st.session_state.generation_status)
            if st.session_state.generated_video:
                st.video(st.session_state.generated_video)
                with open(st.session_state.generated_video, 'rb') as file:
                    st.download_button("📥 Download Video", data=file.read(),
                                       file_name=Path(st.session_state.generated_video).name,
                                       mime="video/mp4")

    def generate_video(self, image, text_prompt, negative_prompt,
                      num_frames, fps, motion_intensity,
                      noise_strength, inference_steps, seed):
        """Generate video (simplified version)"""
        st.session_state.generation_status = "🚀 Video generation started..."
        st.progress(10)

        try:
            video_frames = st.session_state.model_manager.generate_video_pipeline(
                image_input=image,
                text_prompt=text_prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_frames=num_frames,
                fps=fps,
                motion_bucket_id=motion_intensity,
                noise_aug_strength=noise_strength,
                num_inference_steps=inference_steps,
                seed=seed if seed != -1 else None
            )

            video_utils = VideoUtils()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            clean_prompt = FileUtils.clean_filename(text_prompt[:30] if text_prompt else "generated")
            filename = f"{clean_prompt}_{timestamp}.mp4"
            output_path = Config.OUTPUT_DIR / filename

            video_path = video_utils.frames_to_video(video_frames, str(output_path), fps=fps)
            st.session_state.generated_video = video_path
            st.session_state.generation_status = f"✅ Video generated: {filename}"
            st.progress(100)
        except Exception as e:
            st.session_state.generation_status = f"❌ Error: {e}"

# Run app
#StreamlitApp().main()
# Run app
if __name__ == "__main__":
    StreamlitApp().main()

