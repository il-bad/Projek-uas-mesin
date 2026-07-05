import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# 1. Mengatur Tampilan Halaman Web
st.set_page_config(page_title="Deteksi APD Konstruksi", page_icon="👷‍♂️", layout="wide")
st.title("Aplikasi Deteksi APD Pekerja Konstruksi 👷‍♂️🚧")

# 2. Memuat Model YOLO
@st.cache_resource
def load_model():
    return YOLO("model_apd_terbaik.pt")

try:
    model = load_model()
    st.sidebar.success("Model YOLO berhasil dimuat! ✅")
except Exception as e:
    st.sidebar.error("Model 'model_apd_terbaik.pt' tidak ditemukan di folder proyek. ❌")

# 3. Membuat Menu Navigasi Tab
tab1, tab2, tab3 = st.tabs(["📸 Deteksi Foto", "🎥 Deteksi Video", "📹 Kamera Live WebRTC"])

# ================= TAB 1: DETEKSI FOTO =================
with tab1:
    st.header("Upload Foto Pekerja")
    uploaded_image = st.file_uploader("Pilih foto...", type=["jpg", "jpeg", "png"], key="image_uploader")
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Foto Asli")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Hasil Deteksi")
            with st.spinner("Memproses foto..."):
                results = model(image)
                res_plotted = results[0].plot()
                st.image(res_plotted, channels="BGR", use_container_width=True)
                st.success("Foto berhasil dideteksi!")

# ================= TAB 2: DETEKSI VIDEO =================
with tab2:
    st.header("Upload Video Situasi Proyek")
    uploaded_video = st.file_uploader("Pilih video...", type=["mp4", "avi", "mov"], key="video_uploader")
    
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_video.read())
            video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        st.subheader("Proses Deteksi Video (Real-time)")
        frame_placeholder = st.empty()
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            start_detection = st.button("Mulai Deteksi Video", type="primary")
        with col_btn2:
            stop_detection = st.button("🔴 Hentikan Video")
        
        if start_detection:
            st.session_state.is_running = True
            while cap.isOpened():
                if stop_detection:
                    st.session_state.is_running = False
                    st.warning("Deteksi video dihentikan oleh pengguna! ⚠️")
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = model(frame, stream=True)
                frame_output = frame
                for r in results:
                    frame_output = r.plot()
                
                frame_rgb = cv2.cvtColor(frame_output, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            cap.release()
            if getattr(st.session_state, 'is_running', True):
                st.success("Video selesai diproses sampai akhir! 🎉")
            
        try:
            cap.release()
            if os.path.exists(video_path):
                os.unlink(video_path)
        except Exception:
            pass

# ================= TAB 3: KAMERA LIVE =================
with tab3:
    st.header("Deteksi Real-time Lewat Kamera Browser")
    st.write("Klik START untuk mengaktifkan kamera. Hasil deteksi langsung muncul di dalam video.")

    class YOLOVideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            img_output = results[0].plot()
            return img_output

    webrtc_streamer(key="yolo-webrtc", video_transformer_factory=YOLOVideoTransformer)