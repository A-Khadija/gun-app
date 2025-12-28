import streamlit as st
import cv2
import requests
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# --- Configuration ---
API_URL = "https://predict.ultralytics.com"
MODEL_URL = "https://hub.ultralytics.com/models/ae7Z7jXfrZNvReWvofz0"

st.set_page_config(page_title="YOLO API Web App", layout="centered")

st.title("📷 Object Detection (Web Version)")
st.markdown("This works on the cloud! It sends frames from your browser to the server.")

# --- Sidebar ---
st.sidebar.header("Settings")
# Securely get API Key
if "ultralytics_api_key" in st.secrets:
    api_key = st.secrets["ultralytics_api_key"]
else:
    api_key = st.sidebar.text_input("Enter your Ultralytics API Key", type="password")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

# --- Processing Class ---
# --- Processing Class ---
class YoloProcessor(VideoProcessorBase):
    def __init__(self):
        self.api_key = None
        self.conf = 0.25
        self.iou = 0.45
        # Add variables to track frames
        self.frame_count = 0
        self.last_results = None  # Store the last known boxes

    def recv(self, frame):
        # 1. Convert WebRTC frame to OpenCV format
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Skip inference if no API key is provided
        if not self.api_key:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # 2. OPTIMIZATION: Only call API once every 30 frames (approx every 1 second)
        # We assume 30 FPS. This prevents the "Connection Timeout" error.
        if self.frame_count % 30 == 0:
            try:
                # Prepare image
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                _, img_encoded = cv2.imencode('.jpg', img)
                image_bytes = img_encoded.tobytes()

                headers = {"x-api-key": self.api_key}
                data = {
                    "model": MODEL_URL,
                    "imgsz": 640,
                    "conf": self.conf,
                    "iou": self.iou
                }
                files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
                
                # Send to API
                response = requests.post(API_URL, headers=headers, data=data, files=files)
                if response.status_code == 200:
                    self.last_results = response.json()
            except Exception as e:
                print(f"API Error: {e}")

        # 3. Always draw the *last known* results on the current frame
        # This keeps the video smooth even while waiting for the next API update
        if self.last_results:
            img = self.draw_bbox(img, self.last_results)

        # 4. Return the processed frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def draw_bbox(self, frame, results):
        """Helper to draw boxes on the frame"""
        try:
            data = results[0] if isinstance(results, list) else results
            detections = []
            if 'boxes' in data:
                raw_boxes = data['boxes']
                if isinstance(raw_boxes, dict) and 'data' in raw_boxes:
                    detections = raw_boxes['data']
                elif isinstance(raw_boxes, list):
                    detections = raw_boxes
            
            for det in detections:
                if 'box' in det:
                    x1, y1 = int(det['box']['x1']), int(det['box']['y1'])
                    x2, y2 = int(det['box']['x2']), int(det['box']['y2'])
                    label = det.get('name', str(det.get('cls', 'Object')))
                    conf = det.get('conf', 0)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{label} {conf:.2f}"
                    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception:
            pass
        return frame
# --- Main Layout ---

if not api_key:
    st.warning("Please enter your API Key in the sidebar to start.")
else:
   # We create the streamer with better network settings
    ctx = webrtc_streamer(
        key="yolo-stream",
        video_processor_factory=YoloProcessor,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:global.stun.twilio.com:3478"]},
                {"urls": ["stun:stun.framasoft.org:3478"]},
            ]
        }
    )

    # Pass the sidebar settings into the processor
    if ctx.video_processor:
        ctx.video_processor.api_key = api_key
        ctx.video_processor.conf = conf_threshold
        ctx.video_processor.iou = iou_threshold