import streamlit as st
import cv2
import requests
import numpy as np
import json
import time

# --- Configuration ---
# API Configuration
API_URL = "https://predict.ultralytics.com"
MODEL_URL = "https://hub.ultralytics.com/models/ae7Z7jXfrZNvReWvofz0"

st.set_page_config(page_title="YOLO API Real-Time", layout="centered")

st.title("📷 Real-Time Object Detection (API-Based)")
st.markdown("This app sends webcam frames to the Ultralytics API for inference.")

# --- Sidebar ---
st.sidebar.header("Settings")

# Check if the key is in secrets (local file), otherwise ask in sidebar
if "ultralytics_api_key" in st.secrets:
    api_key = st.secrets["ultralytics_api_key"]
else:
    api_key = st.sidebar.text_input("Enter your Ultralytics API Key", type="password")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

# --- Functions ---

def run_inference(image_bytes, api_key, conf, iou):
    """Sends the image to the API and returns the JSON response."""
    headers = {"x-api-key": api_key}
    data = {
        "model": MODEL_URL,
        "imgsz": 640,
        "conf": conf,
        "iou": iou
    }
    
    # We send the bytes directly as a file named 'frame.jpg'
    files = {"file": ("frame.jpg", image_bytes, "image/jpeg")}
    
    try:
        response = requests.post(API_URL, headers=headers, data=data, files=files)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        # Only show error if it's not a common "stop" signal
        st.error(f"API Error: {e}")
        return None

def draw_bbox(frame, results):
    """Parses JSON results and draws bounding boxes on the frame."""
    if not results:
        return frame
        
    try:
        # The Ultralytics API usually returns a list of results.
        data = results[0] if isinstance(results, list) else results
        
        detections = []
        
        # Parse 'boxes' structure
        if 'boxes' in data:
            raw_boxes = data['boxes']
            if isinstance(raw_boxes, dict) and 'data' in raw_boxes:
                detections = raw_boxes['data']
            elif isinstance(raw_boxes, list):
                detections = raw_boxes
        
        for det in detections:
            # Extract coordinates
            if 'box' in det:
                x1 = int(det['box']['x1'])
                y1 = int(det['box']['y1'])
                x2 = int(det['box']['x2'])
                y2 = int(det['box']['y2'])
                label = det.get('name', str(det.get('cls', 'Object')))
                conf = det.get('conf', 0)
            else:
                continue

            # Draw Rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw Label
            text = f"{label} {conf:.2f}"
            t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + t_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    except Exception as e:
        print(f"Parsing Error: {e}")
        
    return frame

def get_working_camera():
    """Tries indices 0 to 2 to find a working camera."""
    for index in range(3):
        # We try with cv2.CAP_DSHOW which is often needed on Windows
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                st.sidebar.success(f"Connected to Camera Index {index}")
                return cap
            else:
                cap.release()
    return None

# --- Main App Loop ---

if not api_key:
    st.warning("Please enter your API Key in the sidebar to start.")
else:
    run_camera = st.checkbox("Start Camera")
    
    # Placeholder for the video frame
    frame_window = st.image([])
    
    # Helper for API status
    status_text = st.empty()

    if run_camera:
        # 1. Find a working camera
        cap = get_working_camera()

        if cap is None:
            st.error("Could not find a working camera (checked indexes 0, 1, and 2). Please check if another app is using it.")
        else:
            # 2. Run the video loop
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.write("Failed to capture video")
                    break
                
                # Convert Frame to Bytes for API
                # Convert BGR (OpenCV) to RGB (PIL/Streamlit)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Encode frame to JPEG bytes
                _, img_encoded = cv2.imencode('.jpg', frame)
                image_bytes = img_encoded.tobytes()
                
                # Call API
                status_text.text("Sending to API...")
                results = run_inference(image_bytes, api_key, conf_threshold, iou_threshold)
                
                # Draw Results
                if results:
                    frame_with_boxes = draw_bbox(frame_rgb.copy(), results)
                    status_text.text("Inference Complete")
                    frame_window.image(frame_with_boxes)
                else:
                    frame_window.image(frame_rgb)
                
                # Optional: Add delay to save API usage if needed
                # time.sleep(0.1)

            # Release camera when the loop ends (user unchecks box)
            cap.release()
