import streamlit as st
import cv2
import requests
import numpy as np
from PIL import Image
import io
import json

# --- Configuration ---
# API Configuration
API_URL = "https://predict.ultralytics.com"
MODEL_URL = "https://hub.ultralytics.com/models/ae7Z7jXfrZNvReWvofz0"

st.set_page_config(page_title="YOLO API Real-Time", layout="centered")

st.title("📷 Real-Time Object Detection (API-Based)")
st.markdown("This app sends webcam frames to the Ultralytics API for inference.")

# --- Sidebar ---
st.sidebar.header("Settings")
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
        st.error(f"API Error: {e}")
        return None

def draw_bbox(frame, results):
    """Parses JSON results and draws bounding boxes on the frame."""
    if not results:
        return frame
        
    # The Ultralytics API usually returns a list of results.
    # We assume the first item corresponds to our single image.
    try:
        data = results[0] if isinstance(results, list) else results
        
        # Depending on API version, boxes might be in 'boxes' or top-level. 
        # We look for 'boxes' containing 'data' or standard list formats.
        detections = []
        
        if 'boxes' in data:
            # Common structure: {'boxes': [{'cls': ..., 'conf': ..., 'box': {'x1':...}}]}
            # Or array based structure. We'll attempt to iterate assuming object structure.
            raw_boxes = data['boxes']
            # If it's the 'data' key inside boxes (newer standard)
            if isinstance(raw_boxes, dict) and 'data' in raw_boxes:
                detections = raw_boxes['data']
            elif isinstance(raw_boxes, list):
                detections = raw_boxes
        
        for det in detections:
            # Extract coordinates. Structure can vary, handling generic JSON response
            # Expected: x1, y1, x2, y2, class, confidence
            
            # Case A: Object with 'box' dictionary
            if 'box' in det:
                x1 = int(det['box']['x1'])
                y1 = int(det['box']['y1'])
                x2 = int(det['box']['x2'])
                y2 = int(det['box']['y2'])
                label = det.get('name', str(det.get('cls', 'Object')))
                conf = det.get('conf', 0)
            
            # Case B: Flat list/dictionary (less common in Hub API but possible)
            else:
                # Fallback or specific parsing logic based on your exact JSON structure
                continue

            # Draw Rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw Label
            text = f"{label} {conf:.2f}"
            t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + t_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    except Exception as e:
        # If parsing fails, print structure to console for debugging
        print(f"Parsing Error: {e}")
        print(f"JSON Structure: {json.dumps(results, indent=2)}")
        
    return frame

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
        cap = cv2.VideoCapture(0)
        
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture video")
                break
            
            # 1. Convert Frame to Bytes for API
            # Convert BGR (OpenCV) to RGB (PIL/Streamlit)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Encode frame to JPEG bytes
            _, img_encoded = cv2.imencode('.jpg', frame)
            image_bytes = img_encoded.tobytes()
            
            # 2. Call API
            status_text.text("Sending to API...")
            results = run_inference(image_bytes, api_key, conf_threshold, iou_threshold)
            
            # 3. Draw Results
            if results:
                # Note: We draw on the original BGR frame or RGB frame depending on preference
                # Let's draw on RGB so colors are correct in Streamlit
                frame_with_boxes = draw_bbox(frame_rgb.copy(), results)
                status_text.text("Inference Complete")
                
                # 4. Display in Streamlit
                frame_window.image(frame_with_boxes)
            else:
                frame_window.image(frame_rgb)
            
            # Optional: Add a small delay if you want to save API credits/bandwidth
            # time.sleep(0.1) 

        cap.release()