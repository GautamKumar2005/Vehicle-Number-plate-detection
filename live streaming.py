import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import easyocr
import pandas as pd
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import uuid
from util import set_background, write_csv

# Initialize models and settings
LICENSE_MODEL_DETECTION_DIR = r"C:\Users\gauta\.project_ml\Streamlit-ANPR\license_plate_detector.pt"
reader = easyocr.Reader(['en'], gpu=False)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)
output_csv_path = "live_video_results.csv"

# Set background
set_background(r"C:\Users\gauta\.project_ml\Streamlit-ANPR\background.png")

# Helper function to read license plates
def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    if not detections:
        return None
    return " ".join([result[1].upper() for result in detections])

# Custom VideoProcessor for live video
class LicensePlateProcessor(VideoProcessorBase):
    def __init__(self):
        self.results = []
        self.frame_rate = 0
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_copy = img.copy()
        license_detections = license_plate_detector(img_copy)[0]

        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Draw bounding box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

            # Crop and process license plate
            license_plate_crop = img[int(y1):int(y2), int(x1):int(x2)]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            license_plate_text = read_license_plate(license_plate_crop_gray)

            if license_plate_text:
                timestamp = self.frame_count / self.frame_rate if self.frame_rate else 0
                self.results.append({
                    "Timestamp (s)": timestamp,
                    "License Plate": license_plate_text,
                    "Bounding Box": f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
                })

                # Overlay license plate text
                cv2.putText(img, license_plate_text, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        self.frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def save_results(self):
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(output_csv_path, index=False)

# Main Streamlit UI
st.title("💥 License Plate Detection - Live Video 🚗")

if st.sidebar.button("Start Live Detection"):
    st.sidebar.write("Live video detection started...")
    
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    processor = LicensePlateProcessor()
    webrtc_streamer(
        key="license-detection",
        video_processor_factory=lambda: processor,
        rtc_configuration=rtc_config,
    )

    if st.button("Save Results"):
        processor.save_results()
        st.success("Results saved to live_video_results.csv")
        with open(output_csv_path, "rb") as file:
            st.download_button(
                label="Download Results as CSV",
                data=file,
                file_name="live_video_results.csv",
                mime="text/csv",
            )
