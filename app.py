import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import pandas as pd
import uuid
import os
from streamlit_webrtc import webrtc_streamer
import av
import tempfile
from paddleocr import PaddleOCR
from datetime import datetime
import requests

# Page config
st.set_page_config(
    page_title="License Plate Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === Configure social links (replace with your real profile URLs) ===
SOCIAL_LINKS = {
    "linkedin": "https://www.linkedin.com/in/your-profile",
    "github": "https://github.com/your-profile",
    "instagram": "https://instagram.com/your-profile"
}
AUTHOR_NAME = "Gautam Kumar"
CURRENT_YEAR = datetime.now().year

# === Modern responsive CSS ===
st.markdown(
    f"""
    <style>
    :root {{
        --accent: #2563eb;
        --accent-2: #7c3aed;
        --muted: #6b7280;
        --card-bg: #ffffff;
        --glass: rgba(255,255,255,0.75);
        --radius: 14px;
        --shadow: 0 10px 30px rgba(2,6,23,0.08);
        --max-width: 1200px;
        --gap: 1rem;
    }}

    html, body, .stApp {{
        background: linear-gradient(135deg, #f7fbff 0%, #ffffff 100%);
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }}

    /* container centering */
    .app-wrap {{
        margin: 0 auto;
        max-width: var(--max-width);
        padding: 1rem;
    }}

    .app-header {{
        display:flex;
        gap:1rem;
        align-items:center;
        justify-content:space-between;
        padding: 1rem;
        margin-top:-4rem;
        border-radius: var(--radius);
        background: linear-gradient(90deg, rgba(255,255,255,0.85), rgba(255,255,255,0.6));
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
        flex-wrap:wrap;
    }}
    .brand {{
        display:flex;
        gap:.9rem;
        align-items:center;
    }}
    .logo {{
        width:56px;
        height:56px;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        display:flex;
        align-items:center;
        justify-content:center;
        color:white;
        font-weight:700;
        font-size:18px;
        box-shadow: 0 6px 18px rgba(37,99,235,0.14);
    }}
    .title-block h1 {{
        margin:0;
        font-size:1.2rem;
        letter-spacing: -0.2px;
    }}
    .title-block p {{
        margin:0;
        color:var(--muted);
        font-size:0.9rem;
    }}

    /* main grid */
    .main-grid {{
        display:grid;
        grid-template-columns: 1fr 340px;
        gap: var(--gap);
        align-items:start;
    }}

    .card {{
        background: var(--card-bg);
        padding: 1rem;
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        overflow:hidden;
    }}

    .muted {{
        color: var(--muted);
        font-size:0.95rem;
    }}

    .controls {{
        display:flex;
        gap:0.5rem;
        flex-wrap:wrap;
    }}

    .footer {{
        margin-top:1.2rem;
        padding: 1rem;
        border-radius: 10px;
        text-align:center;
        color: var(--muted);
    }}

    /* social icons */
    .socials {{
        display:flex;
        gap: 0.6rem;
        align-items:center;
        justify-content:center;
        margin-top:0.6rem;
    }}

    .socials a {{
        display:inline-flex;
        width:38px;
        height:38px;
        align-items:center;
        justify-content:center;
        border-radius:10px;
        background: rgba(15,23,42,0.04);
        transition: transform .12s ease, box-shadow .12s ease;
        text-decoration:none;
    }}
    .socials a:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(15,23,42,0.08);
    }}
    .socials svg {{
        width:20px;
        height:20px;
    }}

    /* small screens */
    @media (max-width: 1000px) {{
        .main-grid {{
            grid-template-columns: 1fr;
        }}
        .app-header {{
            gap: .6rem;
        }}
        .logo {{ width:48px;height:48px;font-size:16px;border-radius:10px }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Detection crops folder (neutral, in system temp)
DETECTIONS_DIR = os.path.join(tempfile.gettempdir(), "license_plate_crops")
os.makedirs(DETECTIONS_DIR, exist_ok=True)

# Model paths (keep your paths here or change to environment/config)
LICENSE_MODEL_DETECTION_DIR = "./license_plate_detector.pt"
COCO_MODEL_DIR = "../models/yolov8n.pt"

# Initialize OCR (this is lightweight until used)
reader = PaddleOCR(use_angle_cls=True, lang='en')

# === Header / Hero ===
st.markdown('<div class="app-wrap">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="app-header">
        <div class="brand">
            <div class="logo">ANPR</div>
            <div class="title-block">
                <h1>License Plate Detection</h1>
                <p class="muted">YOLOv8 + PaddleOCR — Fast detection & OCR for vehicle plates</p>
            </div>
        </div>
        <div class="muted">
            Built with Streamlit • Upload images, process video, or run live webcam detection
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar controls (modes + helpful info)
with st.sidebar:
    st.markdown("## Mode")
    options = ["RTSP", "Video", "Live", "Image"]
    choice = st.radio("Select input mode", options, index=3)

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        """
        - Uses YOLO for car + license plate detection and PaddleOCR for recognition.
        - A blur-guard rejects very blurry frames to avoid bad OCR.
        - Crops are stored temporarily on the server.
        """
    )

# Helper functions (same logic, small UI-friendly tweaks)
def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    denoised = cv2.fastNlMeansDenoising(equalized, h=30)
    sharp = cv2.addWeighted(gray, 1.5, denoised, -0.5, 0)
    return sharp

def is_image_blurry(image, threshold=100):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = image.copy()
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def read_license_plate(license_plate_crop, img):
    scores = 0
    detections = reader.ocr(license_plate_crop)
    if not detections:
        return None, None

    rectangle_size = license_plate_crop.shape[0] * license_plate_crop.shape[1]
    plate = []

    for result in detections:
        if not result:
            continue
        for line in result:
            bbox, text, score = line[0], line[1][0], line[1][1]
            length = np.linalg.norm(np.array(bbox[1]) - np.array(bbox[0]))
            height = np.linalg.norm(np.array(bbox[2]) - np.array(bbox[1]))

            if 4 <= len(text) <= 12 and length * height / rectangle_size > 0.10:
                text = text.upper()
                scores += score
                plate.append(text)

    if plate:
        return " ".join(plate), scores / len(plate)
    return None, 0

def write_csv(results, file_path):
    if not results:
        return
    data = []
    for key, value in results.items():
        car_bbox = value['car']['bbox']
        car_score = value['car']['car_score']
        license_plate_bbox = value['license_plate']['bbox']
        license_plate_text = value['license_plate']['text']
        license_plate_bbox_score = value['license_plate']['bbox_score']
        license_plate_text_score = value['license_plate']['text_score']
        timestamp = value['timestamp']

        data.append([
            car_bbox, car_score,
            license_plate_bbox, license_plate_text,
            license_plate_bbox_score, license_plate_text_score,
            timestamp
        ])

    df = pd.DataFrame(data, columns=[
        'Car BBox', 'Car Score', 'License Plate BBox', 'License Plate Text',
        'License Plate BBox Score', 'License Plate Text Score', 'Timestamp'
    ])
    df.to_csv(file_path, index=False)

# Initialize models lazily to avoid long startup times if not needed
@st.cache_resource
def load_models(coco_path, license_path):
    coco_model = YOLO(coco_path)
    license_model = YOLO(license_path)
    return coco_model, license_model

# Load models using constants (not shown on frontend)
try:
    coco_model, license_plate_detector = load_models(COCO_MODEL_DIR, LICENSE_MODEL_DETECTION_DIR)
except Exception:
    st.warning("Models not loaded yet or not found. Please ensure model files exist at configured paths.")
    coco_model, license_plate_detector = None, None

vehicles = [2]  # COCO class id for cars (common)

# VideoProcessor for webcam/live mode
class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_to_analyze = img.copy()
        if is_image_blurry(img_to_analyze, threshold=60):
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        if license_plate_detector is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        license_detections = license_plate_detector(img_to_analyze)[0]

        if len(license_detections.boxes.cls.tolist()) != 0:
            for license_plate in license_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (16, 185, 129), 3)
                license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]
                license_plate_crop_gray = enhance_image(license_plate_crop)
                license_plate_text, _ = read_license_plate(license_plate_crop_gray, img)
                text = license_plate_text or ""
                cv2.rectangle(img, (int(x1), int(y1) - 28), (int(x2), int(y1)), (255, 255, 255), cv2.FILLED)
                cv2.putText(img, text, (int(x1) + 6, int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Core prediction used by image/video processing (keeps original behavior)
def model_prediction(img):
    if is_image_blurry(img, threshold=50):
        return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]

    if coco_model is None or license_plate_detector is None:
        st.error("Models are not loaded. Check server logs or configured paths.")
        return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]

    license_numbers = 0
    results = {}
    licenses_texts = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    object_detections = coco_model(img)[0]
    license_detections = license_plate_detector(img)[0]

    # draw car boxes
    if len(object_detections.boxes.cls.tolist()) != 0:
        for detection in object_detections.boxes.data.tolist():
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection
            if int(class_id) in vehicles:
                cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (59, 130, 246), 3)
    else:
        xcar1, ycar1, xcar2, ycar2 = 0, 0, 0, 0
        car_score = 0

    if len(license_detections.boxes.cls.tolist()) != 0:
        license_plate_crops_total = []
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (16, 185, 129), 3)
            license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]
            # Save neutral crop
            img_name = f'{uuid.uuid1()}.jpg'
            cv2.imwrite(os.path.join(DETECTIONS_DIR, img_name), license_plate_crop)
            license_plate_crop_gray = enhance_image(license_plate_crop)
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray, img)

            if license_plate_text and license_plate_text_score:
                licenses_texts.append(license_plate_text)
                license_plate_crops_total.append(license_plate_crop)
                results[license_numbers] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2], 'car_score': car_score},
                    'license_plate': {'bbox': [x1, y1, x2, y2], 'text': license_plate_text, 'bbox_score': score,
                                      'text_score': license_plate_text_score},
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                license_numbers += 1

        write_csv(results, os.path.join(DETECTIONS_DIR, "detection_results.csv"))
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return [img_wth_box, licenses_texts, license_plate_crops_total]
    else:
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return [img_wth_box]

def process_video_to_csv(video_path, csv_output_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) else 25
    frame_count = 0
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / frame_rate

        if is_image_blurry(frame, threshold=50):
            continue

        license_detections = license_plate_detector(frame)[0]
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            license_plate_crop_gray = enhance_image(license_plate_crop)
            license_plate_text, _ = read_license_plate(license_plate_crop_gray, frame)

            if license_plate_text:
                results.append({
                    "Timestamp (s)": timestamp,
                    "License Plate": license_plate_text,
                    "Bounding Box": f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})",
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

    cap.release()
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_output_path, index=False)
    return results_df

def connect_to_rtsp_stream(rtsp_url, csv_output_path):
    cap = cv2.VideoCapture(rtsp_url)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) else 25
    frame_count = 0
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        timestamp = frame_count / frame_rate

        if is_image_blurry(frame, threshold=50):
            continue

        license_detections = license_plate_detector(frame)[0]
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            license_plate_crop_gray = enhance_image(license_plate_crop)
            license_plate_text, _ = read_license_plate(license_plate_crop_gray, frame)

            if license_plate_text:
                results.append({
                    "Timestamp (s)": timestamp,
                    "License Plate": license_plate_text,
                    "Bounding Box": f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})",
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

    cap.release()
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_output_path, index=False)
    return results_df

# === Main UI area (responsive) ===
st.markdown('<div class="main-grid">', unsafe_allow_html=True)

# Left column: inputs and main preview
st.markdown('<div class="card">', unsafe_allow_html=True)

if choice == "Image":
    st.markdown("### Image Detection")
    uploaded = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = np.array(Image.open(uploaded).convert("RGB"))
        st.image(image, use_column_width=True, caption="Input image")
        if st.button("Apply Detection", key="detect_image"):
            results = model_prediction(image)
            if len(results) == 3:
                prediction, texts, license_plate_crop = results[0], results[1], results[2]
                texts = [i for i in texts if i is not None]
                st.image(prediction, use_column_width=True, caption="Prediction")
                for i, crop in enumerate(license_plate_crop):
                    st.image(crop, width=320, caption=f"Crop {i}: {texts[i] if i < len(texts) else 'N/A'}")
                csv_path = os.path.join(DETECTIONS_DIR, "detection_results.csv")
                if os.path.exists(csv_path):
                    with open(csv_path, "rb") as f:
                        st.download_button("Download CSV", f, file_name="detection_results.csv")
            else:
                st.image(results[0], use_column_width=True, caption="No license found")

elif choice == "Video":
    st.markdown("### Video Detection")
    video_file = st.file_uploader("Upload a video (mp4, avi, mov)", type=["mp4", "avi", "mov"])
    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(video_file.read())
            video_path = tmp.name
        st.video(video_path)
        if st.button("Process Video", key="process_video"):
            csv_output_path = os.path.join(DETECTIONS_DIR, "video_results.csv")
            with st.spinner("Processing video..."):
                results_df = process_video_to_csv(video_path, csv_output_path)
            st.success("Video processing finished")
            st.dataframe(results_df)
            with open(csv_output_path, "rb") as f:
                st.download_button("Download CSV", f, file_name="video_results.csv")

elif choice == "RTSP":
    st.markdown("### RTSP Stream")
    rtsp_url = st.text_input("Enter RTSP URL")
    if rtsp_url and st.button("Process RTSP Stream"):
        csv_output_path = os.path.join(DETECTIONS_DIR, "rtsp_results.csv")
        with st.spinner("Processing RTSP stream..."):
            results_df = connect_to_rtsp_stream(rtsp_url, csv_output_path)
        st.success("RTSP processing done")
        st.dataframe(results_df)
        with open(csv_output_path, "rb") as f:
            st.download_button("Download CSV", f, file_name="rtsp_results.csv")

elif choice == "Live":
    st.markdown("### Live webcam detection")
    st.markdown("Click the start button below to begin a live detection session. Bounding boxes with recognized plates will be shown on the video.")
    webrtc_streamer(key="live", video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False})

st.markdown('</div>', unsafe_allow_html=True)

# Right column: status, quick actions
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Status")
model_status = "Loaded" if coco_model is not None and license_plate_detector is not None else "Not loaded"
st.write(f"YOLO & License detector: **{model_status}**")
st.markdown("**Crops folder:** (server temporary)")
st.markdown(f"`{DETECTIONS_DIR}`")

st.markdown("---")
st.markdown("### Quick Actions")
if st.button("Open crops folder (system)"):
    st.write("Crops saved to system temp. Open the folder in your OS file explorer if you need to inspect files.")

if os.path.exists(os.path.join(DETECTIONS_DIR, "detection_results.csv")):
    with open(os.path.join(DETECTIONS_DIR, "detection_results.csv"), "rb") as f:
        st.download_button("Download last detection CSV", f, file_name="detection_results.csv")
else:
    st.info("No detection CSV found yet.")

st.markdown("---")
st.markdown("### Help / Tips")
st.markdown(
    """
    - Use clear, well-lit images for best OCR results.
    - If detections are poor, try increasing resolution or moving camera closer.
    - For deployment, set model files on the server (paths inside the script).
    """
)

# ====================== JUPYTER-STYLE MARKDOWN FOOTER (BEAUTIFUL & RESPONSIVE) ======================

# Raw SVGs from svgrepo.com (resized for perfect display)
linkedin_svg = '''
<svg width="32" height="32" viewBox="0 0 382 382" xmlns="http://www.w3.org/2000/svg">
    <path style="fill:#0A66C2;" d="M347.445,0H34.555C15.471,0,0,15.471,0,34.555v312.889C0,366.529,15.471,382,34.555,382h312.889
        C366.529,382,382,366.529,382,347.444V34.555C382,15.471,366.529,0,347.445,0z M118.207,329.844c0,5.554-4.502,10.056-10.056,10.056
        H65.345c-5.554,0-10.056-4.502-10.056-10.056V150.403c0-5.554,4.502-10.056,10.056-10.056h42.806
        c5.554,0,10.056,4.502,10.056,10.056V329.844z M86.748,123.432c-22.459,0-40.666-18.207-40.666-40.666S64.289,42.1,86.748,42.1
        s40.666,18.207,40.666,40.666S109.208,123.432,86.748,123.432z M341.91,330.654c0,5.106-4.14,9.246-9.246,9.246H286.73
        c-5.106,0-9.246-4.14-9.246-9.246v-84.168c0-12.556,3.683-55.021-32.813-55.021c-28.309,0-34.051,29.066-35.204,42.11v97.079
        c0,5.106-4.139,9.246-9.246,9.246h-44.426c-5.106,0-9.246-4.14-9.246-9.246V149.593c0-5.106,4.14-9.246,9.246-9.246h44.426
        c5.106,0,9.246,4.14,9.246,9.246v15.655c10.497-15.753,26.097-27.912,59.312-27.912c73.552,0,73.131,68.716,73.131,106.472
        L341.91,330.654L341.91,330.654z"/>
</svg>
'''

instagram_svg = '''
<svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path fill-rule="evenodd" clip-rule="evenodd" d="M12 2C9.284 2 8.944 2.01133 7.87733 2.06C6.81267 2.10867 6.08533 2.278 5.44933 2.52533C4.78267 2.776 4.178 3.16933 3.678 3.67867C3.16948 4.17809 2.77591 4.78233 2.52467 5.44933C2.27867 6.08533 2.10867 6.81333 2.06 7.878C2.012 8.944 2 9.28333 2 12C2 14.7167 2.01133 15.056 2.06 16.1227C2.10867 17.1873 2.278 17.9147 2.52533 18.5507C2.776 19.2173 3.16933 19.822 3.67867 20.322C4.1781 20.8305 4.78234 21.2241 5.44933 21.4753C6.08533 21.722 6.81267 21.8913 7.87733 21.94C8.944 21.9887 9.284 22 12 22C14.716 22 15.056 21.9887 16.1227 21.94C17.1873 21.8913 17.9147 21.722 18.5507 21.4747C19.2173 21.224 19.822 20.8307 20.322 20.3213C20.8305 19.8219 21.2241 19.2177 21.4753 18.5507C21.722 17.9147 21.8913 17.1873 21.94 16.1227C21.9887 15.056 22 14.716 22 12C22 9.284 21.9887 8.944 21.94 7.87733C21.8913 6.81267 21.722 6.08533 21.4747 5.44933C21.2236 4.78204 20.83 4.17755 20.3213 3.678C19.8219 3.16948 19.2177 2.77591 18.5507 2.52467C17.9147 2.27867 17.1867 2.10867 16.122 2.06C15.056 2.012 14.7167 2 12 2ZM12 3.802C14.67 3.802 14.9867 3.812 16.0413 3.86C17.016 3.90467 17.5453 4.06667 17.898 4.20467C18.3647 4.38533 18.698 4.60267 19.048 4.952C19.398 5.302 19.6147 5.63533 19.7953 6.102C19.9327 6.45467 20.0953 6.984 20.14 7.95867C20.188 9.01333 20.198 9.33 20.198 12C20.198 14.67 20.188 14.9867 20.14 16.0413C20.0953 17.016 19.9333 17.5453 19.7953 17.898C19.6353 18.3324 19.3799 18.7253 19.048 19.048C18.7254 19.38 18.3324 19.6354 17.898 19.7953C17.5453 19.9327 17.016 20.0953 16.0413 20.14C14.9867 20.188 14.6707 20.198 12 20.198C9.32933 20.198 9.01333 20.188 7.95867 20.14C6.984 20.0953 6.45467 19.9333 6.102 19.7953C5.66764 19.6353 5.27467 19.3799 4.952 19.048C4.62012 18.7253 4.36475 18.3323 4.20467 17.898C4.06733 17.5453 3.90467 17.016 3.86 16.0413C3.812 14.9867 3.802 14.67 3.802 12C3.802 9.33 3.812 9.01333 3.86 7.95867C3.90467 6.984 4.06667 6.45467 4.20467 6.102C4.38533 5.63533 4.60267 5.302 4.952 4.952C5.27463 4.62003 5.66761 4.36465 6.102 4.20467C6.45467 4.06733 6.984 3.90467 7.95867 3.86C9.01333 3.812 9.33 3.802 12 3.802Z" fill="#E4405F"/>
    <path fill-rule="evenodd" clip-rule="evenodd" d="M12 15.3367C11.5618 15.3367 11.128 15.2504 10.7231 15.0827C10.3183 14.915 9.95047 14.6692 9.64064 14.3594C9.3308 14.0495 9.08502 13.6817 8.91734 13.2769C8.74965 12.8721 8.66335 12.4382 8.66335 12C8.66335 11.5618 8.74965 11.1279 8.91734 10.7231C9.08502 10.3183 9.3308 9.95046 9.64064 9.64062C9.95047 9.33078 10.3183 9.08501 10.7231 8.91732C11.128 8.74964 11.5618 8.66333 12 8.66333C12.885 8.66333 13.7336 9.01487 14.3594 9.64062C14.9851 10.2664 15.3367 11.1151 15.3367 12C15.3367 12.8849 14.9851 13.7336 14.3594 14.3594C13.7336 14.9851 12.885 15.3367 12 15.3367ZM12 6.86C10.6368 6.86 9.32942 7.40153 8.36549 8.36547C7.40155 9.32941 6.86002 10.6368 6.86002 12C6.86002 13.3632 7.40155 14.6706 8.36549 15.6345C9.32942 16.5985 10.6368 17.14 12 17.14C13.3632 17.14 14.6706 16.5985 15.6345 15.6345C16.5985 14.6706 17.14 13.3632 17.14 12C17.14 10.6368 16.5985 9.32941 15.6345 8.36547C14.6706 7.40153 13.3632 6.86 12 6.86ZM18.6353 6.76667C18.6353 7.0889 18.5073 7.39794 18.2795 7.6258C18.0516 7.85366 17.7426 7.98167 17.4204 7.98167C17.0981 7.98167 16.7891 7.85366 16.5612 7.6258C16.3334 7.39794 16.2053 7.0889 16.2053 6.76667C16.2053 6.44443 16.3334 6.13539 16.5612 5.90753C16.7891 5.67968 17.0981 5.55167 17.4204 5.55167C17.7426 5.55167 18.0516 5.67968 18.2795 5.90753C18.5073 6.13539 18.6353 6.44443 18.6353 6.76667Z" fill="#E4405F"/>
</svg>
'''

github_svg = '''
<svg width="32" height="32" viewBox="0 -0.5 25 25" xmlns="http://www.w3.org/2000/svg" fill="#ffffgh">
    <path d="m12.301 0h.093c2.242 0 4.34.613 6.137 1.68l-.055-.031c1.871 1.094 3.386 2.609 4.449 4.422l.031.058c1.04 1.769 1.654 3.896 1.654 6.166 0 5.406-3.483 10-8.327 11.658l-.087.026c-.063.02-.135.031-.209.031-.162 0-.312-.054-.433-.144l.002.001c-.128-.115-.208-.281-.208-.466 0-.005 0-.01 0-.014v.001q0-.048.008-1.226t.008-2.154c.007-.075.011-.161.011-.249 0-.792-.323-1.508-.844-2.025.618-.061 1.176-.163 1.718-.305l-.076.017c.573-.16 1.073-.373 1.537-.642l-.031.017c.508-.28.938-.636 1.292-1.058l.006-.007c.372-.476.663-1.036.84-1.645l.009-.035c.209-.683.329-1.468.329-2.281 0-.045 0-.091-.001-.136v.007c0-.022.001-.047.001-.072 0-1.248-.482-2.383-1.269-3.23l.003.003c.168-.44.265-.948.265-1.479 0-.649-.145-1.263-.404-1.814l.011.026c-.115-.022-.246-.035-.381-.035-.334 0-.649.078-.929.216l.012-.005c-.568.21-1.054.448-1.512.726l.038-.022-.609.384c-.922-.264-1.981-.416-3.075-.416s-2.153.152-3.157.436l.081-.02q-.256-.176-.681-.433c-.373-.214-.814-.421-1.272-.595l-.066-.022c-.293-.154-.64-.244-1.009-.244-.124 0-.246.01-.364.03l.013-.002c-.248.524-.393 1.139-.393 1.788 0 .531.097 1.04.275 1.509l-.01-.029c-.785.844-1.266 1.979-1.266 3.227 0 .025 0 .051.001.076v-.004c-.001.039-.001.084-.001.13 0 .809.12 1.591.344 2.327l-.015-.057c.189.643.476 1.202.85 1.693l-.009-.013c.354.435.782.793 1.267 1.062l.022.011c.432.252.933.465 1.46.614l.046.011c.466.125 1.024.227 1.595.284l.046.004c-.431.428-.718 1-.784 1.638l-.001.012c-.207.101-.448.183-.699.236l-.021.004c-.256.051-.549.08-.85.08-.022 0-.044 0-.066 0h.003c-.394-.008-.756-.136-1.055-.348l.006.004c-.371-.259-.671-.595-.881-.986l-.007-.015c-.198-.336-.459-.614-.768-.827l-.009-.006c-.225-.169-.49-.301-.776-.38l-.016-.004-.32-.048c-.023-.002-.05-.003-.077-.003-.14 0-.273.028-.394.077l.007-.003q-.128.072-.08.184c.039.086.087.16.145.225l-.001-.001c.061.072.13.135.205.19l.003.002.112.08c.283.148.516.354.693.603l.004.006c.191.237.359.505.494.792l.01.024.16.368c.135.402.38.738.7.981l.005.004c.3.234.662.402 1.057.478l.016.002c.33.064.714.104 1.106.112h.007c.045.002.097.002.15.002.261 0 .517-.021.767-.062l-.027.004.368-.064q0 .609.008 1.418t.008.873v.014c0 .185-.08.351-.208.466h-.001c-.119.089-.268.143-.431.143-.075 0-.147-.011-.214-.032l.005.001c-4.929-1.689-8.409-6.283-8.409-11.69 0-2.268.612-4.393 1.681-6.219l-.032.058c1.094-1.871 2.609-3.386 4.422-4.449l.058-.031c1.739-1.034 3.835-1.645 6.073-1.645h.098-.005zm-7.64 17.666q.048-.112-.112-.192-.16-.048-.208.032-.048.112.112.192.144.096.208-.032zm.497.545q.112-.08-.032-.256-.16-.144-.256-.048-.112.08.032.256.159.157.256.047zm.48.72q.144-.112 0-.304-.128-.208-.272-.096-.144.08 0 .288t.272.112zm.672.673q.128-.128-.064-.304-.192-.192-.32-.048-.144.128.064.304.192.192.320 .044zm.913.4q.048-.176-.208-.256-.24-.064-.304.112t.208.24q.24.097.304-.096zm1.009.08q0-.208-.272-.176-.256 0-.256.176 0 .208.272.176.256.001.256-.175zm.929-.16q-.032-.176-.288-.144-.256.048-.224.24t.288.128.225-.224z"/>
</svg>
'''

# Your details
AUTHOR_NAME = "Gautam Kumar"
SOCIAL_LINKS = {
    "linkedin": "https://www.linkedin.com/in/gautam-kumar-489903281/",   # ← Update
    "github": "https://github.com/GautamKumar2005",            # ← Update
    "instagram": "https://instagram.com/yourprofile"        # ← Update
}

# Markdown-style footer using Streamlit markdown + HTML
st.markdown("---")

st.markdown(f"""
<div style="text-align: center; padding: 3rem 1rem; font-family: 'Segoe UI', sans-serif;">


### Made by **{AUTHOR_NAME}**

<div style="color: #1C1C1C; font-size: 1.1rem; margin: 1rem 0;">
© 2025 All Rights Reserved • Delhi, India
</div>

<div style="display: flex; justify-content: center; gap: 2rem; margin: 2.5rem 0; flex-wrap: wrap;">
    <a href="{SOCIAL_LINKS['linkedin']}" target="_blank">{linkedin_svg}</a>
    <a href="{SOCIAL_LINKS['github']}" target="_blank">{github_svg}</a>
    <a href="{SOCIAL_LINKS['instagram']}" target="_blank">{instagram_svg}</a>
</div>

<div style="color: #1C1C1C; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.2);">
🔍 Automatic Number Plate Recognition • Real-time Detection • Open Source Inspired
</div>

<div style="font-size: 0.9rem; color: #1C1C1C; margin-top: 0.5rem;">
Now • {datetime.now().strftime('%B %d, %Y • %H:%M:%S IST')}
</div>

</div>
""", unsafe_allow_html=True)