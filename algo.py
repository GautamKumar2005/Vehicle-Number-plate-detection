import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import google.generativeai as genai

# Set your Gemini API key
genai.configure(api_key="AIzaSyDvyZIf-3SJ76U-yEDH8ADONPLa1fX1Wz0")

# Function to load YOLO model
def load_yolo(yolo_path):
    labels_path = os.path.sep.join([yolo_path, "coco.names"])
    labels = open(labels_path).read().strip().split("\n")
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    weights_path = os.path.sep.join([yolo_path, "yolov3.weights"])
    config_path = os.path.sep.join([yolo_path, "yolov3.cfg"])

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]  # Updated line

    return net, ln, labels, colors

# Function to classify detected objects using Gemini API
def classify_objects(objects):
    classifications = []
    for obj in objects:
        response = genai.generate_text(
            model="models/gemini-pro",
            prompt=f"Classify the following object: {obj}",
            max_output_tokens=10
        )
        classifications.append(response.text.strip())
    return classifications

# Function to process image using YOLO
def process_image_yolo(image_path, yolo_path, confidence=0.5, threshold=0.3):
    net, ln, labels, colors = load_yolo(yolo_path)

    image = cv2.imread(image_path)
    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    boxes = []
    confidences = []
    class_ids = []
    objects = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]

            if conf > confidence:
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(conf))
                class_ids.append(class_id)
                objects.append(labels[class_id])

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    if len(idxs) > 0:
        classifications = classify_objects([labels[class_ids[i]] for i in idxs.flatten()])
        for i, idx in enumerate(idxs.flatten()):
            (x, y) = (boxes[idx][0], boxes[idx][1])
            (w, h) = (boxes[idx][2], boxes[idx][3])
            color = [0, 0, 255]  # Red color for bounding box
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f} - {}".format(labels[class_ids[idx]], confidences[idx], classifications[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the output image
    output_image_path = "output_image.jpg"
    cv2.imwrite(output_image_path, image)

    return output_image_path, objects, classifications

# Streamlit UI
st.title("Object Detection and Classification with Gemini API")

option = st.selectbox("Select Input Type", ["Image"])

if option == "Image":
    image_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    yolo_path = st.text_input("Enter YOLO model path", "yolo-coco")

    if st.button("Process Image"):
        if image_file is not None:
            image = Image.open(image_file)
            image_path = "uploaded_image." + image.format.lower()
            image.save(image_path)

            output_image_path, objects, classifications = process_image_yolo(image_path, yolo_path)
            st.image(output_image_path, caption="Processed Image", use_column_width=True)

            st.write("Detected Objects and Classifications:")
            for obj, cls in zip(objects, classifications):
                st.write(f"- {obj}: {cls}")