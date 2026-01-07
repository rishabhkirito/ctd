# streamlit_app.py

import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import cv2
import numpy as np

# --- 1. CONFIGURATION ---

# Path to your fine-tuned model
CUSTOM_YOLO_MODEL_PATH = os.path.join('runs', 'detect', 'high_res_finetune2', 'weights', 'best.pt')

# Configuration for the CLIP model and ZSL classes
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
ZSL_CLASSES = [
    'a T-90 main battle tank, a modern russian tank',
    'a BTR-80, an eight-wheeled boxy armored personnel carrier',
    'a ZSU-23-4 Shilka, a tracked anti-aircraft vehicle with four large cannons',
    'a Leopard 2A7, a modern german main battle tank with a very blocky angular turret',
    'an M2 Bradley, an american tracked infantry fighting vehicle with a prominent turret',
    'a Humvee, a small, wide military light utility truck',
    'an F-22 Raptor, a stealth fighter jet with twin tails and sharp angles',
    'an AH-64 Apache, a military attack helicopter with a front-mounted cannon and side-mounted rocket pods',
    'a CH-47 Chinook, a large transport helicopter with two main rotors',
    'a person in uniform' # Kept the ignore class for filtering
]
IGNORE_CLASS = 'a person in uniform'

# --- 2. MODEL LOADING (with Streamlit Caching) ---

@st.cache_resource
def load_models():
    """Loads all the required models and caches them."""
    print("Loading models...")
    # Check if the custom model path exists before loading
    if not os.path.exists(CUSTOM_YOLO_MODEL_PATH):
        st.error(f"Custom fine-tuned model not found at: {CUSTOM_YOLO_MODEL_PATH}")
        st.stop()
    custom_yolo = YOLO(CUSTOM_YOLO_MODEL_PATH)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    print("Models loaded successfully.")
    return custom_yolo, clip_model, clip_processor, device

# --- 3. CORE PROCESSING PIPELINE ---

def run_pipeline(yolo_model, clip_model, clip_processor, device, original_image, conf_threshold):
    """Runs the full detection and classification pipeline on an image."""
    
    results = yolo_model.predict(source=original_image, conf=conf_threshold, iou=0.5)
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    
    st.write(f"Detected {len(boxes)} potential objects.")

    if len(boxes) == 0:
        # Return the original image if no objects are detected
        cv_image_original = np.array(original_image)
        return cv2.cvtColor(cv_image_original, cv2.COLOR_RGB2BGR)


    cv_image = np.array(original_image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    st.write("Classifying detected objects with CLIP (Zero-Shot)...")
    
    for box in boxes:
        x1, y1, x2, y2 = box
        if (x2 - x1) < 20 or (y2 - y1) < 20: continue

        cropped_image = original_image.crop((x1, y1, x2, y2))
        
        inputs = clip_processor(text=ZSL_CLASSES, images=cropped_image, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
        
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        best_idx = np.argmax(probs)
        predicted_class = ZSL_CLASSES[best_idx]
        confidence = probs[best_idx]
        
        if predicted_class == IGNORE_CLASS and confidence > 0.9:
            continue

        if predicted_class != IGNORE_CLASS and confidence > 0.1:
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            short_label = predicted_class.split(',')[0].replace("a ", "").replace("an ", "")
            label = f"{short_label} ({confidence:.2f})"
            cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return cv_image

# --- 4. STREAMLIT FRONTEND ---

st.set_page_config(layout="wide", page_title="Zero-Shot Military Vehicle Detection")

st.title("👁️ Camouflaged Target Detection")
st.write("Upload an image to identify military vehicles using a custom fine-tuned detection model.")

# Load all models using the cached function
with st.spinner('Loading models, please wait...'):
    custom_yolo, clip_model, clip_processor, device = load_models()

# Sidebar for user inputs
with st.sidebar:
    st.header("Controls")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    # Process button
    process_button = st.button("Process Image")

# Main panel for displaying results
col1, col2 = st.columns(2)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)

    if process_button:
        # Set a constant confidence threshold
        CONF_THRESHOLD = 0.15
        
        with st.spinner('Running detection and classification...'):
            #st.info(f"Using your custom fine-tuned model with a fixed confidence of {CONF_THRESHOLD}.")
            
            processed_image_cv = run_pipeline(custom_yolo, clip_model, clip_processor, device, original_image, CONF_THRESHOLD)
            
            processed_image_rgb = cv2.cvtColor(processed_image_cv, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.subheader("Processed Image")
                st.image(processed_image_rgb, use_container_width=True)
else:
    st.info("Please upload an image to begin.")
