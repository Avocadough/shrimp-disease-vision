import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time

# ===============================================================
#                      PAGE CONFIGURATION
# ===============================================================
st.set_page_config(
    page_title="Shrimp Disease Detection",
    page_icon="🦐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================================================
#                        MODEL LOADING
# ===============================================================
@st.cache_resource
def load_model(model_path):
    """Loads the YOLOv8 model from the specified path."""
    model = YOLO(model_path)
    return model

# ===============================================================
#                        DISEASE INFO
# ===============================================================
DISEASE_INFO = {
    'BG': {
        'name': 'Black Gill Disease',
        'description': 'พบจุดสีดำที่บริเวณเหงือก อาจเกิดจากการติดเชื้อหรือสภาพน้ำไม่ดี',
        'severity': 'Medium',
        'color': (75, 183, 209) # BGR format for OpenCV
    },
    'WSSV': {
        'name': 'White Spot Syndrome Virus',
        'description': 'โรคจุดขาว ไวรัสร้ายแรงที่ทำให้กุ้งตายเป็นจำนวนมาก',
        'severity': 'High',
        'color': (78, 205, 196)
    },
    'WSSV_BG': {
        'name': 'WSSV + Black Gill',
        'description': 'ติดเชื้อทั้งสองโรคพร้อมกัน สภาพรุนแรงมาก',
        'severity': 'Critical',
        'color': (255, 107, 107)
    }
}
# Make sure your model's class names match these keys
CLASS_NAMES = ['BG', 'WSSV', 'WSSV_BG']


# ===============================================================
#                         SESSION STATE
# ===============================================================
# Initialize session state variables if they don't exist
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None


# ===============================================================
#                           SIDEBAR
# ===============================================================
with st.sidebar:
    st.title("🦐 Shrimp Disease Detection")
    st.markdown("AI-powered detection for Black Gill, WSSV, and combined infections.")
    st.divider()

    st.header("⚙️ Settings")
    confidence = st.slider(
        "Confidence Threshold", 
        min_value=0.0, max_value=1.0, value=0.25, step=0.05,
        help="Minimum probability to consider a detection valid."
    )
    
    st.divider()
    st.header("📊 Model Info")
    st.info("• **Model:** YOLOv8\n• **Classes:** 3 Diseases")
    
    st.divider()
    st.header("🦠 Disease Types")
    for key, info in DISEASE_INFO.items():
        st.markdown(f"**{info['name']}**")
        st.caption(f"Severity: {info['severity']}")

# ===============================================================
#                         MAIN INTERFACE
# ===============================================================
st.header("Upload Image or Use Camera")

# Create two columns for uploader and camera
uploader_col, camera_col = st.columns(2)
with uploader_col:
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of shrimp"
    )
with camera_col:
     camera_photo = st.camera_input("Or take a photo", key="camera_input")

# Determine which image to use
image_file = uploaded_file if uploaded_file is not None else camera_photo

# Main processing logic
if image_file:
    # Load and store the original image
    st.session_state.original_image = Image.open(image_file).convert("RGB")
    
    # Reset previous results if a new image is uploaded
    if "new_image" not in st.session_state or st.session_state.new_image != image_file:
        st.session_state.new_image = image_file
        st.session_state.detection_results = None

    # Centered Detect Button
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        detect_button = st.button("🚀 Detect Disease", type="primary", use_container_width=True)

    if detect_button:
        with st.spinner("🔍 Analyzing image... Please wait."):
            # Load the model
            model = load_model('C:/Users/Admin/Desktop/กุ้ง/runs/detect/shrimp_disease_v3/weights/best.pt')
            
            # Convert PIL image to numpy array for model prediction
            img_array = np.array(st.session_state.original_image)
            
            # Make prediction
            start_time = time.time()
            results = model.predict(img_array, conf=confidence)
            end_time = time.time()
            
            # Store results in session state
            st.session_state.detection_results = {
                'results': results,
                'inference_time': (end_time - start_time)
            }
            
# Display results if they exist in session state
if st.session_state.detection_results:
    st.divider()
    st.header("🔍 Detection Results")
    
    # Get results from session state
    results = st.session_state.detection_results['results']
    inference_time = st.session_state.detection_results['inference_time']
    
    # Prepare annotated image
    annotated_image = results[0].plot(line_width=2, font_size=10) # Using YOLO's plot function
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # --- Side-by-Side Image Comparison ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(st.session_state.original_image, use_container_width=True) # <-- FIXED
    with col2:
        st.subheader("Annotated Image")
        st.image(annotated_image, use_container_width=True) # <-- FIXED

    # --- Detection Details Section ---
    if results[0].boxes and len(results[0].boxes) > 0:
        st.success(f"✅ Found {len(results[0].boxes)} potential disease area(s).")
        st.caption(f"⚡ Inference time: {inference_time*1000:.1f} ms")
        
        st.subheader("Detection Details:")
        
        for i, box in enumerate(results[0].boxes):
            class_id = int(box.cls[0])
            disease_key = CLASS_NAMES[class_id]
            disease = DISEASE_INFO[disease_key]
            
            # Using a container with a border for each detection
            with st.container(border=True):
                st.markdown(f"**Detection #{i+1}: {disease['name']}**")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Confidence", f"{box.conf[0]:.2%}")
                with c2:
                    severity_color = {'Medium': '🟡', 'High': '🟠', 'Critical': '🔴'}
                    st.metric("Severity", f"{severity_color[disease['severity']]} {disease['severity']}")
                
                st.info(f"ℹ️ **Description:** {disease['description']}")
                
    else:
        st.warning("✅ No specific disease areas were detected at this confidence level.")
        st.info("The shrimp appears to be healthy or the image quality might be insufficient for detection.")
        st.caption(f"⚡ Inference time: {inference_time*1000:.1f} ms")

else:
    st.info("👆 Please upload an image or take a photo to start detection.")

# Footer
st.divider()
st.markdown("<div style='text-align: center;'><p>🦐 Shrimp Disease Detection v2.0 | Powered by YOLOv8 & Streamlit</p></div>", unsafe_allow_html=True)