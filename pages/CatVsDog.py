import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from streamlit_drawable_canvas import st_canvas
from pathlib import Path


# --- 2. CSS FOR STYLING ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

/* 🌙 Apply custom font only to main content (ไม่แตะ sidebar) /
div.block-container {
    font-family: 'Poppins', sans-serif;
    background-color: #1E1E1E;
    color: #FFFFFF;
}



/ Body /
html, body {
    background-color: #121212;
    color: #FFFFFF;
}

/ Headings /
h1, h2, h3 {
    color: #25D3D0;
}

/ Buttons /
div.stButton > button {
    background: linear-gradient(45deg, #25D3D0, #8A2BE2);
    color: white;
    border-radius: 25px;
    padding: 12px 25px;
    border: none;
    font-weight: 600;
    transition: all 0.3s ease;
    width: 100%;
    margin-top: 1rem;
    box-shadow: 0 4px 15px 0 rgba(45, 255, 255, 0.5);
}
div.stButton > button:hover {
    transform: scale(1.03);
    box-shadow: 0 6px 20px 0 rgba(45, 255, 255, 0.7);
}

/ Canvas /
[data-testid="stCanvas"] {
    margin: 0 !important;
    padding: 0 !important;
    display: flex;
    justify-content: center;
}
[data-testid="stCanvas"] > canvas {
    border-radius: 15px;
    border: 2px solid #25D3D0;
    background-color: #000000 !important;
}

/ Info box /
div[data-testid="stInfo"] {
    background-color: rgba(37, 211, 208, 0.1);
    border-radius: 10px;
    padding: 1rem;
}

/ Metric cards /
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: #25D3D0 !important;
}
</style>
""", unsafe_allow_html=True)



# --- 3. MODEL LOADING ---
@st.cache_resource
def load_pytorch_model():
    """
    โหลดโมเดล PyTorch ResNet18 ที่แก้ไขแล้ว
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.to(device)

        ckpt = Path("resource/dogvscat/resnet18_dogcat_best.pth")
        if not ckpt.exists():
            st.error(f"🚨 ไม่พบไฟล์โมเดลที่: {ckpt}", icon="🚨")
            return None, None

        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()
        
        return model, device
    except Exception as e:
        st.error(f"[ERROR] ไม่สามารถโหลดโมเดลได้: {e}")
        return None, None

model, device = load_pytorch_model()
if model is None:
    st.error("🚨 ไม่สามารถโหลดโมเดลได้! กรุณาตรวจสอบว่าไฟล์โมเดลมีอยู่จริงและสถาปัตยกรรมถูกต้อง")
    st.stop()


# --- 4. IMAGE PROCESSING & HELPERS ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

CLASS_NAMES = ["Cat 🐱", "Dog 🐶"]

def process_and_predict(image_data: np.ndarray, model, device):
    """
    แปลงข้อมูลจาก canvas, ประมวลผล, และทายผลด้วยโมเดล
    """
    if image_data is None or image_data.max() == 0:
        return None, None, None

    pil_image = Image.fromarray(image_data.astype('uint8'), 'RGBA')
    pil_image_gray = pil_image.convert("L")
    pil_image_inverted = ImageOps.invert(pil_image_gray)
    
    image_tensor = transform(pil_image_inverted).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        prediction_idx = torch.argmax(probabilities).item()
        confidence = probabilities[prediction_idx].item()
    
    predicted_class = CLASS_NAMES[prediction_idx]
    
    return predicted_class, confidence, pil_image_inverted


# --- 5. APP LAYOUT ---

# ส่วนหัวข้อและคำอธิบาย
st.title("🎨 Drawing Classifier")
st.markdown("วาดรูป **หมา** 🐶 หรือ **แมว** 🐱 ด้านล่าง แล้วปล่อยให้ AI ทายผล!")
st.divider()

# --- Main Content ---
# ส่วนพื้นที่วาดรูปและแสดงผลลัพธ์
canvas_col, result_col = st.columns([0.6, 0.4], gap="large")

with canvas_col:
    st.header("🎨 พื้นที่วาดรูป")

    # ส่วนควบคุม
    stroke_width = st.slider("ความหนาของเส้น:", 1, 50, 15)
    drawing_mode = st.selectbox("โหมดการวาด:", ("freedraw", "line", "rect", "circle", "transform"))
    
    # Canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#1E1E1E",
        update_streamlit=True,
        width=1000,
        height=500,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    # ปุ่ม Predict
    predict_button = st.button("✨ ทายผล!", use_container_width=True)

with result_col:
    st.header("🤖 ผลการทำนาย")

    # ใช้ div ที่มี class="result-card" เพื่อแสดงผลลัพธ์ในรูปแบบการ์ด
    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    if predict_button and canvas_result.image_data is not None:
        predicted_class, confidence, processed_image = process_and_predict(
            canvas_result.image_data, model, device
        )
        
        if predicted_class:
            st.markdown("ฉันคิดว่าเป็น...")
            # แสดงผลลัพธ์ในรูปแบบ Badge
            st.markdown(f'<span class="prediction-badge">{predicted_class}</span>', unsafe_allow_html=True)
            st.metric(label="ความมั่นใจ", value=f"{confidence:.2%}")
            
            st.divider()
            
            st.write("**ภาพที่ประมวลผลแล้ว** (สิ่งที่โมเดลเห็น)")
            st.image(processed_image, use_column_width=True)
        else:
            st.warning("ดูเหมือนว่าคุณยังไม่ได้วาดอะไรเลยนะ 🤔")
    else:
        st.info("วาดรูปในช่องด้านซ้าย แล้วกดปุ่ม 'ทายผล' เพื่อดูผลลัพธ์ที่นี่")
    
    st.markdown('</div>', unsafe_allow_html=True)
