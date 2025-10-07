import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from streamlit_drawable_canvas import st_canvas
import os

# -----------------------------
# 1. Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2. Load Model (ResNet18)
# -----------------------------
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: cat/dog
model = model.to(device)

# -----------------------------
# 2.1 Safe Path Helper
# -----------------------------
def get_path(*relative_parts):
    """
    Build absolute path relative to project root (one level above this file)
    Example: root/resource/dogvscat/filename.pth
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, ".."))  # go up one level
    return os.path.join(root_dir, *relative_parts)

# -----------------------------
# 2.2 Load Model
# -----------------------------
model_path = get_path("resource", "dogvscat", "resnet18_dogcat_best.pth")

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.warning(f"⚠ Could not load model: {e}. Using untrained model.")
else:
    st.warning("⚠ Model file not found! Please upload or check your path.")

# -----------------------------
# 3. Streamlit Canvas
# -----------------------------
st.title("🐶🐱 Cat vs Dog Drawing Classifier")


canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=500,
    width=500,
    drawing_mode="freedraw",
    key="canvas"
)

# -----------------------------
# 4. Predict Button
# -----------------------------
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert("L")

        # Invert color (black = foreground)
        img = Image.fromarray(255 - np.array(img))

        # Resize for ResNet
        img = img.resize((224, 224))

        # Transform to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        x = transform(img).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(x)
            prob = F.softmax(output, dim=1)
            pred = output.argmax(1).item()
            confidence = prob[0][pred].item()

        st.image(img, caption="Processed Image", width=200)
        st.write(f"**Prediction:** {'🐶 Dog' if pred == 1 else '🐱 Cat'}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")
    else:
        st.warning("✏️ Please draw something first!")