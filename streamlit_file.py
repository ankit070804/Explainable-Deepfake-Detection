import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import tempfile

from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Explainable Deepfake Detection",
    layout="wide"
)

st.title("🔍 Explainable Deepfake Detection System")

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2
    )

    model.load_state_dict(
        torch.load(
            r"E:\7th sem\pro\dataset5\efficientnet_b0.pth",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model

model = load_model()

# ----------------------------
# Image Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

classes = ["fake", "real"]

# ----------------------------
# Upload Image
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload Face Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, width=300)

    # ----------------------------
    # Prediction
    # ----------------------------
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    prediction = classes[predicted.item()]
    confidence_score = confidence.item() * 100

    # ----------------------------
    # Display Prediction
    # ----------------------------
    st.subheader("Prediction")

    if prediction == "fake":
        st.error(
            f"⚠️ Deepfake Detected ({confidence_score:.2f}%)"
        )
    else:
        st.success(
            f"✅ Real Image ({confidence_score:.2f}%)"
        )

    # ----------------------------
    # Grad-CAM
    # ----------------------------
    rgb_img = np.array(image.resize((224, 224)))

    rgb_float = rgb_img.astype(np.float32) / 255.0

    target_layers = [model.features[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    visualization = show_cam_on_image(
        rgb_float,
        grayscale_cam,
        use_rgb=True
    )

    # ----------------------------
    # Show Heatmap
    # ----------------------------
    st.subheader("Grad-CAM Explanation")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            rgb_img,
            caption="Original Image"
        )

    with col2:
        st.image(
            visualization,
            caption="Grad-CAM Heatmap"
        )