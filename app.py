# app.py
import streamlit as st
import joblib
from PIL import Image
import torch
from torchvision import transforms
import cv2
import tempfile

st.set_page_config(page_title="Fake News & Deepfake Demo", layout="wide")
st.title("📰 Fake News & 🖼 Deepfake Detection Demo")

# ---------------------------
# 1️⃣ Fake News Detection
# ---------------------------
st.header("1️⃣ Fake News Detection (Text)")

# Load your trained Fake News model (Scikit-learn / XGBoost / etc.)
@st.cache_resource
def load_fake_news_model():
    return joblib.load("fake_news_model.pkl")  # 👈 put your model path here

fake_news_model = load_fake_news_model()

text_input = st.text_area("Enter text/article for classification:")

if st.button("Check Text"):
    if text_input.strip() != "":
        prediction = fake_news_model.predict([text_input])[0]
        st.write(f"Prediction: **{prediction}**")
    else:
        st.warning("Please enter some text to classify.")

# ---------------------------
# 2️⃣ Deepfake Detection (Images / Videos)
# ---------------------------
st.header("2️⃣ Deepfake Detection (Images / Videos)")

# Load your trained Deepfake Detection model (PyTorch)
@st.cache_resource
def load_deepfake_model():
    model = torch.load("deepfake_detector.pth", map_location="cpu")  # 👈 your model
    model.eval()
    return model

deepfake_model = load_deepfake_model()

file = st.file_uploader("Upload image or video", type=["jpg", "png", "mp4", "mov"])

if file:
    file_type = file.type

    # ---------------- Image ----------------
    if "image" in file_type:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        input_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            output = deepfake_model(input_tensor)
            pred = torch.argmax(output, dim=1).item()

        st.write(f"Deepfake Prediction: **{'FAKE' if pred == 1 else 'REAL'}**")

    # ---------------- Video ----------------
    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.write(f"Video uploaded. Number of frames: {frame_count}")

        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb).convert("RGB")
            st.image(frame_pil, caption="First frame from video", use_column_width=True)

            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

            input_tensor = preprocess(frame_pil).unsqueeze(0)

            with torch.no_grad():
                output = deepfake_model(input_tensor)
                pred = torch.argmax(output, dim=1).item()

            st.write(f"Deepfake Prediction (first frame): **{'FAKE' if pred == 1 else 'REAL'}**")

        cap.release()
