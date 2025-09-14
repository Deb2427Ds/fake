# app.py
import streamlit as st
from transformers import pipeline
from PIL import Image
import torch
from torchvision import models, transforms
import cv2
import tempfile

st.set_page_config(page_title="Fake News & Deepfake Demo", layout="wide")
st.title("📰 Fake News & 🖼 Deepfake Detection Demo")

# ---------------------------
# 1️⃣ Fake News Detection
# ---------------------------
st.header("1️⃣ Fake News Detection (Text)")

text_input = st.text_area("Enter text/article for classification:")

if st.button("Check Text"):
    if text_input.strip() != "":
        # Load pre-trained fake news model from HuggingFace
        classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news")
        result = classifier(text_input)[0]
        st.write(f"Label: **{result['label']}** | Confidence: {result['score']:.2f}")
    else:
        st.warning("Please enter some text to classify.")

# ---------------------------
# 2️⃣ Deepfake Detection (Images / Videos)
# ---------------------------
st.header("2️⃣ Deepfake Detection (Images / Videos)")

file = st.file_uploader("Upload image or video", type=["jpg", "png", "mp4", "mov"])

if file:
    file_type = file.type

    # ---------------- Image ----------------
    if "image" in file_type:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.info("Using pre-trained placeholder deepfake detector (ResNet18)")

        # Preprocess
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(image).unsqueeze(0)

        # Load ResNet18
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.eval()

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)
            top1_prob, top1_catid = torch.max(prob, dim=1)
            # Placeholder: even/odd top1_catid -> FAKE/REAL
            st.write(f"Fake/Real Prediction (placeholder): **{'FAKE' if top1_catid.item()%2==0 else 'REAL'}** | Confidence: {top1_prob.item():.2f}")

    # ---------------- Video ----------------
    elif "video" in file_type:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        cap = cv2.VideoCapture(tfile.name)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.write(f"Video uploaded. Number of frames: {frame_count}")

        ret, frame = cap.read()
        if ret:
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb).convert("RGB")
            st.image(frame_pil, caption="First frame from video", use_column_width=True)

            st.info("Using pre-trained placeholder deepfake detector (ResNet18) on first frame")

            # Preprocess
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(frame_pil).unsqueeze(0)

            # Load ResNet18
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.eval()

            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.softmax(output, dim=1)
                top1_prob, top1_catid = torch.max(prob, dim=1)
                st.write(f"Fake/Real Prediction (placeholder): **{'FAKE' if top1_catid.item()%2==0 else 'REAL'}** | Confidence: {top1_prob.item():.2f}")

        cap.release()
