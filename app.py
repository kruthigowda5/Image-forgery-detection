import streamlit as st
import torch
import timm
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime
import shutil
import io

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Image Forgery Detection", layout="wide")

# ================== SESSION STATE ==================
if "show_history" not in st.session_state:
    st.session_state.show_history = False

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# ================== TOP BAR ==================
left, right = st.columns([9,1])
with left:
    st.markdown("## Image Forgery Detection")
with right:
    theme_choice = st.selectbox(
        " ",
        ["Dark", "Light"],
        index=0 if st.session_state.theme == "Dark" else 1
    )
    st.session_state.theme = theme_choice

# ================== THEME ==================
if st.session_state.theme == "Dark":
    st.markdown("""
    <style>
    #MainMenu, footer {visibility:hidden;}
    .stApp {background:#020617;color:#e5e7eb;font-family:'Segoe UI';}
    h1,h2,h3 {color:#f97316;}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    #MainMenu, footer {visibility:hidden;}
    .stApp {background:white;color:black;font-family:'Segoe UI';}
    h1,h2,h3 {color:#1e3a8a;}
    </style>
    """, unsafe_allow_html=True)

# ================== MODEL ==================
THRESHOLD = 0.70

model = timm.create_model("resnet18", pretrained=False, num_classes=1)
model.load_state_dict(torch.load("casia_forgery_model.pt", map_location="cpu"))
model.eval()

# ================== PREPROCESS ==================
def highpass(img):
    kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def predict_image(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = highpass(img)
    img = cv2.resize(img, (224,224))
    img = img.astype("float32") / 255.0
    img = np.transpose(img, (2,0,1))
    img = torch.tensor(img).unsqueeze(0)
    with torch.no_grad():
        return torch.sigmoid(model(img)).item()

# ================== PDF REPORT ==================
def generate_pdf_report(results):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, h-50, "Image Forgery Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(40, h-80, f"Generated on: {datetime.now().strftime('%d %b %Y %H:%M')}")

    y = h - 130

    for name, img, label, prob in results:
        if y < 250:
            c.showPage()
            y = h - 80

        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, f"{name}")
        y -= 20

        c.setFont("Helvetica", 11)
        c.drawString(40, y, f"Result: {label}")
        y -= 15
        c.drawString(40, y, f"Confidence: {prob*100:.2f}%")
        y -= 15

        img_reader = ImageReader(img)
        c.drawImage(img_reader, 40, y-160, width=160, height=160)

        y -= 190

    c.save()
    buffer.seek(0)
    return buffer

# ================== UPLOAD ==================
st.markdown("Upload one or more images to detect forgery.")

uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

results = []

# ================== RESULTS ==================
if uploaded_files:

    st.markdown("##  Detection Results")
    cols = st.columns(3)
    col_idx = 0

    os.makedirs("history", exist_ok=True)

    for uploaded in uploaded_files:
        image = Image.open(uploaded).convert("RGB")
        prob = predict_image(image)

        if prob >= THRESHOLD:
            label = "FORGED"
            color = "#ef4444"
        else:
            label = "REAL"
            color = "#22c55e"

        results.append((uploaded.name, image, label, prob))

        with cols[col_idx]:
            st.image(image, use_column_width=True)
            st.markdown(f"""
            <div style="padding:10px;border-radius:10px;
                        border:2px solid {color};
                        text-align:center;font-weight:700;
                        color:{color};background:#020617;">
                {label}<br>{prob*100:.2f} %
            </div>
            """, unsafe_allow_html=True)

        image.save(f"history/{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png")

        col_idx += 1
        if col_idx == 3:
            cols = st.columns(3)
            col_idx = 0

    # ================== DOWNLOAD PDF ==================
    st.markdown("##  Download Scan Report")
    pdf_buffer = generate_pdf_report(results)

    st.download_button(
        "⬇ Download PDF Report",
        pdf_buffer,
        file_name="forgery_scan_report.pdf",
        mime="application/pdf"
    )

# ================== HISTORY ==================
st.markdown("---")

if st.button(" View History"):
    st.session_state.show_history = not st.session_state.show_history

if st.session_state.show_history:
    st.markdown("## Scan History")

    if st.button("🗑 Delete History"):
        shutil.rmtree("history", ignore_errors=True)
        os.makedirs("history", exist_ok=True)
        st.success("History deleted.")

    if os.listdir("history"):
        cols = st.columns(4)
        i = 0
        for f in sorted(os.listdir("history"), reverse=True):
            with cols[i]:
                st.image("history/" + f, use_column_width=True)
            i = (i + 1) % 4
    else:
        st.info("No history available.")
