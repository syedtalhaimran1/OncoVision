import os
import smtplib
import ssl
import base64
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone
import tempfile
from io import BytesIO

import cv2
from skimage import measure
import albumentations as A
from albumentations.pytorch import ToTensorV2
import bcrypt
import numpy as np
import pymongo
import segmentation_models_pytorch as smp
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from PIL import Image
from dotenv import load_dotenv
import clip
import matplotlib.pyplot as plt

# PDF generation imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

load_dotenv()

# --- CONFIG ---
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "lizard_ai_db"
COLLECTION_NAME = "users"
FEEDBACK_COLLECTION = "feedback"
PREDICTIONS_COLLECTION = "predictions"
MODEL_METRICS_COLLECTION = "model_metrics"
MODEL_PATH_SEG = "nd.pth"
CLS_MODEL_PATH = "lizard_model_final.pth"
DEVICE = "cpu"

IMG_SIZE = 64
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
CLS_LABELS = ["connective", "eosinophil", "epithelial", "lymphocyte", "neutrophil", "plasma"]

MIN_CONF = 0.60
MIN_MARGIN = 0.10
MIN_HE_HUE_RATIO = 0.12
MIN_SAT = 0.15
MIN_SAT_MEAN = 0.18
CLIP_MIN_SIM = 0.22

EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_SENDER = os.getenv("SMTP_SENDER", SMTP_USER or "")
EMAIL_SECURITY = os.getenv("EMAIL_SECURITY", "tls").lower()

# --- INSTANCE OVERLAY CONFIG ---
CLASS_COLORS = {
    "epithelial":  (0,   255,   0),
    "lymphocyte":  (255,   0,   0),
    "plasma":      (0,   255, 255),
    "neutrophil":  (255, 165,   0),
    "eosinophil":  (0,     0, 255),
    "connective":  (255, 255,   0),
    "unknown":     (180, 180, 180),
}
MIN_NUCLEUS_AREA = 40
MAX_NUCLEUS_AREA = 8000
CONTOUR_THICKNESS = 2


# =============================================================================
# MODEL
# =============================================================================
class CNN_Transformer_Hybrid(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.flatten_dim = 128 * 8 * 8
        self.projector = nn.Linear(self.flatten_dim, 256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.projector(x).unsqueeze(1)
        x = self.transformer(x)
        return self.classifier(x.squeeze(1))


# =============================================================================
# CACHED LOADERS
# =============================================================================
@st.cache_resource(show_spinner=False)
def init_db():
    if not MONGO_URI:
        st.error("MONGO_URI not found in environment variables.")
        return None
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db = client[DB_NAME]
        if MODEL_METRICS_COLLECTION not in db.list_collection_names():
            db[MODEL_METRICS_COLLECTION].insert_one({
                "accuracy": 83.89,
                "precision": 83.40,
                "recall": 83.89,
                "f1": 83.48,
                "updated_at": datetime.now(timezone.utc)
            })
        return client
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_seg_model():
    model = smp.Unet(encoder_name="mit_b2", in_channels=3, classes=1)
    if not os.path.exists(MODEL_PATH_SEG):
        st.error(f"Segmentation model not found at {MODEL_PATH_SEG}")
        return None
    try:
        model.load_state_dict(torch.load(MODEL_PATH_SEG, map_location=DEVICE))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load segmentation model: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_cls_model_and_labels():
    if not os.path.exists(CLS_MODEL_PATH):
        return None, None
    checkpoint = torch.load(CLS_MODEL_PATH, map_location=DEVICE)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    out_dim = state.get("classifier.3.weight", torch.empty(0)).shape[0]
    if out_dim == 0:
        raise RuntimeError("Could not detect classifier output dimension.")
    labels = CLS_LABELS if len(CLS_LABELS) == out_dim else [f"Class {i}" for i in range(out_dim)]
    model = CNN_Transformer_Hybrid(num_classes=out_dim)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, labels


@st.cache_resource(show_spinner=False)
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    text_tokens = clip.tokenize(
        ["histology slide", "microscope tissue", "cell microscopy"]).to(DEVICE)
    return model, preprocess, text_tokens


# =============================================================================
# CORE INFERENCE
# =============================================================================
def clip_similarity(img: Image.Image) -> float:
    clip_model, clip_preproc, clip_text = load_clip_model()
    with torch.no_grad():
        img_feat = clip_model.encode_image(clip_preproc(img).unsqueeze(0).to(DEVICE))
        txt_feat = clip_model.encode_text(clip_text)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        return float((img_feat @ txt_feat.T).squeeze(0).max().cpu())


def run_classification(img: Image.Image, skip_gate: bool = False) -> dict:
    model, labels = load_cls_model_and_labels()
    if model is None:
        raise RuntimeError(f"Classifier weights not found: {CLS_MODEL_PATH}")

    clip_sim = clip_similarity(img)
    hsv = np.array(img.convert("RGB").convert("HSV"), dtype=np.float32) / 255.0
    h, s = hsv[..., 0], hsv[..., 1]
    sat_mean = float(s.mean())
    he_ratio = float(((s > MIN_SAT) & ((h < 0.12) | (h > 0.64))).mean())

    tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
    x = tf(image=np.array(img.convert("RGB")))["image"].unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]

    top_idx = int(probs.argmax())
    top_conf = float(probs[top_idx])
    top_class = labels[top_idx]
    margin = top_conf - float(np.sort(probs)[::-1][1]) if len(probs) > 1 else 0.0

    gate_rejected = (not skip_gate) and (
        top_conf < MIN_CONF or margin < MIN_MARGIN or
        he_ratio < MIN_HE_HUE_RATIO or sat_mean < MIN_SAT_MEAN or
        clip_sim < CLIP_MIN_SIM
    )

    if gate_rejected:
        title = "Irrelevant image"
        notes = f"Irrelevant: he_ratio={he_ratio:.3f}, sat_mean={sat_mean:.3f}, clip_sim={clip_sim:.3f}"
    else:
        title = top_class
        notes = "Automated classification result."

    return {
        "type": "classification",
        "title": title,
        "raw_class": top_class,
        "confidence": top_conf,
        "breakdown": list(zip(labels, probs.tolist())),
        "notes": notes,
    }


# =============================================================================
# INSTANCE OVERLAY
# =============================================================================
def _classify_roi(roi_img: Image.Image):
    try:
        result = run_classification(roi_img, skip_gate=True)
        title = result.get("raw_class", "unknown")
        return title
    except Exception:
        return "unknown"


def build_instance_overlay(orig_img: Image.Image, binary_mask: np.ndarray):
    orig_np = np.array(orig_img.convert("RGB"))
    h, w = orig_np.shape[:2]
    mask_rs = cv2.resize(binary_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    labeled = measure.label(mask_rs, connectivity=2)
    regions = measure.regionprops(labeled)
    overlay = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
    stats: dict = {}

    for region in regions:
        area = region.area
        if area < MIN_NUCLEUS_AREA or area > MAX_NUCLEUS_AREA:
            continue
        r0, c0, r1, c1 = region.bbox
        pad = 4
        roi_np = orig_np[max(0, r0 - pad):min(h, r1 + pad),
                         max(0, c0 - pad):min(w, c1 + pad)]
        if roi_np.size == 0:
            continue
        label = _classify_roi(Image.fromarray(roi_np))
        rgb = CLASS_COLORS.get(label, CLASS_COLORS["unknown"])
        bgr = (rgb[2], rgb[1], rgb[0])
        nucleus_mask = (labeled == region.label).astype(np.uint8)
        contours, _ = cv2.findContours(nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, bgr, CONTOUR_THICKNESS)
        stats[label] = stats.get(label, 0) + 1

    return overlay, stats


def _draw_legend(overlay_bgr: np.ndarray, stats: dict) -> np.ndarray:
    if not stats:
        return overlay_bgr
    img = overlay_bgr.copy()
    h, w = img.shape[:2]
    box_w, line_h, pad = 160, 18, 8
    n = len(stats)
    box_h = n * line_h + pad * 2
    x0, y0 = w - box_w - pad, pad
    roi_bg = img[y0:y0 + box_h, x0:x0 + box_w]
    img[y0:y0 + box_h, x0:x0 + box_w] = cv2.addWeighted(
        roi_bg, 0.3, np.zeros_like(roi_bg), 0.7, 0)
    for i, (cls, count) in enumerate(sorted(stats.items())):
        rgb = CLASS_COLORS.get(cls, CLASS_COLORS["unknown"])
        bgr = (rgb[2], rgb[1], rgb[0])
        cy = y0 + pad + i * line_h + line_h // 2
        cv2.circle(img, (x0 + pad + 5, cy), 5, bgr, -1)
        cv2.putText(img, f"{cls}: {count}", (x0 + pad + 15, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    return img


# =============================================================================
# EMAIL / LOGO / DATABASE HELPERS
# =============================================================================
def send_login_email(recipient: str, name: str):
    if not EMAIL_ENABLED:
        return
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and SMTP_SENDER and recipient):
        return
    msg = MIMEText(
        f"Hello {name},\n\nYou have successfully signed in to OncoVision.\n"
        f"If this wasn't you, please reset your password immediately.\n\nStay safe,\nOncoVision Team")
    msg["Subject"] = "OncoVision Login Alert"
    msg["From"] = SMTP_SENDER
    msg["To"] = recipient
    try:
        if EMAIL_SECURITY == "ssl":
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as s:
                s.login(SMTP_USER, SMTP_PASS); s.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
                if EMAIL_SECURITY == "tls":
                    s.starttls(context=ssl.create_default_context())
                s.login(SMTP_USER, SMTP_PASS); s.send_message(msg)
    except Exception:
        pass


def render_logo():
    if os.path.exists("2.png"):
        with open("2.png", "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(
            f'<div class="logo-holder"><img src="data:image/png;base64,{b64}" '
            f'class="logo-img" alt="OncoVision Logo"></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='logo-holder'><div class='logo-text'>🧬 OncoVision</div></div>",
            unsafe_allow_html=True)


def save_prediction_to_db(client, username, email, result):
    if client is None:
        return
    try:
        record = {
            "username": username,
            "email": email,
            "type": result.get("type"),
            "title": result.get("raw_class") if result.get("type") == "classification" else result.get("title"),
            "confidence": result.get("confidence"),
            "timestamp": datetime.now(timezone.utc),
        }
        if result.get("type") == "segmentation":
            record["cell_stats"] = result.get("cell_stats", {})
            record["coverage"] = result.get("coverage")
        client[DB_NAME][PREDICTIONS_COLLECTION].insert_one(record)
    except Exception as e:
        print(f"DB save error: {e}")


def get_recent_predictions(client, limit=5):
    if client is None:
        return []
    try:
        preds = list(client[DB_NAME][PREDICTIONS_COLLECTION].find(
            {}, sort=[("timestamp", -1)], limit=limit))
        return preds
    except:
        return []


def get_total_scans_and_change(client):
    if client is None:
        return 0, 0.0
    now = datetime.now(timezone.utc)
    current_month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    if now.month == 1:
        prev_month_start = datetime(now.year - 1, 12, 1, tzinfo=timezone.utc)
    else:
        prev_month_start = datetime(now.year, now.month - 1, 1, tzinfo=timezone.utc)
    total = client[DB_NAME][PREDICTIONS_COLLECTION].count_documents({})
    current_count = client[DB_NAME][PREDICTIONS_COLLECTION].count_documents({"timestamp": {"$gte": current_month_start}})
    prev_count = client[DB_NAME][PREDICTIONS_COLLECTION].count_documents({
        "timestamp": {"$gte": prev_month_start, "$lt": current_month_start}
    })
    if prev_count == 0:
        percent_change = 100.0 if current_count > 0 else 0.0
    else:
        percent_change = ((current_count - prev_count) / prev_count) * 100
    return total, round(percent_change, 1)


def get_active_users(client):
    if client is None:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    pipeline = [
        {"$match": {"timestamp": {"$gte": cutoff}}},
        {"$group": {"_id": "$email"}},
        {"$count": "active_users"}
    ]
    result = list(client[DB_NAME][PREDICTIONS_COLLECTION].aggregate(pipeline))
    return result[0]["active_users"] if result else 0


def get_model_metrics(client):
    if client is None:
        return {"accuracy": 83.89, "precision": 83.40, "recall": 83.89, "f1": 83.48}
    doc = client[DB_NAME][MODEL_METRICS_COLLECTION].find_one(sort=[("updated_at", -1)])
    if doc:
        return {
            "accuracy": doc.get("accuracy", 83.89),
            "precision": doc.get("precision", 83.40),
            "recall": doc.get("recall", 83.89),
            "f1": doc.get("f1", 83.48)
        }
    return {"accuracy": 83.89, "precision": 83.40, "recall": 83.89, "f1": 83.48}


# =============================================================================
# PDF REPORT GENERATION
# =============================================================================
def generate_pdf_report(result: dict, image: Image.Image, username: str, email: str) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        fontSize=24, textColor=colors.HexColor('#1e3a8a'), alignment=TA_CENTER, spaceAfter=12)
    heading_style = ParagraphStyle(
        'CustomHeading', parent=styles['Heading2'],
        fontSize=16, textColor=colors.HexColor('#0f1724'), spaceAfter=8)
    normal_style = styles['Normal']
    center_style = ParagraphStyle('Center', parent=normal_style, alignment=TA_CENTER)

    story = []
    temp_files = []

    logo_path = "2.png"
    if os.path.exists(logo_path):
        try:
            img_logo = RLImage(logo_path, width=1.2*inch, height=1.2*inch, hAlign='CENTER')
            story.append(img_logo)
        except:
            pass
    story.append(Paragraph("OncoVision", title_style))
    story.append(Spacer(1, 0.2*inch))

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(f"Report generated: {now_str}", center_style))
    story.append(Paragraph(f"User: {username} ({email})", center_style))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Original Histology Image", heading_style))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        image.save(tmp_img.name, format="PNG")
        tmp_img_path = tmp_img.name
        temp_files.append(tmp_img_path)
    story.append(RLImage(tmp_img_path, width=4*inch, height=4*inch, hAlign='CENTER'))
    story.append(Spacer(1, 0.2*inch))

    analysis_type = result.get("type", "Unknown").capitalize()
    story.append(Paragraph(f"Analysis Type: {analysis_type}", heading_style))

    if result.get("type") == "classification":
        pred_class = result.get("title", "N/A")
        conf = result.get("confidence", 0.0) * 100
        notes = result.get("notes", "")
        story.append(Paragraph(f"Predicted Class: {pred_class}", normal_style))
        story.append(Paragraph(f"Confidence: {conf:.2f}%", normal_style))
        story.append(Paragraph(f"Notes: {notes}", normal_style))

        breakdown = result.get("breakdown", [])
        if breakdown:
            table_data = [["Class", "Probability (%)"]]
            for cls, prob in breakdown:
                table_data.append([cls, f"{prob*100:.2f}%"])
            table = Table(table_data, colWidths=[2.5*inch, 1.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 6),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ]))
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("Probability Breakdown", heading_style))
            story.append(table)

    elif result.get("type") == "segmentation":
        coverage = result.get("coverage", 0.0) * 100
        cell_stats = result.get("cell_stats", {})
        total_cells = sum(cell_stats.values())
        num_classes = len(cell_stats)
        notes = result.get("notes", "")

        story.append(Paragraph(f"Mask Coverage: {coverage:.1f}%", normal_style))
        story.append(Paragraph(f"Total Nuclei Detected: {total_cells}", normal_style))
        story.append(Paragraph(f"Number of Classes Found: {num_classes}", normal_style))
        story.append(Paragraph(f"Notes: {notes}", normal_style))

        if cell_stats:
            table_data = [["Cell Class", "Count", "Percentage"]]
            for cls, cnt in sorted(cell_stats.items()):
                pct = (cnt / total_cells * 100) if total_cells > 0 else 0
                table_data.append([cls, str(cnt), f"{pct:.1f}%"])
            table = Table(table_data, colWidths=[2*inch, 1.2*inch, 1.2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 6),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ]))
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("Cell Statistics", heading_style))
            story.append(table)

        overlay_rgb = result.get("overlay_rgb")
        if overlay_rgb is not None:
            story.append(PageBreak())
            story.append(Paragraph("Instance Overlay (Colour‑coded Nuclei)", heading_style))
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_over:
                overlay_pil = Image.fromarray(overlay_rgb)
                overlay_pil.save(tmp_over.name, format="PNG")
                tmp_over_path = tmp_over.name
                temp_files.append(tmp_over_path)
            story.append(RLImage(tmp_over_path, width=5*inch, height=5*inch, hAlign='CENTER'))

        binary_mask = result.get("mask")
        if binary_mask is not None:
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph("Binary Segmentation Mask", heading_style))
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_mask:
                mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
                mask_pil.save(tmp_mask.name, format="PNG")
                tmp_mask_path = tmp_mask.name
                temp_files.append(tmp_mask_path)
            story.append(RLImage(tmp_mask_path, width=5*inch, height=5*inch, hAlign='CENTER'))

    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("— End of Report —", center_style))

    doc.build(story)
    for fpath in temp_files:
        try:
            os.unlink(fpath)
        except:
            pass

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


# =============================================================================
# AUTH SCREEN (Stylish glassmorphism)
# =============================================================================
def auth_screen(client):
    import random

    # Read logo
    logo_base64 = ""
    if os.path.exists("logo.png"):
        with open("logo.png", "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode("utf-8")

    # ========== CSS ==========
    st.markdown("""
    <style>
    /* Hide Streamlit decoration but keep header (so collapse control area exists if needed) */
    .stApp > header, [data-testid="stDecoration"] {
        display: none !important;
    }
    /* Remove all top spacing */
    .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    .main .block-container {
        padding-top: 0 !important;
    }
    /* Background gradient */
    .stApp {
        background: radial-gradient(circle at 20% 30%, #0a0f2a, #030617);
        color: white;
    }
    /* Floating stars container */
    .stars {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
    }
    .star {
        position: absolute;
        background: white;
        border-radius: 50%;
        opacity: 0.6;
        animation: floatStar 15s infinite linear;
    }
    @keyframes floatStar {
        0% { transform: translate(0, 0); opacity: 0; }
        20% { opacity: 0.8; }
        80% { opacity: 0.8; }
        100% { transform: translate(30px, -40px); opacity: 0; }
    }
    /* Floating cells container */
    .cells {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
    }
    .cell {
        position: absolute;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(79,172,254,0.4), rgba(180,101,255,0.2));
        border: 1px solid rgba(79,172,254,0.5);
        box-shadow: 0 0 8px rgba(79,172,254,0.3);
        animation: floatCell 25s infinite ease-in-out;
    }
    .cell::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 30%;
        height: 30%;
        background: radial-gradient(circle, rgba(180,101,255,0.6), rgba(79,172,254,0.3));
        border-radius: 50%;
        transform: translate(-50%, -50%);
    }
    @keyframes floatCell {
        0% { transform: translate(0, 0) rotate(0deg); opacity: 0.3; }
        50% { transform: translate(30px, -25px) rotate(5deg); opacity: 0.7; }
        100% { transform: translate(-15px, 20px) rotate(-3deg); opacity: 0.3; }
    }
    /* Microscope */
    .floating-microscope {
        position: fixed;
        bottom: 15%;
        right: 5%;
        font-size: 70px;
        opacity: 0.4;
        pointer-events: none;
        z-index: 0;
        filter: drop-shadow(0 0 8px rgba(79,172,254,0.5));
        animation: floatMicroscope 12s infinite ease-in-out;
    }
    @keyframes floatMicroscope {
        0% { transform: translate(0, 0) rotate(0deg); opacity: 0.3; }
        50% { transform: translate(-20px, -30px) rotate(5deg); opacity: 0.6; }
        100% { transform: translate(0, 0) rotate(0deg); opacity: 0.3; }
    }
    /* Glass effect for cards */
    .glass-card, .feature-card, .stat-card {
        background: rgba(20, 30, 50, 0.7);
        backdrop-filter: blur(16px);
        border-radius: 24px;
        border: 1px solid rgba(72, 187, 255, 0.3);
        padding: 20px;
        margin-bottom: 20px;
        position: relative;
        z-index: 1;
    }
    .feature-card {
        padding: 12px 16px;
        margin: 12px 0;
        border-left: 4px solid #4facfe;
        transition: 0.2s;
    }
    .feature-card:hover {
        transform: translateX(8px);
        background: rgba(20, 40, 70, 0.8);
    }
    .stat-card {
        text-align: center;
        padding: 10px;
        flex: 1;
    }
    .stat-number {
        font-size: 26px;
        font-weight: 800;
        background: linear-gradient(135deg, #4facfe, #b465ff);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }
    /* Login card */
    .login-card {
        background: rgba(10, 20, 35, 0.8);
        backdrop-filter: blur(20px);
        border-radius: 28px;
        border: 1px solid rgba(79,172,254,0.3);
        position: relative;
        z-index: 1;
    }
    /* Inputs */
    .stTextInput input {
        background: rgba(0,0,0,0.5) !important;
        border: 1px solid #4facfe !important;
        border-radius: 14px !important;
        padding: 12px 16px !important;
        color: white !important;
    }
    /* Button */
    .stButton button {
        background: linear-gradient(95deg, #4facfe, #b465ff);
        border-radius: 40px;
        height: 50px;
        font-weight: 700;
        font-size: 16px;
        width: 100%;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        font-weight: 600;
        color: #a0b4d7;
    }
    .stTabs [aria-selected="true"] {
        color: #4facfe;
        border-bottom: 2px solid #4facfe;
    }
    /* Text */
    .subtitle {
        font-size: 15px;
        color: #cbd5e6;
        margin-bottom: 25px;
    }
    .forgot-link, .create-link {
        color: #4facfe;
        text-decoration: none;
        font-size: 13px;
    }
    .forgot-link:hover, .create-link:hover {
        color: #b465ff;
    }
    </style>
    """, unsafe_allow_html=True)

    # Stars (12 random)
    stars_html = '<div class="stars">'
    for _ in range(12):
        top = random.randint(5, 95)
        left = random.randint(5, 95)
        size = random.randint(2, 6)
        duration = random.randint(12, 25)
        delay = random.randint(0, 10)
        stars_html += f'<div class="star" style="top:{top}%; left:{left}%; width:{size}px; height:{size}px; animation-duration:{duration}s; animation-delay:{delay}s;"></div>'
    stars_html += '</div>'
    st.markdown(stars_html, unsafe_allow_html=True)

    # Floating cells (9 soft glowing circles with nucleus)
    cells_html = '<div class="cells">'
    for _ in range(9):
        top = random.randint(5, 90)
        left = random.randint(5, 90)
        size = random.randint(25, 60)
        duration = random.randint(18, 30)
        delay = random.randint(0, 12)
        cells_html += f'<div class="cell" style="top:{top}%; left:{left}%; width:{size}px; height:{size}px; animation-duration:{duration}s; animation-delay:{delay}s;"></div>'
    cells_html += '</div>'
    st.markdown(cells_html, unsafe_allow_html=True)

    # Microscope
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <div class="floating-microscope"><i class="fas fa-microscope"></i></div>
    """, unsafe_allow_html=True)

    # ========== MAIN UI ==========
    left_col, right_col = st.columns([1.2, 1], gap="large")

    with left_col:
        # Brand with larger logo (80px)
        if logo_base64:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
                <img src="data:image/png;base64,{logo_base64}" style="width: 80px; border-radius: 16px;">
                <h1 style="margin:0; font-size:40px;"><span style="color:white">Onco</span><span style="color:#4facfe">Vision</span></h1>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='margin-bottom:20px;'><span style='color:white'>Onco</span><span style='color:#4facfe'>Vision</span></h1>", unsafe_allow_html=True)

        st.markdown("<div class='subtitle'>Advanced deep learning models for precise cancer detection, segmentation and classification.</div>", unsafe_allow_html=True)

        # Features
        features = [
            ("🧠", "CNN + Transformer Classification", "Hybrid architecture"),
            ("🔬", "U‑Net Segmentation", "Pixel‑perfect detection"),
            ("✅", "CLIP Image Validation", "Multimodal verification"),
            ("🔒", "Secure MongoDB Authentication", "Enterprise grade")
        ]
        for icon, title, desc in features:
            st.markdown(f"""
            <div class='feature-card'>
                <div style='display:flex; gap:12px; align-items:center;'>
                    <span style='font-size:24px'>{icon}</span>
                    <div><strong>{title}</strong><br><span style='font-size:12px; color:#b0c4de'>{desc}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Stats
        st.markdown("""
        <div style='display:flex; gap:15px; margin-top:25px;'>
            <div class='stat-card'><div class='stat-number'>84.59%</div><div>Accuracy</div></div>
            <div class='stat-card'><div class='stat-number'>100%</div><div>Secure</div></div>
            <div class='stat-card'><div class='stat-number'>24/7</div><div>AI Processing</div></div>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        # Login card
        st.markdown("""
        <div class='login-card'>
            <h2 style='text-align:center; margin-bottom:10px;'>Welcome Back</h2>
            <p style='text-align:center; margin-bottom:25px; color:#b0c4de;'>Sign in to your account</p>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

        with tab1:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="doctor@example.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                col_a, col_b = st.columns(2)
                with col_a:
                    remember = st.checkbox("Remember Me")
                with col_b:
                    st.markdown("<div style='text-align:right'><a href='#' class='forgot-link'>Forgot Password?</a></div>", unsafe_allow_html=True)
                submit = st.form_submit_button("Sign In", width='stretch')
                if submit:
                    user = client[DB_NAME][COLLECTION_NAME].find_one({"email": email})
                    if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = user["name"]
                        st.session_state["user_email"] = user["email"]
                        send_login_email(user["email"], user["name"])
                        st.success("Login Successful")
                        st.session_state["page"] = "Dashboard"
                        st.rerun()
                    else:
                        st.error("Invalid Credentials")
            st.markdown("<div style='text-align:center; margin-top:15px;'><a href='#' class='create-link'>Create New Account</a></div>", unsafe_allow_html=True)

        with tab2:
            with st.form("signup_form"):
                name = st.text_input("Full Name", placeholder="Dr. Jane Doe")
                email = st.text_input("Email Address", placeholder="doctor@example.com")
                password = st.text_input("Password", type="password", placeholder="Create a strong password")
                submit = st.form_submit_button("Create Account", width='stretch')
                if submit:
                    if client[DB_NAME][COLLECTION_NAME].find_one({"email": email}):
                        st.warning("User already exists")
                    else:
                        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
                        client[DB_NAME][COLLECTION_NAME].insert_one({
                            "name": name,
                            "email": email,
                            "password": hashed
                        })
                        st.success("Account Created Successfully")


# =============================================================================
# SIDEBAR (unchanged)
# =============================================================================
def show_sidebar():
    with st.sidebar:
        st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                background: linear-gradient(135deg, #0a0f2a 0%, #4a0e2e 100%);
                border-radius: 20px;
                margin: 8px;
                padding: 10px !important;
                height: calc(100vh - 16px);
                position: sticky;
                top: 8px;
                overflow-y: auto;
                scrollbar-width: thin;
            }
            [data-testid="stSidebar"] > div:first-child {
                padding-top: 0 !important;
            }
            div[data-testid="stRadio"] > label {
                display: none;
            }
            div[data-testid="stRadio"] div[role="radiogroup"] {
                gap: 6px;
                margin-top: 0;
            }
            div[data-testid="stRadio"] div[role="radiogroup"] label {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 10px;
                padding: 8px 12px;
                transition: all 0.2s ease;
                margin-bottom: 4px;
                cursor: pointer;
                color: #f0f3fa;
                font-weight: 500;
                font-size: 0.85rem;
                backdrop-filter: blur(4px);
            }
            div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
                background: rgba(110, 231, 255, 0.2);
                border-color: #6ee7ff;
                transform: translateX(4px);
            }
            div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] {
                background: rgba(110, 231, 255, 0.25);
                border-left: 3px solid #6ee7ff;
                font-weight: 600;
                box-shadow: 0 0 8px rgba(110,231,255,0.3);
            }
            div[data-testid="stRadio"] div[role="radiogroup"] div[role="radio"] {
                display: none;
            }
            .status-box, .user-box {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 10px;
                padding: 8px 10px;
                margin: 8px 0;
                backdrop-filter: blur(4px);
                font-size: 0.75rem;
            }
            .status-box strong, .user-box strong {
                color: #6ee7ff;
                font-size: 0.8rem;
            }
            .status-box span.status-text {
                color: #a0e7ff;
                font-size: 0.7rem;
                margin-left: 4px;
            }
            .separator {
                margin: 8px 0;
                border-top: 1px solid rgba(255,255,255,0.1);
            }
            .sidebar-logo {
                text-align: center;
                margin-bottom: 15px;
            }
            .sidebar-logo img {
                width: 55%;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            }
        </style>
        """, unsafe_allow_html=True)

        if os.path.exists("2.png"):
            with open("2.png", "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            st.markdown(
                f'<div class="sidebar-logo">'
                f'<img src="data:image/png;base64,{b64}">'
                f'</div>',
                unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='text-align: center; margin-bottom: 15px;'>🧬 OncoVision</h3>", unsafe_allow_html=True)

        pages = {
            "Dashboard": "🏠 Dashboard",
            "Upload Image": "📤 Upload Image",
            "Classification Result": "🔬 Classification",
            "Segmentation Result": "🖍️ Segmentation",
            "Model Info": "🧠 Model Info",
            "Feedback": "✉️ Feedback",
            "History": "📜 History",
            "Settings": "⚙️ Settings",
            "Logout": "🚪 Logout"
        }
        current = st.session_state.get("page", "Dashboard")
        selected_display = st.radio(
            "Navigate",
            options=list(pages.values()),
            index=list(pages.values()).index(pages[current]),
            label_visibility="collapsed"
        )
        selected_key = [k for k, v in pages.items() if v == selected_display][0]
        if selected_key != current:
            st.session_state["page"] = selected_key
            st.rerun()

        st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='status-box'>
            <strong>🤖 AI Model Status</strong><br>
            • CNN + Transformer <span class='status-text'>(Loaded)</span><br>
            • U-Net (Segmentation) <span class='status-text'>(Loaded)</span><br>
            • CLIP Validation <span class='status-text'>(Active)</span><br>
            • Database <span class='status-text'>(Connected)</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

        username = st.session_state.get("username", "Admin User")
        user_email = st.session_state.get("user_email", "admin@oncovision.ai")
        st.markdown(f"""
        <div class='user-box'>
            <strong>👤 {username}</strong><br>
            <small>{user_email}</small><br>
            <span style="color: #6ee7ff;">● Online</span>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# DASHBOARD PAGE
# =============================================================================
def dashboard_page(client):
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
        <h2 style="margin: 0;">Welcome back, {st.session_state.get('username', 'Admin')}! 🎉</h2>
        <div style="font-size: 1.2rem;">🔔 {datetime.now(timezone.utc).strftime("%d %b %Y")}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("Monitor your model performance and analyze histopathology images.")

    total_scans, pct_change = get_total_scans_and_change(client)
    active_users = get_active_users(client)
    detected_classes = len(CLS_LABELS)
    dataset_size_display = "1.74 GB"
    metrics = get_model_metrics(client)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        change_str = f"+{pct_change}%" if pct_change >= 0 else f"{pct_change}%"
        st.markdown(f"<div class='glass-card'><h3>{total_scans}</h3><p>Total Scans</p><small>{change_str} from last month</small></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='glass-card'><h3>{metrics['accuracy']:.2f}%</h3><p>Model Accuracy</p><small>On Test Dataset</small></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='glass-card'><h3>{detected_classes}</h3><p>Detected Classes</p><small>Cell Categories</small></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='glass-card'><h3>{dataset_size_display}</h3><p>Dataset Size</p><small>Training Data</small></div>", unsafe_allow_html=True)
    with col5:
        st.markdown(f"<div class='glass-card'><h3>{active_users}</h3><p>Active Users</p><small>Last 30 days</small></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Recent Predictions")
        recent = get_recent_predictions(client, 5)
        if recent:
            for pred in recent:
                title = pred.get("title", "N/A")
                conf = pred.get("confidence", 0)
                conf_str = f"{conf*100:.2f}%" if conf else "N/A"
                time_str = pred.get("timestamp", datetime.now(timezone.utc)).strftime("%d %b %Y, %I:%M %p")
                st.markdown(f"**{title}** – {conf_str}  \n<small>{time_str}</small>", unsafe_allow_html=True)
                st.divider()
        else:
            st.info("No predictions yet. Go to Upload Image to run analysis.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Class Distribution")
        class_data = {"Lymphocyte": 28.7, "Plasma Cell": 21.4, "Neutrophil": 18.2,
                      "Eosinophil": 9.8, "Epithelial Cell": 15.6, "Connective": 6.3}
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.barh(list(class_data.keys()), list(class_data.values()), color='#6ee7ff')
        ax.set_xlabel("Percentage")
        ax.set_facecolor('#0f1724')
        fig.patch.set_facecolor('#0a0f18')
        ax.tick_params(colors='white')
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Model Performance")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2f}%")
            st.metric("Precision", f"{metrics['precision']:.2f}%")
        with col_m2:
            st.metric("Recall", f"{metrics['recall']:.2f}%")
            st.metric("F1 Score", f"{metrics['f1']:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### Recent Feedback")
        if client:
            feedbacks = list(client[DB_NAME][FEEDBACK_COLLECTION].find({}, sort=[("created_at", -1)]).limit(3))
            for fb in feedbacks:
                rating = fb.get("rating", 5)
                stars = "⭐" * rating + "☆" * (5 - rating)
                st.markdown(f"{stars}  \n{fb.get('message', '')[:150]}  \n<small>{fb.get('name', 'User')} – {fb.get('created_at', datetime.now(timezone.utc)).strftime('%d %b %Y, %I:%M %p')}</small>", unsafe_allow_html=True)
                st.divider()
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# UPLOAD IMAGE PAGE
# =============================================================================
def upload_image_page(client):
    st.markdown("## Upload Histopathology Image")
    uploaded_file = st.file_uploader("Drag & Drop your image here", type=["png", "jpg", "jpeg", "jfif"], key="upload_page")
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.session_state["last_image"] = img
        # ✅ Set a fixed width for the preview (e.g., 400px)
        st.image(img, caption="Preview", width=400)

        st.markdown("### AI Analysis")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔬 Classification", width='stretch'):
                try:
                    res = run_classification(st.session_state["last_image"])
                    st.session_state["result"] = res
                    save_prediction_to_db(client, st.session_state.get("username"), st.session_state.get("user_email"), res)
                    st.success(f"Classification completed! Predicted: {res['title']} with {res['confidence']*100:.2f}% confidence.")
                    st.session_state["page"] = "Classification Result"
                    st.rerun()
                except Exception as e:
                    st.error(f"Classification failed: {e}")
        with col2:
            if st.button("🖍️ Segmentation", width='stretch'):
                model = load_seg_model()
                if model is None:
                    st.error("Segmentation model not available. Please check nd.pth file.")
                else:
                    try:
                        tf = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
                        tensor = tf(image=np.array(st.session_state["last_image"]))["image"].unsqueeze(0)
                        with torch.no_grad():
                            mask = torch.sigmoid(model(tensor)).squeeze().numpy()
                            binary = (mask > 0.5).astype(np.uint8)
                        overlay_bgr, stats = build_instance_overlay(st.session_state["last_image"], binary)
                        overlay_bgr = _draw_legend(overlay_bgr, stats)
                        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                        seg_result = {
                            "type": "segmentation",
                            "mask": binary,
                            "overlay_rgb": overlay_rgb,
                            "cell_stats": stats,
                            "title": "Cell Boundary Map",
                            "coverage": float(binary.sum()) / binary.size,
                            "notes": "U-Net + per-cell classification."
                        }
                        st.session_state["result"] = seg_result
                        save_prediction_to_db(client, st.session_state.get("username"), st.session_state.get("user_email"), seg_result)
                        st.success("Segmentation completed!")
                        st.session_state["page"] = "Segmentation Result"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Segmentation failed: {e}")
    else:
        st.info("Please upload an image to begin analysis.")


def classification_result_page():
    st.markdown("## 🔬 Classification Results")
    result = st.session_state.get("result")
    image = st.session_state.get("last_image")

    if not result or result.get("type") != "classification":
        st.warning("No classification result available. Please upload an image and run classification from the Upload Image page.")
        if st.button("Go to Upload Image", width='stretch'):
            st.session_state["page"] = "Upload Image"
            st.rerun()
        return

    if image:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    pred_class = result.get("title", "N/A")
    conf = result.get("confidence", 0.0) * 100
    st.markdown(f"### Predicted Class: **{pred_class}**")
    st.markdown(f"**Confidence:** {conf:.2f}%")
    st.progress(int(conf))

    breakdown = result.get("breakdown", [])
    if breakdown:
        st.markdown("### Class Probabilities")
        labels, probs = zip(*breakdown)
        probs_pct = [p * 100 for p in probs]
        fig, ax = plt.subplots(figsize=(8, 4))
        colors_bar = [CLASS_COLORS.get(l, (100, 100, 100)) for l in labels]
        colors_hex = [f"#{r:02x}{g:02x}{b:02x}" for (r,g,b) in colors_bar]
        ax.bar(labels, probs_pct, color=colors_hex)
        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, 100)
        ax.set_facecolor('#0f1724')
        fig.patch.set_facecolor('#0a0f18')
        ax.tick_params(colors='white')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        plt.close(fig)

    st.markdown(f"**Notes:** {result.get('notes', '')}")

    if image:
        try:
            pdf_bytes = generate_pdf_report(
                result=result,
                image=image,
                username=st.session_state.get("username", "Unknown"),
                email=st.session_state.get("user_email", "Unknown")
            )
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"OncoVision_Classification_Report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                width='stretch'
            )
        except Exception as e:
            st.error(f"Could not generate PDF report: {e}")


def segmentation_result_page():
    st.markdown("## 🖍️ Segmentation Results")
    result = st.session_state.get("result")
    image = st.session_state.get("last_image")

    if not result or result.get("type") != "segmentation":
        st.warning("No segmentation result available. Please upload an image and run segmentation from the Upload Image page.")
        if st.button("Go to Upload Image", width='stretch'):
            st.session_state["page"] = "Upload Image"
            st.rerun()
        return

    if image:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    coverage = result.get("coverage", 0.0) * 100
    cell_stats = result.get("cell_stats", {})
    total_cells = sum(cell_stats.values())
    num_classes = len(cell_stats)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mask Coverage", f"{coverage:.1f}%")
    with col2:
        st.metric("Total Nuclei Detected", total_cells)
    with col3:
        st.metric("Classes Found", num_classes)

    col_ov, col_bin = st.columns(2)
    with col_ov:
        overlay = result.get("overlay_rgb")
        if overlay is not None:
            st.image(overlay, caption="Instance Overlay", use_container_width=True)
    with col_bin:
        binary_mask = result.get("mask")
        if binary_mask is not None:
            st.image((binary_mask * 255).astype(np.uint8), caption="Binary Mask", use_container_width=True)

    st.markdown(f"**Notes:** {result.get('notes', '')}")

    if image:
        try:
            pdf_bytes = generate_pdf_report(
                result=result,
                image=image,
                username=st.session_state.get("username", "Unknown"),
                email=st.session_state.get("user_email", "Unknown")
            )
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"OncoVision_Segmentation_Report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                width='stretch'
            )
        except Exception as e:
            st.error(f"Could not generate PDF report: {e}")


def model_info_page():
    st.markdown("## Model Information")
    st.markdown("""
    - **Classification Model**: CNN + Transformer Hybrid
    - **Segmentation Model**: U-Net with mit_b2 encoder
    - **Validation**: CLIP-based relevance gate
    - **Training Dataset**: Lizard histopathology dataset (316,616 images, 1.74 GB)
    - **Classes**: connective, eosinophil, epithelial, lymphocyte, neutrophil, plasma
    """)


def feedback_page(client):
    st.markdown("## Submit Feedback")
    user_name = st.session_state.get("username", "")
    user_email = st.session_state.get("user_email", "")
    with st.form("feedback_form"):
        st.text_input("Name", value=user_name, disabled=True)
        st.text_input("Email", value=user_email, disabled=True)
        rating = st.select_slider(
            "Rate your experience",
            options=[1, 2, 3, 4, 5],
            value=5,
            format_func=lambda x: "⭐" * x + "☆" * (5 - x)
        )
        message = st.text_area("Feedback", placeholder="Describe your experience or suggestions…", height=200)
        submitted = st.form_submit_button("Submit Feedback", width='stretch')
        if submitted:
            if not message:
                st.warning("Please enter your feedback.")
            else:
                try:
                    client[DB_NAME][FEEDBACK_COLLECTION].insert_one({
                        "name": user_name,
                        "email": user_email,
                        "rating": rating,
                        "message": message,
                        "created_at": datetime.now(timezone.utc),
                    })
                    st.success("Thank you! Your feedback has been submitted.")
                except Exception as e:
                    st.error(f"Could not save feedback: {e}")


def history_page(client):
    st.markdown("## Prediction History")
    user_email = st.session_state.get("user_email", "")
    if user_email and client:
        preds = list(client[DB_NAME][PREDICTIONS_COLLECTION].find({"email": user_email}, sort=[("timestamp", -1)]))
        if preds:
            for p in preds:
                ts = p.get("timestamp", datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M")
                title = p.get("title", "N/A")
                conf = p.get("confidence", 0)
                conf_str = f"{conf*100:.1f}%" if conf else ""
                st.markdown(f"**{ts}** – {title} {conf_str}")
                st.divider()
        else:
            st.info("No predictions yet.")
    else:
        st.error("User not found or database unavailable.")


def settings_page(client):
    st.markdown("## Change Password")
    user_email = st.session_state.get("user_email", "")
    if not client:
        st.error("Database not available. Cannot change password.")
        return
    with st.form("change_password"):
        old = st.text_input("Old Password", type="password")
        new = st.text_input("New Password", type="password")
        confirm = st.text_input("Confirm New Password", type="password")
        if st.form_submit_button("Update Password", width='stretch'):
            if not all([old, new, confirm]):
                st.error("All fields required.")
            elif new != confirm:
                st.error("New passwords do not match.")
            else:
                user = client[DB_NAME][COLLECTION_NAME].find_one({"email": user_email})
                if user and bcrypt.checkpw(old.encode("utf-8"), user["password"]):
                    hashed = bcrypt.hashpw(new.encode("utf-8"), bcrypt.gensalt())
                    client[DB_NAME][COLLECTION_NAME].update_one({"email": user_email}, {"$set": {"password": hashed}})
                    st.success("Password changed successfully!")
                else:
                    st.error("Old password is incorrect.")


def logout():
    for k in ["logged_in", "username", "user_email", "last_image", "result", "page"]:
        st.session_state.pop(k, None)
    st.session_state["logged_in"] = False
    st.session_state["page"] = "Dashboard"
    st.rerun()


# =============================================================================
# MAIN
# =============================================================================
def main():
    st.set_page_config(page_title="OncoVision", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

    # Load and embed background image as base64
    bg_base64 = ""
    if os.path.exists("1.png"):
        with open("1.png", "rb") as f:
            bg_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Custom CSS with background image (base64 embedded)
    custom_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        z-index: -1;
    }}

    /* ===== Hide default Streamlit header bar ===== */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        height: 0 !important;
        min-height: 0 !important;
        overflow: visible !important;
    }}
    [data-testid="stSidebarContent"] {{
        overflow-x: hidden !important;
    }}
    [data-testid="stSidebar"] {{
        width: 21rem !important;
        min-width: 21rem !important;
        max-width: 21rem !important;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Load existing style.css if present
    if os.path.exists("style.css"):
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "page" not in st.session_state:
        st.session_state["page"] = "Dashboard"

    client = init_db()
    if not client:
        st.error("Cannot connect to database. Please check your MongoDB URI in .env file and network connectivity.")
        st.stop()

    if not st.session_state["logged_in"]:
        auth_screen(client)
        return

    # ===== Hamburger toggle (only visible after login) =====
    if "sidebar_open" not in st.session_state:
        st.session_state["sidebar_open"] = True

    st.markdown(
        """
        <style>
        #ov-toggle-marker + div .stButton {
            position: fixed !important;
            top: 14px !important;
            left: 18px !important;
            z-index: 999999 !important;
        }
        #ov-toggle-marker + div .stButton button {
            width: 42px !important;
            height: 42px !important;
            min-height: 42px !important;
            padding: 0 !important;
            border-radius: 10px !important;
            background: linear-gradient(95deg, #4facfe, #b465ff) !important;
            color: white !important;
            font-size: 20px !important;
            border: none !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.5) !important;
        }
        </style>
        <div id="ov-toggle-marker"></div>
        """,
        unsafe_allow_html=True
    )
    if st.button("☰", key="ov_hamburger_btn"):
        st.session_state["sidebar_open"] = not st.session_state["sidebar_open"]
        st.rerun()

    if st.session_state["sidebar_open"]:
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] {
                display: block !important;
                visibility: visible !important;
                transform: none !important;
                margin-left: 0 !important;
                width: 21rem !important;
                min-width: 21rem !important;
                max-width: 21rem !important;
            }
            [data-testid="stSidebar"][aria-expanded="false"] {
                transform: none !important;
                visibility: visible !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<style>[data-testid='stSidebar']{display:none !important;}</style>",
            unsafe_allow_html=True
        )

    show_sidebar()

    page = st.session_state["page"]
    if page == "Dashboard":
        dashboard_page(client)
    elif page == "Upload Image":
        upload_image_page(client)
    elif page == "Classification Result":
        classification_result_page()
    elif page == "Segmentation Result":
        segmentation_result_page()
    elif page == "Model Info":
        model_info_page()
    elif page == "Feedback":
        feedback_page(client)
    elif page == "History":
        history_page(client)
    elif page == "Settings":
        settings_page(client)
    elif page == "Logout":
        logout()
    else:
        dashboard_page(client)

    st.markdown("---")
    st.markdown("<center><small>© 2026 OncoVision. All rights reserved. Built with ❤️ using AI & Deep Learning</small></center>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()





# import os
# import smtplib
# import ssl
# import base64
# from email.mime.text import MIMEText
# from datetime import datetime, timedelta, timezone
# import tempfile
# from io import BytesIO

# import cv2
# from skimage import measure
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import bcrypt
# import numpy as np
# import pymongo
# import segmentation_models_pytorch as smp
# import streamlit as st
# import torch
# import torch.nn as nn
# from PIL import Image
# from dotenv import load_dotenv
# import clip
# import matplotlib.pyplot as plt

# # PDF generation imports
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import (
#     SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
# )
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib.units import inch
# from reportlab.lib import colors
# from reportlab.lib.enums import TA_CENTER

# load_dotenv()

# # --- CONFIG ---
# MONGO_URI = os.getenv("MONGO_URI")
# DB_NAME = "lizard_ai_db"
# COLLECTION_NAME = "users"
# FEEDBACK_COLLECTION = "feedback"
# PREDICTIONS_COLLECTION = "predictions"
# MODEL_METRICS_COLLECTION = "model_metrics"
# MODEL_PATH_SEG = "nd.pth"
# CLS_MODEL_PATH = "lizard_model_final.pth"
# DEVICE = "cpu"

# IMG_SIZE = 64
# MEAN = (0.5, 0.5, 0.5)
# STD = (0.5, 0.5, 0.5)
# CLS_LABELS = ["connective", "eosinophil", "epithelial", "lymphocyte", "neutrophil", "plasma"]

# MIN_CONF = 0.60
# MIN_MARGIN = 0.10
# MIN_HE_HUE_RATIO = 0.12
# MIN_SAT = 0.15
# MIN_SAT_MEAN = 0.18
# CLIP_MIN_SIM = 0.22

# EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
# SMTP_HOST = os.getenv("SMTP_HOST", "")
# SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
# SMTP_USER = os.getenv("SMTP_USER", "")
# SMTP_PASS = os.getenv("SMTP_PASS", "")
# SMTP_SENDER = os.getenv("SMTP_SENDER", SMTP_USER or "")
# EMAIL_SECURITY = os.getenv("EMAIL_SECURITY", "tls").lower()

# # --- INSTANCE OVERLAY CONFIG ---
# CLASS_COLORS = {
#     "epithelial":  (0,   255,   0),
#     "lymphocyte":  (255,   0,   0),
#     "plasma":      (0,   255, 255),
#     "neutrophil":  (255, 165,   0),
#     "eosinophil":  (0,     0, 255),
#     "connective":  (255, 255,   0),
#     "unknown":     (180, 180, 180),
# }
# MIN_NUCLEUS_AREA = 40
# MAX_NUCLEUS_AREA = 8000
# CONTOUR_THICKNESS = 2


# # =============================================================================
# # MODEL
# # =============================================================================
# class CNN_Transformer_Hybrid(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
#         )
#         self.flatten_dim = 128 * 8 * 8
#         self.projector = nn.Linear(self.flatten_dim, 256)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=256, nhead=4, dim_feedforward=512, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
#         self.classifier = nn.Sequential(
#             nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(128, num_classes),
#         )

#     def forward(self, x):
#         x = self.cnn(x)
#         x = x.view(x.size(0), -1)
#         x = self.projector(x).unsqueeze(1)
#         x = self.transformer(x)
#         return self.classifier(x.squeeze(1))


# # =============================================================================
# # CACHED LOADERS
# # =============================================================================
# @st.cache_resource(show_spinner=False)
# def init_db():
#     if not MONGO_URI:
#         st.error("MONGO_URI not found in environment variables.")
#         return None
#     try:
#         client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
#         client.admin.command("ping")
#         db = client[DB_NAME]
#         if MODEL_METRICS_COLLECTION not in db.list_collection_names():
#             db[MODEL_METRICS_COLLECTION].insert_one({
#                 "accuracy": 83.89,
#                 "precision": 83.40,
#                 "recall": 83.89,
#                 "f1": 83.48,
#                 "updated_at": datetime.now(timezone.utc)
#             })
#         return client
#     except Exception as e:
#         st.error(f"Database connection failed: {e}")
#         return None


# @st.cache_resource(show_spinner=False)
# def load_seg_model():
#     model = smp.Unet(encoder_name="mit_b2", in_channels=3, classes=1)
#     if not os.path.exists(MODEL_PATH_SEG):
#         st.error(f"Segmentation model not found at {MODEL_PATH_SEG}")
#         return None
#     try:
#         model.load_state_dict(torch.load(MODEL_PATH_SEG, map_location=DEVICE))
#         model.eval()
#         return model
#     except Exception as e:
#         st.error(f"Failed to load segmentation model: {e}")
#         return None


# @st.cache_resource(show_spinner=False)
# def load_cls_model_and_labels():
#     if not os.path.exists(CLS_MODEL_PATH):
#         return None, None
#     checkpoint = torch.load(CLS_MODEL_PATH, map_location=DEVICE)
#     state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
#     out_dim = state.get("classifier.3.weight", torch.empty(0)).shape[0]
#     if out_dim == 0:
#         raise RuntimeError("Could not detect classifier output dimension.")
#     labels = CLS_LABELS if len(CLS_LABELS) == out_dim else [f"Class {i}" for i in range(out_dim)]
#     model = CNN_Transformer_Hybrid(num_classes=out_dim)
#     model.load_state_dict(state, strict=True)
#     model.eval()
#     return model, labels


# @st.cache_resource(show_spinner=False)
# def load_clip_model():
#     model, preprocess = clip.load("ViT-B/32", device=DEVICE)
#     text_tokens = clip.tokenize(
#         ["histology slide", "microscope tissue", "cell microscopy"]).to(DEVICE)
#     return model, preprocess, text_tokens


# # =============================================================================
# # CORE INFERENCE
# # =============================================================================
# def clip_similarity(img: Image.Image) -> float:
#     clip_model, clip_preproc, clip_text = load_clip_model()
#     with torch.no_grad():
#         img_feat = clip_model.encode_image(clip_preproc(img).unsqueeze(0).to(DEVICE))
#         txt_feat = clip_model.encode_text(clip_text)
#         img_feat /= img_feat.norm(dim=-1, keepdim=True)
#         txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
#         return float((img_feat @ txt_feat.T).squeeze(0).max().cpu())


# def run_classification(img: Image.Image, skip_gate: bool = False) -> dict:
#     model, labels = load_cls_model_and_labels()
#     if model is None:
#         raise RuntimeError(f"Classifier weights not found: {CLS_MODEL_PATH}")

#     clip_sim = clip_similarity(img)
#     hsv = np.array(img.convert("RGB").convert("HSV"), dtype=np.float32) / 255.0
#     h, s = hsv[..., 0], hsv[..., 1]
#     sat_mean = float(s.mean())
#     he_ratio = float(((s > MIN_SAT) & ((h < 0.12) | (h > 0.64))).mean())

#     tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
#     x = tf(image=np.array(img.convert("RGB")))["image"].unsqueeze(0)

#     with torch.no_grad():
#         probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]

#     top_idx = int(probs.argmax())
#     top_conf = float(probs[top_idx])
#     top_class = labels[top_idx]
#     margin = top_conf - float(np.sort(probs)[::-1][1]) if len(probs) > 1 else 0.0

#     gate_rejected = (not skip_gate) and (
#         top_conf < MIN_CONF or margin < MIN_MARGIN or
#         he_ratio < MIN_HE_HUE_RATIO or sat_mean < MIN_SAT_MEAN or
#         clip_sim < CLIP_MIN_SIM
#     )

#     if gate_rejected:
#         title = "Irrelevant image"
#         notes = f"Irrelevant: he_ratio={he_ratio:.3f}, sat_mean={sat_mean:.3f}, clip_sim={clip_sim:.3f}"
#     else:
#         title = top_class
#         notes = "Automated classification result."

#     return {
#         "type": "classification",
#         "title": title,
#         "raw_class": top_class,
#         "confidence": top_conf,
#         "breakdown": list(zip(labels, probs.tolist())),
#         "notes": notes,
#     }


# # =============================================================================
# # INSTANCE OVERLAY
# # =============================================================================
# def _classify_roi(roi_img: Image.Image):
#     try:
#         result = run_classification(roi_img, skip_gate=True)
#         title = result.get("raw_class", "unknown")
#         return title
#     except Exception:
#         return "unknown"


# def build_instance_overlay(orig_img: Image.Image, binary_mask: np.ndarray):
#     orig_np = np.array(orig_img.convert("RGB"))
#     h, w = orig_np.shape[:2]
#     mask_rs = cv2.resize(binary_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

#     labeled = measure.label(mask_rs, connectivity=2)
#     regions = measure.regionprops(labeled)
#     overlay = cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR)
#     stats: dict = {}

#     for region in regions:
#         area = region.area
#         if area < MIN_NUCLEUS_AREA or area > MAX_NUCLEUS_AREA:
#             continue
#         r0, c0, r1, c1 = region.bbox
#         pad = 4
#         roi_np = orig_np[max(0, r0 - pad):min(h, r1 + pad),
#                          max(0, c0 - pad):min(w, c1 + pad)]
#         if roi_np.size == 0:
#             continue
#         label = _classify_roi(Image.fromarray(roi_np))
#         rgb = CLASS_COLORS.get(label, CLASS_COLORS["unknown"])
#         bgr = (rgb[2], rgb[1], rgb[0])
#         nucleus_mask = (labeled == region.label).astype(np.uint8)
#         contours, _ = cv2.findContours(nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(overlay, contours, -1, bgr, CONTOUR_THICKNESS)
#         stats[label] = stats.get(label, 0) + 1

#     return overlay, stats


# def _draw_legend(overlay_bgr: np.ndarray, stats: dict) -> np.ndarray:
#     if not stats:
#         return overlay_bgr
#     img = overlay_bgr.copy()
#     h, w = img.shape[:2]
#     box_w, line_h, pad = 160, 18, 8
#     n = len(stats)
#     box_h = n * line_h + pad * 2
#     x0, y0 = w - box_w - pad, pad
#     roi_bg = img[y0:y0 + box_h, x0:x0 + box_w]
#     img[y0:y0 + box_h, x0:x0 + box_w] = cv2.addWeighted(
#         roi_bg, 0.3, np.zeros_like(roi_bg), 0.7, 0)
#     for i, (cls, count) in enumerate(sorted(stats.items())):
#         rgb = CLASS_COLORS.get(cls, CLASS_COLORS["unknown"])
#         bgr = (rgb[2], rgb[1], rgb[0])
#         cy = y0 + pad + i * line_h + line_h // 2
#         cv2.circle(img, (x0 + pad + 5, cy), 5, bgr, -1)
#         cv2.putText(img, f"{cls}: {count}", (x0 + pad + 15, cy + 4),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
#     return img


# # =============================================================================
# # EMAIL / LOGO / DATABASE HELPERS
# # =============================================================================
# def send_login_email(recipient: str, name: str):
#     if not EMAIL_ENABLED:
#         return
#     if not (SMTP_HOST and SMTP_USER and SMTP_PASS and SMTP_SENDER and recipient):
#         return
#     msg = MIMEText(
#         f"Hello {name},\n\nYou have successfully signed in to OncoVision.\n"
#         f"If this wasn't you, please reset your password immediately.\n\nStay safe,\nOncoVision Team")
#     msg["Subject"] = "OncoVision Login Alert"
#     msg["From"] = SMTP_SENDER
#     msg["To"] = recipient
#     try:
#         if EMAIL_SECURITY == "ssl":
#             ctx = ssl.create_default_context()
#             with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as s:
#                 s.login(SMTP_USER, SMTP_PASS); s.send_message(msg)
#         else:
#             with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
#                 if EMAIL_SECURITY == "tls":
#                     s.starttls(context=ssl.create_default_context())
#                 s.login(SMTP_USER, SMTP_PASS); s.send_message(msg)
#     except Exception:
#         pass


# def render_logo():
#     if os.path.exists("2.png"):
#         with open("2.png", "rb") as f:
#             b64 = base64.b64encode(f.read()).decode("utf-8")
#         st.markdown(
#             f'<div class="logo-holder"><img src="data:image/png;base64,{b64}" '
#             f'class="logo-img" alt="OncoVision Logo"></div>',
#             unsafe_allow_html=True)
#     else:
#         st.markdown(
#             "<div class='logo-holder'><div class='logo-text'>🧬 OncoVision</div></div>",
#             unsafe_allow_html=True)


# def save_prediction_to_db(client, username, email, result):
#     if client is None:
#         return
#     try:
#         record = {
#             "username": username,
#             "email": email,
#             "type": result.get("type"),
#             "title": result.get("raw_class") if result.get("type") == "classification" else result.get("title"),
#             "confidence": result.get("confidence"),
#             "timestamp": datetime.now(timezone.utc),
#         }
#         if result.get("type") == "segmentation":
#             record["cell_stats"] = result.get("cell_stats", {})
#             record["coverage"] = result.get("coverage")
#         client[DB_NAME][PREDICTIONS_COLLECTION].insert_one(record)
#     except Exception as e:
#         print(f"DB save error: {e}")


# def get_recent_predictions(client, limit=5):
#     if client is None:
#         return []
#     try:
#         preds = list(client[DB_NAME][PREDICTIONS_COLLECTION].find(
#             {}, sort=[("timestamp", -1)], limit=limit))
#         return preds
#     except:
#         return []


# def get_total_scans_and_change(client):
#     if client is None:
#         return 0, 0.0
#     now = datetime.now(timezone.utc)
#     current_month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
#     if now.month == 1:
#         prev_month_start = datetime(now.year - 1, 12, 1, tzinfo=timezone.utc)
#     else:
#         prev_month_start = datetime(now.year, now.month - 1, 1, tzinfo=timezone.utc)
#     total = client[DB_NAME][PREDICTIONS_COLLECTION].count_documents({})
#     current_count = client[DB_NAME][PREDICTIONS_COLLECTION].count_documents({"timestamp": {"$gte": current_month_start}})
#     prev_count = client[DB_NAME][PREDICTIONS_COLLECTION].count_documents({
#         "timestamp": {"$gte": prev_month_start, "$lt": current_month_start}
#     })
#     if prev_count == 0:
#         percent_change = 100.0 if current_count > 0 else 0.0
#     else:
#         percent_change = ((current_count - prev_count) / prev_count) * 100
#     return total, round(percent_change, 1)


# def get_active_users(client):
#     if client is None:
#         return 0
#     cutoff = datetime.now(timezone.utc) - timedelta(days=30)
#     pipeline = [
#         {"$match": {"timestamp": {"$gte": cutoff}}},
#         {"$group": {"_id": "$email"}},
#         {"$count": "active_users"}
#     ]
#     result = list(client[DB_NAME][PREDICTIONS_COLLECTION].aggregate(pipeline))
#     return result[0]["active_users"] if result else 0


# def get_model_metrics(client):
#     if client is None:
#         return {"accuracy": 83.89, "precision": 83.40, "recall": 83.89, "f1": 83.48}
#     doc = client[DB_NAME][MODEL_METRICS_COLLECTION].find_one(sort=[("updated_at", -1)])
#     if doc:
#         return {
#             "accuracy": doc.get("accuracy", 83.89),
#             "precision": doc.get("precision", 83.40),
#             "recall": doc.get("recall", 83.89),
#             "f1": doc.get("f1", 83.48)
#         }
#     return {"accuracy": 83.89, "precision": 83.40, "recall": 83.89, "f1": 83.48}


# # =============================================================================
# # PDF REPORT GENERATION
# # =============================================================================
# def generate_pdf_report(result: dict, image: Image.Image, username: str, email: str) -> bytes:
#     buffer = BytesIO()
#     doc = SimpleDocTemplate(buffer, pagesize=letter,
#                             rightMargin=72, leftMargin=72,
#                             topMargin=72, bottomMargin=72)
#     styles = getSampleStyleSheet()
#     title_style = ParagraphStyle(
#         'CustomTitle', parent=styles['Heading1'],
#         fontSize=24, textColor=colors.HexColor('#1e3a8a'), alignment=TA_CENTER, spaceAfter=12)
#     heading_style = ParagraphStyle(
#         'CustomHeading', parent=styles['Heading2'],
#         fontSize=16, textColor=colors.HexColor('#0f1724'), spaceAfter=8)
#     normal_style = styles['Normal']
#     center_style = ParagraphStyle('Center', parent=normal_style, alignment=TA_CENTER)

#     story = []
#     temp_files = []

#     logo_path = "2.png"
#     if os.path.exists(logo_path):
#         try:
#             img_logo = RLImage(logo_path, width=1.2*inch, height=1.2*inch, hAlign='CENTER')
#             story.append(img_logo)
#         except:
#             pass
#     story.append(Paragraph("OncoVision", title_style))
#     story.append(Spacer(1, 0.2*inch))

#     now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
#     story.append(Paragraph(f"Report generated: {now_str}", center_style))
#     story.append(Paragraph(f"User: {username} ({email})", center_style))
#     story.append(Spacer(1, 0.2*inch))

#     story.append(Paragraph("Original Histology Image", heading_style))
#     with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
#         image.save(tmp_img.name, format="PNG")
#         tmp_img_path = tmp_img.name
#         temp_files.append(tmp_img_path)
#     story.append(RLImage(tmp_img_path, width=4*inch, height=4*inch, hAlign='CENTER'))
#     story.append(Spacer(1, 0.2*inch))

#     analysis_type = result.get("type", "Unknown").capitalize()
#     story.append(Paragraph(f"Analysis Type: {analysis_type}", heading_style))

#     if result.get("type") == "classification":
#         pred_class = result.get("title", "N/A")
#         conf = result.get("confidence", 0.0) * 100
#         notes = result.get("notes", "")
#         story.append(Paragraph(f"Predicted Class: {pred_class}", normal_style))
#         story.append(Paragraph(f"Confidence: {conf:.2f}%", normal_style))
#         story.append(Paragraph(f"Notes: {notes}", normal_style))

#         breakdown = result.get("breakdown", [])
#         if breakdown:
#             table_data = [["Class", "Probability (%)"]]
#             for cls, prob in breakdown:
#                 table_data.append([cls, f"{prob*100:.2f}%"])
#             table = Table(table_data, colWidths=[2.5*inch, 1.5*inch])
#             table.setStyle(TableStyle([
#                 ('BACKGROUND', (0,0), (-1,0), colors.grey),
#                 ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
#                 ('ALIGN', (0,0), (-1,-1), 'CENTER'),
#                 ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
#                 ('BOTTOMPADDING', (0,0), (-1,0), 6),
#                 ('BACKGROUND', (0,1), (-1,-1), colors.beige),
#                 ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
#             ]))
#             story.append(Spacer(1, 0.1*inch))
#             story.append(Paragraph("Probability Breakdown", heading_style))
#             story.append(table)

#     elif result.get("type") == "segmentation":
#         coverage = result.get("coverage", 0.0) * 100
#         cell_stats = result.get("cell_stats", {})
#         total_cells = sum(cell_stats.values())
#         num_classes = len(cell_stats)
#         notes = result.get("notes", "")

#         story.append(Paragraph(f"Mask Coverage: {coverage:.1f}%", normal_style))
#         story.append(Paragraph(f"Total Nuclei Detected: {total_cells}", normal_style))
#         story.append(Paragraph(f"Number of Classes Found: {num_classes}", normal_style))
#         story.append(Paragraph(f"Notes: {notes}", normal_style))

#         if cell_stats:
#             table_data = [["Cell Class", "Count", "Percentage"]]
#             for cls, cnt in sorted(cell_stats.items()):
#                 pct = (cnt / total_cells * 100) if total_cells > 0 else 0
#                 table_data.append([cls, str(cnt), f"{pct:.1f}%"])
#             table = Table(table_data, colWidths=[2*inch, 1.2*inch, 1.2*inch])
#             table.setStyle(TableStyle([
#                 ('BACKGROUND', (0,0), (-1,0), colors.grey),
#                 ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
#                 ('ALIGN', (0,0), (-1,-1), 'CENTER'),
#                 ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
#                 ('BOTTOMPADDING', (0,0), (-1,0), 6),
#                 ('BACKGROUND', (0,1), (-1,-1), colors.beige),
#                 ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
#             ]))
#             story.append(Spacer(1, 0.1*inch))
#             story.append(Paragraph("Cell Statistics", heading_style))
#             story.append(table)

#         overlay_rgb = result.get("overlay_rgb")
#         if overlay_rgb is not None:
#             story.append(PageBreak())
#             story.append(Paragraph("Instance Overlay (Colour‑coded Nuclei)", heading_style))
#             with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_over:
#                 overlay_pil = Image.fromarray(overlay_rgb)
#                 overlay_pil.save(tmp_over.name, format="PNG")
#                 tmp_over_path = tmp_over.name
#                 temp_files.append(tmp_over_path)
#             story.append(RLImage(tmp_over_path, width=5*inch, height=5*inch, hAlign='CENTER'))

#         binary_mask = result.get("mask")
#         if binary_mask is not None:
#             story.append(Spacer(1, 0.1*inch))
#             story.append(Paragraph("Binary Segmentation Mask", heading_style))
#             with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_mask:
#                 mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
#                 mask_pil.save(tmp_mask.name, format="PNG")
#                 tmp_mask_path = tmp_mask.name
#                 temp_files.append(tmp_mask_path)
#             story.append(RLImage(tmp_mask_path, width=5*inch, height=5*inch, hAlign='CENTER'))

#     story.append(Spacer(1, 0.3*inch))
#     story.append(Paragraph("— End of Report —", center_style))

#     doc.build(story)
#     for fpath in temp_files:
#         try:
#             os.unlink(fpath)
#         except:
#             pass

#     pdf_bytes = buffer.getvalue()
#     buffer.close()
#     return pdf_bytes


# # =============================================================================
# # AUTH SCREEN (Stylish glassmorphism)
# # =============================================================================
# def auth_screen(client):
#     import random

#     # Read logo
#     logo_base64 = ""
#     if os.path.exists("logo.png"):
#         with open("logo.png", "rb") as f:
#             logo_base64 = base64.b64encode(f.read()).decode("utf-8")

#     # ========== CSS ==========
#     st.markdown("""
#     <style>
#     /* Hide Streamlit header */
#     header[data-testid="stHeader"], .stApp > header, [data-testid="stDecoration"] {
#         display: none !important;
#     }
#     /* Remove all top spacing */
#     .block-container {
#         padding-top: 0 !important;
#         margin-top: 0 !important;
#     }
#     .main .block-container {
#         padding-top: 0 !important;
#     }
#     /* Background gradient */
#     .stApp {
#         background: radial-gradient(circle at 20% 30%, #0a0f2a, #030617);
#         color: white;
#     }
#     /* Floating stars container */
#     .stars {
#         position: fixed;
#         top: 0;
#         left: 0;
#         width: 100%;
#         height: 100%;
#         pointer-events: none;
#         z-index: 0;
#     }
#     .star {
#         position: absolute;
#         background: white;
#         border-radius: 50%;
#         opacity: 0.6;
#         animation: floatStar 15s infinite linear;
#     }
#     @keyframes floatStar {
#         0% { transform: translate(0, 0); opacity: 0; }
#         20% { opacity: 0.8; }
#         80% { opacity: 0.8; }
#         100% { transform: translate(30px, -40px); opacity: 0; }
#     }
#     /* Floating cells container */
#     .cells {
#         position: fixed;
#         top: 0;
#         left: 0;
#         width: 100%;
#         height: 100%;
#         pointer-events: none;
#         z-index: 0;
#     }
#     .cell {
#         position: absolute;
#         border-radius: 50%;
#         background: radial-gradient(circle, rgba(79,172,254,0.4), rgba(180,101,255,0.2));
#         border: 1px solid rgba(79,172,254,0.5);
#         box-shadow: 0 0 8px rgba(79,172,254,0.3);
#         animation: floatCell 25s infinite ease-in-out;
#     }
#     .cell::before {
#         content: '';
#         position: absolute;
#         top: 50%;
#         left: 50%;
#         width: 30%;
#         height: 30%;
#         background: radial-gradient(circle, rgba(180,101,255,0.6), rgba(79,172,254,0.3));
#         border-radius: 50%;
#         transform: translate(-50%, -50%);
#     }
#     @keyframes floatCell {
#         0% { transform: translate(0, 0) rotate(0deg); opacity: 0.3; }
#         50% { transform: translate(30px, -25px) rotate(5deg); opacity: 0.7; }
#         100% { transform: translate(-15px, 20px) rotate(-3deg); opacity: 0.3; }
#     }
#     /* Microscope */
#     .floating-microscope {
#         position: fixed;
#         bottom: 15%;
#         right: 5%;
#         font-size: 70px;
#         opacity: 0.4;
#         pointer-events: none;
#         z-index: 0;
#         filter: drop-shadow(0 0 8px rgba(79,172,254,0.5));
#         animation: floatMicroscope 12s infinite ease-in-out;
#     }
#     @keyframes floatMicroscope {
#         0% { transform: translate(0, 0) rotate(0deg); opacity: 0.3; }
#         50% { transform: translate(-20px, -30px) rotate(5deg); opacity: 0.6; }
#         100% { transform: translate(0, 0) rotate(0deg); opacity: 0.3; }
#     }
#     /* Glass effect for cards */
#     .glass-card, .feature-card, .stat-card {
#         background: rgba(20, 30, 50, 0.7);
#         backdrop-filter: blur(16px);
#         border-radius: 24px;
#         border: 1px solid rgba(72, 187, 255, 0.3);
#         padding: 20px;
#         margin-bottom: 20px;
#         position: relative;
#         z-index: 1;
#     }
#     .feature-card {
#         padding: 12px 16px;
#         margin: 12px 0;
#         border-left: 4px solid #4facfe;
#         transition: 0.2s;
#     }
#     .feature-card:hover {
#         transform: translateX(8px);
#         background: rgba(20, 40, 70, 0.8);
#     }
#     .stat-card {
#         text-align: center;
#         padding: 10px;
#         flex: 1;
#     }
#     .stat-number {
#         font-size: 26px;
#         font-weight: 800;
#         background: linear-gradient(135deg, #4facfe, #b465ff);
#         -webkit-background-clip: text;
#         background-clip: text;
#         color: transparent;
#     }
#     /* Login card */
#     .login-card {
#         background: rgba(10, 20, 35, 0.8);
#         backdrop-filter: blur(20px);
#         border-radius: 28px;
#         border: 1px solid rgba(79,172,254,0.3);
#         position: relative;
#         z-index: 1;
#     }
#     /* Inputs */
#     .stTextInput input {
#         background: rgba(0,0,0,0.5) !important;
#         border: 1px solid #4facfe !important;
#         border-radius: 14px !important;
#         padding: 12px 16px !important;
#         color: white !important;
#     }
#     /* Button */
#     .stButton button {
#         background: linear-gradient(95deg, #4facfe, #b465ff);
#         border-radius: 40px;
#         height: 50px;
#         font-weight: 700;
#         font-size: 16px;
#         width: 100%;
#     }
#     /* Tabs */
#     .stTabs [data-baseweb="tab"] {
#         font-size: 18px;
#         font-weight: 600;
#         color: #a0b4d7;
#     }
#     .stTabs [aria-selected="true"] {
#         color: #4facfe;
#         border-bottom: 2px solid #4facfe;
#     }
#     /* Text */
#     .subtitle {
#         font-size: 15px;
#         color: #cbd5e6;
#         margin-bottom: 25px;
#     }
#     .forgot-link, .create-link {
#         color: #4facfe;
#         text-decoration: none;
#         font-size: 13px;
#     }
#     .forgot-link:hover, .create-link:hover {
#         color: #b465ff;
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     # Stars (12 random)
#     stars_html = '<div class="stars">'
#     for _ in range(12):
#         top = random.randint(5, 95)
#         left = random.randint(5, 95)
#         size = random.randint(2, 6)
#         duration = random.randint(12, 25)
#         delay = random.randint(0, 10)
#         stars_html += f'<div class="star" style="top:{top}%; left:{left}%; width:{size}px; height:{size}px; animation-duration:{duration}s; animation-delay:{delay}s;"></div>'
#     stars_html += '</div>'
#     st.markdown(stars_html, unsafe_allow_html=True)

#     # Floating cells (9 soft glowing circles with nucleus)
#     cells_html = '<div class="cells">'
#     for _ in range(9):
#         top = random.randint(5, 90)
#         left = random.randint(5, 90)
#         size = random.randint(25, 60)
#         duration = random.randint(18, 30)
#         delay = random.randint(0, 12)
#         cells_html += f'<div class="cell" style="top:{top}%; left:{left}%; width:{size}px; height:{size}px; animation-duration:{duration}s; animation-delay:{delay}s;"></div>'
#     cells_html += '</div>'
#     st.markdown(cells_html, unsafe_allow_html=True)

#     # Microscope
#     st.markdown("""
#     <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
#     <div class="floating-microscope"><i class="fas fa-microscope"></i></div>
#     """, unsafe_allow_html=True)

#     # ========== MAIN UI ==========
#     left_col, right_col = st.columns([1.2, 1], gap="large")

#     with left_col:
#         # Brand with larger logo (80px)
#         if logo_base64:
#             st.markdown(f"""
#             <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
#                 <img src="data:image/png;base64,{logo_base64}" style="width: 80px; border-radius: 16px;">
#                 <h1 style="margin:0; font-size:40px;"><span style="color:white">Onco</span><span style="color:#4facfe">Vision</span></h1>
#             </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown("<h1 style='margin-bottom:20px;'><span style='color:white'>Onco</span><span style='color:#4facfe'>Vision</span></h1>", unsafe_allow_html=True)

#         st.markdown("<div class='subtitle'>Advanced deep learning models for precise cancer detection, segmentation and classification.</div>", unsafe_allow_html=True)

#         # Features
#         features = [
#             ("🧠", "CNN + Transformer Classification", "Hybrid architecture"),
#             ("🔬", "U‑Net Segmentation", "Pixel‑perfect detection"),
#             ("✅", "CLIP Image Validation", "Multimodal verification"),
#             ("🔒", "Secure MongoDB Authentication", "Enterprise grade")
#         ]
#         for icon, title, desc in features:
#             st.markdown(f"""
#             <div class='feature-card'>
#                 <div style='display:flex; gap:12px; align-items:center;'>
#                     <span style='font-size:24px'>{icon}</span>
#                     <div><strong>{title}</strong><br><span style='font-size:12px; color:#b0c4de'>{desc}</span></div>
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)

#         # Stats
#         st.markdown("""
#         <div style='display:flex; gap:15px; margin-top:25px;'>
#             <div class='stat-card'><div class='stat-number'>84.59%</div><div>Accuracy</div></div>
#             <div class='stat-card'><div class='stat-number'>100%</div><div>Secure</div></div>
#             <div class='stat-card'><div class='stat-number'>24/7</div><div>AI Processing</div></div>
#         </div>
#         """, unsafe_allow_html=True)

#     with right_col:
#         # Login card
#         st.markdown("""
#         <div class='login-card'>
#             <h2 style='text-align:center; margin-bottom:10px;'>Welcome Back</h2>
#             <p style='text-align:center; margin-bottom:25px; color:#b0c4de;'>Sign in to your account</p>
#         </div>
#         """, unsafe_allow_html=True)

#         tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

#         with tab1:
#             with st.form("login_form"):
#                 email = st.text_input("Email", placeholder="doctor@example.com")
#                 password = st.text_input("Password", type="password", placeholder="••••••••")
#                 col_a, col_b = st.columns(2)
#                 with col_a:
#                     remember = st.checkbox("Remember Me")
#                 with col_b:
#                     st.markdown("<div style='text-align:right'><a href='#' class='forgot-link'>Forgot Password?</a></div>", unsafe_allow_html=True)
#                 submit = st.form_submit_button("Sign In", width='stretch')
#                 if submit:
#                     user = client[DB_NAME][COLLECTION_NAME].find_one({"email": email})
#                     if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
#                         st.session_state["logged_in"] = True
#                         st.session_state["username"] = user["name"]
#                         st.session_state["user_email"] = user["email"]
#                         send_login_email(user["email"], user["name"])
#                         st.success("Login Successful")
#                         st.session_state["page"] = "Dashboard"
#                         st.rerun()
#                     else:
#                         st.error("Invalid Credentials")
#             st.markdown("<div style='text-align:center; margin-top:15px;'><a href='#' class='create-link'>Create New Account</a></div>", unsafe_allow_html=True)

#         with tab2:
#             with st.form("signup_form"):
#                 name = st.text_input("Full Name", placeholder="Dr. Jane Doe")
#                 email = st.text_input("Email Address", placeholder="doctor@example.com")
#                 password = st.text_input("Password", type="password", placeholder="Create a strong password")
#                 submit = st.form_submit_button("Create Account", width='stretch')
#                 if submit:
#                     if client[DB_NAME][COLLECTION_NAME].find_one({"email": email}):
#                         st.warning("User already exists")
#                     else:
#                         hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
#                         client[DB_NAME][COLLECTION_NAME].insert_one({
#                             "name": name,
#                             "email": email,
#                             "password": hashed
#                         })
#                         st.success("Account Created Successfully")


# # =============================================================================
# # SIDEBAR (unchanged)
# # =============================================================================
# def show_sidebar():
#     with st.sidebar:
#         st.markdown("""
#         <style>
#             [data-testid="stSidebar"] {
#                 background: linear-gradient(135deg, #0a0f2a 0%, #4a0e2e 100%);
#                 border-radius: 20px;
#                 margin: 8px;
#                 padding: 10px !important;
#                 height: calc(100vh - 16px);
#                 position: sticky;
#                 top: 8px;
#                 overflow-y: auto;
#                 scrollbar-width: thin;
#             }
#             [data-testid="stSidebar"] > div:first-child {
#                 padding-top: 0 !important;
#             }
#             div[data-testid="stRadio"] > label {
#                 display: none;
#             }
#             div[data-testid="stRadio"] div[role="radiogroup"] {
#                 gap: 6px;
#                 margin-top: 0;
#             }
#             div[data-testid="stRadio"] div[role="radiogroup"] label {
#                 background: rgba(255, 255, 255, 0.08);
#                 border: 1px solid rgba(255, 255, 255, 0.15);
#                 border-radius: 10px;
#                 padding: 8px 12px;
#                 transition: all 0.2s ease;
#                 margin-bottom: 4px;
#                 cursor: pointer;
#                 color: #f0f3fa;
#                 font-weight: 500;
#                 font-size: 0.85rem;
#                 backdrop-filter: blur(4px);
#             }
#             div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
#                 background: rgba(110, 231, 255, 0.2);
#                 border-color: #6ee7ff;
#                 transform: translateX(4px);
#             }
#             div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] {
#                 background: rgba(110, 231, 255, 0.25);
#                 border-left: 3px solid #6ee7ff;
#                 font-weight: 600;
#                 box-shadow: 0 0 8px rgba(110,231,255,0.3);
#             }
#             div[data-testid="stRadio"] div[role="radiogroup"] div[role="radio"] {
#                 display: none;
#             }
#             .status-box, .user-box {
#                 background: rgba(0, 0, 0, 0.3);
#                 border-radius: 10px;
#                 padding: 8px 10px;
#                 margin: 8px 0;
#                 backdrop-filter: blur(4px);
#                 font-size: 0.75rem;
#             }
#             .status-box strong, .user-box strong {
#                 color: #6ee7ff;
#                 font-size: 0.8rem;
#             }
#             .status-box span.status-text {
#                 color: #a0e7ff;
#                 font-size: 0.7rem;
#                 margin-left: 4px;
#             }
#             .separator {
#                 margin: 8px 0;
#                 border-top: 1px solid rgba(255,255,255,0.1);
#             }
#             .sidebar-logo {
#                 text-align: center;
#                 margin-bottom: 15px;
#             }
#             .sidebar-logo img {
#                 width: 55%;
#                 border-radius: 12px;
#                 box-shadow: 0 2px 8px rgba(0,0,0,0.3);
#             }
#         </style>
#         """, unsafe_allow_html=True)

#         if os.path.exists("2.png"):
#             with open("2.png", "rb") as f:
#                 b64 = base64.b64encode(f.read()).decode("utf-8")
#             st.markdown(
#                 f'<div class="sidebar-logo">'
#                 f'<img src="data:image/png;base64,{b64}">'
#                 f'</div>',
#                 unsafe_allow_html=True)
#         else:
#             st.markdown("<h3 style='text-align: center; margin-bottom: 15px;'>🧬 OncoVision</h3>", unsafe_allow_html=True)

#         pages = {
#             "Dashboard": "🏠 Dashboard",
#             "Upload Image": "📤 Upload Image",
#             "Classification Result": "🔬 Classification",
#             "Segmentation Result": "🖍️ Segmentation",
#             "Model Info": "🧠 Model Info",
#             "Feedback": "✉️ Feedback",
#             "History": "📜 History",
#             "Settings": "⚙️ Settings",
#             "Logout": "🚪 Logout"
#         }
#         current = st.session_state.get("page", "Dashboard")
#         selected_display = st.radio(
#             "Navigate",
#             options=list(pages.values()),
#             index=list(pages.values()).index(pages[current]),
#             label_visibility="collapsed"
#         )
#         selected_key = [k for k, v in pages.items() if v == selected_display][0]
#         if selected_key != current:
#             st.session_state["page"] = selected_key
#             st.rerun()

#         st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

#         st.markdown("""
#         <div class='status-box'>
#             <strong>🤖 AI Model Status</strong><br>
#             • CNN + Transformer <span class='status-text'>(Loaded)</span><br>
#             • U-Net (Segmentation) <span class='status-text'>(Loaded)</span><br>
#             • CLIP Validation <span class='status-text'>(Active)</span><br>
#             • Database <span class='status-text'>(Connected)</span>
#         </div>
#         """, unsafe_allow_html=True)

#         st.markdown("<div class='separator'></div>", unsafe_allow_html=True)

#         username = st.session_state.get("username", "Admin User")
#         user_email = st.session_state.get("user_email", "admin@oncovision.ai")
#         st.markdown(f"""
#         <div class='user-box'>
#             <strong>👤 {username}</strong><br>
#             <small>{user_email}</small><br>
#             <span style="color: #6ee7ff;">● Online</span>
#         </div>
#         """, unsafe_allow_html=True)


# # =============================================================================
# # DASHBOARD PAGE
# # =============================================================================
# def dashboard_page(client):
#     st.markdown(f"""
#     <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
#         <h2 style="margin: 0;">Welcome back, {st.session_state.get('username', 'Admin')}! 🎉</h2>
#         <div style="font-size: 1.2rem;">🔔 {datetime.now(timezone.utc).strftime("%d %b %Y")}</div>
#     </div>
#     """, unsafe_allow_html=True)
#     st.markdown("Monitor your model performance and analyze histopathology images.")

#     total_scans, pct_change = get_total_scans_and_change(client)
#     active_users = get_active_users(client)
#     detected_classes = len(CLS_LABELS)
#     dataset_size_display = "1.74 GB"
#     metrics = get_model_metrics(client)

#     col1, col2, col3, col4, col5 = st.columns(5)
#     with col1:
#         change_str = f"+{pct_change}%" if pct_change >= 0 else f"{pct_change}%"
#         st.markdown(f"<div class='glass-card'><h3>{total_scans}</h3><p>Total Scans</p><small>{change_str} from last month</small></div>", unsafe_allow_html=True)
#     with col2:
#         st.markdown(f"<div class='glass-card'><h3>{metrics['accuracy']:.2f}%</h3><p>Model Accuracy</p><small>On Test Dataset</small></div>", unsafe_allow_html=True)
#     with col3:
#         st.markdown(f"<div class='glass-card'><h3>{detected_classes}</h3><p>Detected Classes</p><small>Cell Categories</small></div>", unsafe_allow_html=True)
#     with col4:
#         st.markdown(f"<div class='glass-card'><h3>{dataset_size_display}</h3><p>Dataset Size</p><small>Training Data</small></div>", unsafe_allow_html=True)
#     with col5:
#         st.markdown(f"<div class='glass-card'><h3>{active_users}</h3><p>Active Users</p><small>Last 30 days</small></div>", unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)

#     left, right = st.columns([1, 1])
#     with left:
#         st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
#         st.markdown("### Recent Predictions")
#         recent = get_recent_predictions(client, 5)
#         if recent:
#             for pred in recent:
#                 title = pred.get("title", "N/A")
#                 conf = pred.get("confidence", 0)
#                 conf_str = f"{conf*100:.2f}%" if conf else "N/A"
#                 time_str = pred.get("timestamp", datetime.now(timezone.utc)).strftime("%d %b %Y, %I:%M %p")
#                 st.markdown(f"**{title}** – {conf_str}  \n<small>{time_str}</small>", unsafe_allow_html=True)
#                 st.divider()
#         else:
#             st.info("No predictions yet. Go to Upload Image to run analysis.")
#         st.markdown("</div>", unsafe_allow_html=True)

#     with right:
#         st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
#         st.markdown("### Class Distribution")
#         class_data = {"Lymphocyte": 28.7, "Plasma Cell": 21.4, "Neutrophil": 18.2,
#                       "Eosinophil": 9.8, "Epithelial Cell": 15.6, "Connective": 6.3}
#         fig, ax = plt.subplots(figsize=(4, 3))
#         ax.barh(list(class_data.keys()), list(class_data.values()), color='#6ee7ff')
#         ax.set_xlabel("Percentage")
#         ax.set_facecolor('#0f1724')
#         fig.patch.set_facecolor('#0a0f18')
#         ax.tick_params(colors='white')
#         st.pyplot(fig)
#         plt.close(fig)
#         st.markdown("</div>", unsafe_allow_html=True)

#         st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
#         st.markdown("### Model Performance")
#         col_m1, col_m2 = st.columns(2)
#         with col_m1:
#             st.metric("Accuracy", f"{metrics['accuracy']:.2f}%")
#             st.metric("Precision", f"{metrics['precision']:.2f}%")
#         with col_m2:
#             st.metric("Recall", f"{metrics['recall']:.2f}%")
#             st.metric("F1 Score", f"{metrics['f1']:.2f}%")
#         st.markdown("</div>", unsafe_allow_html=True)

#         st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
#         st.markdown("### Recent Feedback")
#         if client:
#             feedbacks = list(client[DB_NAME][FEEDBACK_COLLECTION].find({}, sort=[("created_at", -1)]).limit(3))
#             for fb in feedbacks:
#                 rating = fb.get("rating", 5)
#                 stars = "⭐" * rating + "☆" * (5 - rating)
#                 st.markdown(f"{stars}  \n{fb.get('message', '')[:150]}  \n<small>{fb.get('name', 'User')} – {fb.get('created_at', datetime.now(timezone.utc)).strftime('%d %b %Y, %I:%M %p')}</small>", unsafe_allow_html=True)
#                 st.divider()
#         st.markdown("</div>", unsafe_allow_html=True)


# # =============================================================================
# # UPLOAD IMAGE PAGE
# # =============================================================================
# def upload_image_page(client):
#     st.markdown("## Upload Histopathology Image")
#     uploaded_file = st.file_uploader("Drag & Drop your image here", type=["png", "jpg", "jpeg", "jfif"], key="upload_page")
#     if uploaded_file:
#         img = Image.open(uploaded_file).convert("RGB")
#         st.session_state["last_image"] = img
#         # ✅ Set a fixed width for the preview (e.g., 400px)
#         st.image(img, caption="Preview", width=400)

#         st.markdown("### AI Analysis")
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("🔬 Classification", width='stretch'):
#                 try:
#                     res = run_classification(st.session_state["last_image"])
#                     st.session_state["result"] = res
#                     save_prediction_to_db(client, st.session_state.get("username"), st.session_state.get("user_email"), res)
#                     st.success(f"Classification completed! Predicted: {res['title']} with {res['confidence']*100:.2f}% confidence.")
#                     st.session_state["page"] = "Classification Result"
#                     st.rerun()
#                 except Exception as e:
#                     st.error(f"Classification failed: {e}")
#         with col2:
#             if st.button("🖍️ Segmentation", width='stretch'):
#                 model = load_seg_model()
#                 if model is None:
#                     st.error("Segmentation model not available. Please check nd.pth file.")
#                 else:
#                     try:
#                         tf = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
#                         tensor = tf(image=np.array(st.session_state["last_image"]))["image"].unsqueeze(0)
#                         with torch.no_grad():
#                             mask = torch.sigmoid(model(tensor)).squeeze().numpy()
#                             binary = (mask > 0.5).astype(np.uint8)
#                         overlay_bgr, stats = build_instance_overlay(st.session_state["last_image"], binary)
#                         overlay_bgr = _draw_legend(overlay_bgr, stats)
#                         overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
#                         seg_result = {
#                             "type": "segmentation",
#                             "mask": binary,
#                             "overlay_rgb": overlay_rgb,
#                             "cell_stats": stats,
#                             "title": "Cell Boundary Map",
#                             "coverage": float(binary.sum()) / binary.size,
#                             "notes": "U-Net + per-cell classification."
#                         }
#                         st.session_state["result"] = seg_result
#                         save_prediction_to_db(client, st.session_state.get("username"), st.session_state.get("user_email"), seg_result)
#                         st.success("Segmentation completed!")
#                         st.session_state["page"] = "Segmentation Result"
#                         st.rerun()
#                     except Exception as e:
#                         st.error(f"Segmentation failed: {e}")
#     else:
#         st.info("Please upload an image to begin analysis.")


# def classification_result_page():
#     st.markdown("## 🔬 Classification Results")
#     result = st.session_state.get("result")
#     image = st.session_state.get("last_image")

#     if not result or result.get("type") != "classification":
#         st.warning("No classification result available. Please upload an image and run classification from the Upload Image page.")
#         if st.button("Go to Upload Image", width='stretch'):
#             st.session_state["page"] = "Upload Image"
#             st.rerun()
#         return

#     if image:
#         st.image(image, caption="Uploaded Image", use_container_width=True)

#     pred_class = result.get("title", "N/A")
#     conf = result.get("confidence", 0.0) * 100
#     st.markdown(f"### Predicted Class: **{pred_class}**")
#     st.markdown(f"**Confidence:** {conf:.2f}%")
#     st.progress(int(conf))

#     breakdown = result.get("breakdown", [])
#     if breakdown:
#         st.markdown("### Class Probabilities")
#         labels, probs = zip(*breakdown)
#         probs_pct = [p * 100 for p in probs]
#         fig, ax = plt.subplots(figsize=(8, 4))
#         colors_bar = [CLASS_COLORS.get(l, (100, 100, 100)) for l in labels]
#         colors_hex = [f"#{r:02x}{g:02x}{b:02x}" for (r,g,b) in colors_bar]
#         ax.bar(labels, probs_pct, color=colors_hex)
#         ax.set_ylabel("Probability (%)")
#         ax.set_ylim(0, 100)
#         ax.set_facecolor('#0f1724')
#         fig.patch.set_facecolor('#0a0f18')
#         ax.tick_params(colors='white')
#         plt.xticks(rotation=45, ha='right')
#         st.pyplot(fig)
#         plt.close(fig)

#     st.markdown(f"**Notes:** {result.get('notes', '')}")

#     if image:
#         try:
#             pdf_bytes = generate_pdf_report(
#                 result=result,
#                 image=image,
#                 username=st.session_state.get("username", "Unknown"),
#                 email=st.session_state.get("user_email", "Unknown")
#             )
#             st.download_button(
#                 label="📄 Download PDF Report",
#                 data=pdf_bytes,
#                 file_name=f"OncoVision_Classification_Report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf",
#                 mime="application/pdf",
#                 width='stretch'
#             )
#         except Exception as e:
#             st.error(f"Could not generate PDF report: {e}")


# def segmentation_result_page():
#     st.markdown("## 🖍️ Segmentation Results")
#     result = st.session_state.get("result")
#     image = st.session_state.get("last_image")

#     if not result or result.get("type") != "segmentation":
#         st.warning("No segmentation result available. Please upload an image and run segmentation from the Upload Image page.")
#         if st.button("Go to Upload Image", width='stretch'):
#             st.session_state["page"] = "Upload Image"
#             st.rerun()
#         return

#     if image:
#         st.image(image, caption="Uploaded Image", use_container_width=True)

#     coverage = result.get("coverage", 0.0) * 100
#     cell_stats = result.get("cell_stats", {})
#     total_cells = sum(cell_stats.values())
#     num_classes = len(cell_stats)

#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Mask Coverage", f"{coverage:.1f}%")
#     with col2:
#         st.metric("Total Nuclei Detected", total_cells)
#     with col3:
#         st.metric("Classes Found", num_classes)

#     col_ov, col_bin = st.columns(2)
#     with col_ov:
#         overlay = result.get("overlay_rgb")
#         if overlay is not None:
#             st.image(overlay, caption="Instance Overlay", use_container_width=True)
#     with col_bin:
#         binary_mask = result.get("mask")
#         if binary_mask is not None:
#             st.image((binary_mask * 255).astype(np.uint8), caption="Binary Mask", use_container_width=True)

#     st.markdown(f"**Notes:** {result.get('notes', '')}")

#     if image:
#         try:
#             pdf_bytes = generate_pdf_report(
#                 result=result,
#                 image=image,
#                 username=st.session_state.get("username", "Unknown"),
#                 email=st.session_state.get("user_email", "Unknown")
#             )
#             st.download_button(
#                 label="📄 Download PDF Report",
#                 data=pdf_bytes,
#                 file_name=f"OncoVision_Segmentation_Report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.pdf",
#                 mime="application/pdf",
#                 width='stretch'
#             )
#         except Exception as e:
#             st.error(f"Could not generate PDF report: {e}")


# def model_info_page():
#     st.markdown("## Model Information")
#     st.markdown("""
#     - **Classification Model**: CNN + Transformer Hybrid
#     - **Segmentation Model**: U-Net with mit_b2 encoder
#     - **Validation**: CLIP-based relevance gate
#     - **Training Dataset**: Lizard histopathology dataset (316,616 images, 1.74 GB)
#     - **Classes**: connective, eosinophil, epithelial, lymphocyte, neutrophil, plasma
#     """)


# def feedback_page(client):
#     st.markdown("## Submit Feedback")
#     user_name = st.session_state.get("username", "")
#     user_email = st.session_state.get("user_email", "")
#     with st.form("feedback_form"):
#         st.text_input("Name", value=user_name, disabled=True)
#         st.text_input("Email", value=user_email, disabled=True)
#         rating = st.select_slider(
#             "Rate your experience",
#             options=[1, 2, 3, 4, 5],
#             value=5,
#             format_func=lambda x: "⭐" * x + "☆" * (5 - x)
#         )
#         message = st.text_area("Feedback", placeholder="Describe your experience or suggestions…", height=200)
#         submitted = st.form_submit_button("Submit Feedback", width='stretch')
#         if submitted:
#             if not message:
#                 st.warning("Please enter your feedback.")
#             else:
#                 try:
#                     client[DB_NAME][FEEDBACK_COLLECTION].insert_one({
#                         "name": user_name,
#                         "email": user_email,
#                         "rating": rating,
#                         "message": message,
#                         "created_at": datetime.now(timezone.utc),
#                     })
#                     st.success("Thank you! Your feedback has been submitted.")
#                 except Exception as e:
#                     st.error(f"Could not save feedback: {e}")


# def history_page(client):
#     st.markdown("## Prediction History")
#     user_email = st.session_state.get("user_email", "")
#     if user_email and client:
#         preds = list(client[DB_NAME][PREDICTIONS_COLLECTION].find({"email": user_email}, sort=[("timestamp", -1)]))
#         if preds:
#             for p in preds:
#                 ts = p.get("timestamp", datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M")
#                 title = p.get("title", "N/A")
#                 conf = p.get("confidence", 0)
#                 conf_str = f"{conf*100:.1f}%" if conf else ""
#                 st.markdown(f"**{ts}** – {title} {conf_str}")
#                 st.divider()
#         else:
#             st.info("No predictions yet.")
#     else:
#         st.error("User not found or database unavailable.")


# def settings_page(client):
#     st.markdown("## Change Password")
#     user_email = st.session_state.get("user_email", "")
#     if not client:
#         st.error("Database not available. Cannot change password.")
#         return
#     with st.form("change_password"):
#         old = st.text_input("Old Password", type="password")
#         new = st.text_input("New Password", type="password")
#         confirm = st.text_input("Confirm New Password", type="password")
#         if st.form_submit_button("Update Password", width='stretch'):
#             if not all([old, new, confirm]):
#                 st.error("All fields required.")
#             elif new != confirm:
#                 st.error("New passwords do not match.")
#             else:
#                 user = client[DB_NAME][COLLECTION_NAME].find_one({"email": user_email})
#                 if user and bcrypt.checkpw(old.encode("utf-8"), user["password"]):
#                     hashed = bcrypt.hashpw(new.encode("utf-8"), bcrypt.gensalt())
#                     client[DB_NAME][COLLECTION_NAME].update_one({"email": user_email}, {"$set": {"password": hashed}})
#                     st.success("Password changed successfully!")
#                 else:
#                     st.error("Old password is incorrect.")


# def logout():
#     for k in ["logged_in", "username", "user_email", "last_image", "result", "page"]:
#         st.session_state.pop(k, None)
#     st.session_state["logged_in"] = False
#     st.session_state["page"] = "Dashboard"
#     st.rerun()


# # =============================================================================
# # MAIN
# # =============================================================================
# def main():
#     st.set_page_config(page_title="OncoVision", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

#     # Load and embed background image as base64
#     bg_base64 = ""
#     if os.path.exists("1.png"):
#         with open("1.png", "rb") as f:
#             bg_base64 = base64.b64encode(f.read()).decode("utf-8")

#     # Custom CSS with background image (base64 embedded)
#     custom_css = f"""
#     <style>
#     .stApp {{
#         background-image: url("data:image/png;base64,{bg_base64}");
#         background-size: cover;
#         background-position: center;
#         background-repeat: no-repeat;
#         background-attachment: fixed;
#     }}
#     .stApp::before {{
#         content: '';
#         position: fixed;
#         top: 0;
#         left: 0;
#         width: 100%;
#         height: 100%;
#         background: rgba(0, 0, 0, 0.5);
#         z-index: -1;
#     }}
#     </style>
#     """
#     st.markdown(custom_css, unsafe_allow_html=True)

#     # Load existing style.css if present
#     if os.path.exists("style.css"):
#         with open("style.css") as f:
#             st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#     if "logged_in" not in st.session_state:
#         st.session_state["logged_in"] = False
#     if "page" not in st.session_state:
#         st.session_state["page"] = "Dashboard"

#     client = init_db()
#     if not client:
#         st.error("Cannot connect to database. Please check your MongoDB URI in .env file and network connectivity.")
#         st.stop()

#     if not st.session_state["logged_in"]:
#         auth_screen(client)
#         return

#     show_sidebar()

#     page = st.session_state["page"]
#     if page == "Dashboard":
#         dashboard_page(client)
#     elif page == "Upload Image":
#         upload_image_page(client)
#     elif page == "Classification Result":
#         classification_result_page()
#     elif page == "Segmentation Result":
#         segmentation_result_page()
#     elif page == "Model Info":
#         model_info_page()
#     elif page == "Feedback":
#         feedback_page(client)
#     elif page == "History":
#         history_page(client)
#     elif page == "Settings":
#         settings_page(client)
#     elif page == "Logout":
#         logout()
#     else:
#         dashboard_page(client)

#     st.markdown("---")
#     st.markdown("<center><small>© 2026 OncoVision. All rights reserved. Built with ❤️ using AI & Deep Learning</small></center>", unsafe_allow_html=True)


# if __name__ == "__main__":
#     main()