import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from tqdm import tqdm

# =========================
# CONFIG
# =========================
DATA_DIR = "data"                  # Normal / Carries folders
MODEL_PATH = "checkpoints/best_model_balanced.pth"
OUTPUT_DIR = "inference_outputs"

IMG_SIZE = 512
THRESHOLD = 0.5

SEPARATOR_WIDTH = 6
SEPARATOR_COLOR = (255, 255, 255)  # white separator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("✅ Model loaded")

# =========================
# PREPROCESS
# =========================
def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # HWC → CHW
    return torch.tensor(img).unsqueeze(0)

# =========================
# OVERLAY
# =========================
def create_overlay(image, mask, color=(255, 0, 0)):
    overlay = image.copy()
    overlay[mask > 0] = (
        0.6 * overlay[mask > 0] + 0.4 * np.array(color)
    )
    return overlay.astype(np.uint8)

# =========================
# INFERENCE
# =========================
for folder in ["Normal", "Carries"]:
    img_dir = os.path.join(DATA_DIR, folder)
    out_dir = os.path.join(OUTPUT_DIR, folder)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n🔍 Running inference on {folder} images")

    for file in tqdm(os.listdir(img_dir)):
        if not file.endswith(".png") or "mask" in file.lower():
            continue

        img_path = os.path.join(img_dir, file)
        mask_path = os.path.join(img_dir, file.replace(".png", "-mask.png"))

        # Original image
        orig = cv2.imread(img_path)
        orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

        # Ground truth mask
        gt_mask = None
        if os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, 0)
            gt_mask = cv2.resize(gt_mask, (IMG_SIZE, IMG_SIZE))
            gt_mask = (gt_mask > 127).astype(np.uint8)

        # Prediction
        x = preprocess(img_path).to(DEVICE)
        with torch.no_grad():
            pred = torch.sigmoid(model(x))[0, 0].cpu().numpy()

        pred_mask = (pred > THRESHOLD).astype(np.uint8)

        # Visualizations
        pred_overlay = create_overlay(orig_rgb, pred_mask, color=(255, 0, 0))

        # Panels
        panels = []

        panels.append(orig_rgb)

        if gt_mask is not None:
            gt_vis = np.stack([gt_mask * 255] * 3, axis=-1)
            panels.append(gt_vis)
        else:
            panels.append(np.zeros_like(orig_rgb))

        pred_vis = np.stack([pred_mask * 255] * 3, axis=-1)
        panels.append(pred_vis)

        panels.append(pred_overlay)

        # =========================
        # ADD SEPARATION LINES
        # =========================
        h, _, _ = panels[0].shape
        separator = np.ones((h, SEPARATOR_WIDTH, 3), dtype=np.uint8)
        separator[:] = SEPARATOR_COLOR

        combined = panels[0]
        for p in panels[1:]:
            combined = np.concatenate([combined, separator, p], axis=1)

        # Save
        out_path = os.path.join(out_dir, file.replace(".png", "_panel.png"))
        cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

print("\n🎉 Inference completed!")
print(f"📁 Results saved in: {OUTPUT_DIR}")
