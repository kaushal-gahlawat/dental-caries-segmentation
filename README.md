# Automated Dental Caries Segmentation & Classification

This repository contains the implementation for automated segmentation (and optional classification) of dental caries from dental X-ray images using deep learning.

The project was developed as part of a **Medical Image Processing Hackathon**.

---

## 📁 Project Structure

med_hackathon/
│
├── data/
│ ├── Normal/
│ │ ├── normal-1.png
│ │ ├── normal-1-mask.png
│ │ └── ...
│ └── Carries/
│ ├── benign-1.png
│ ├── benign-1-mask.png
│ └── ...
│
├── checkpoints/
│ └── best_model_balanced.pth
│
├── train_seg.py
├── infer.py
├── requirements.txt
└── README.md

---

## 🧠 File Descriptions

### 🔹 `train_seg.py` (MAIN TRAINING SCRIPT)
- Primary script used for **training the segmentation model**
- Implements:
  - U-Net with EfficientNet-B0 encoder
  - Dice + BCE based loss
  - Validation Dice & IoU calculation
  - Learning rate scheduling
  - Early stopping
  - Best model checkpoint saving
- **This is the main file used in the final experiments**

---

### 🔹 `infer_seg.py`
- Used for **model inference and visualization**
- Generates:
  - Original dental X-ray
  - Ground truth mask (if available)
  - Predicted segmentation mask
  - Overlay visualization
- Saves **side-by-side panels** suitable for PPT presentation

---

### 🔹 `checkpoints/best_model_balanced.pth`
- Saved best segmentation model
- Selected based on **highest validation Dice score**

---

### 🔹 `data/`
- Contains dataset organized into:
  - `Normal/` → Non-carious images
  - `Carries/` → Carious images
- Each image has a corresponding `*-mask.png`

---

## 📊 Evaluation Metrics Used

- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)
- Pixel-wise Accuracy
- Precision & Recall (pixel-level)

---

## 🚀 How to Run

### Train the Model
```bash
python train_seg.py

Run Inference & Generate Visualizations
python infer_seg.py
