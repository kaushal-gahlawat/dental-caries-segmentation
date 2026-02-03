import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =========================
# CONFIG - BALANCED & PROVEN
# =========================
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

IMG_SIZE = 512  # 🔥 KEEP LARGE - dental images need detail
BATCH_SIZE = 8
EPOCHS = 150
LR = 5e-5  # 🔥 LOWER LR - your original was too high
PATIENCE = 25
NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# MODERATE AUGMENTATIONS - Not too weak, not too strong
# =========================
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    
    # Geometric - MODERATE
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=15, p=0.5),  # 🔥 Reduced from 30 degrees
    A.ShiftScaleRotate(
        shift_limit=0.0625,
        scale_limit=0.1,
        rotate_limit=10,
        p=0.5
    ),
    
    # Intensity - CAREFULLY TUNED
    A.CLAHE(clip_limit=2.0, p=0.5),  # 🔥 Reduced from 4.0
    A.RandomBrightnessContrast(
        brightness_limit=0.2,  # 🔥 Reduced from 0.3
        contrast_limit=0.2,
        p=0.6
    ),
    A.RandomGamma(gamma_limit=(85, 115), p=0.4),  # 🔥 Narrower range
    
    # Light noise/blur - OPTIONAL
    A.OneOf([
        A.GaussianBlur(blur_limit=3, p=1.0),
        A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
    ], p=0.3),
    
    # Normalize
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# =========================
# DATASET
# =========================
class DentalSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
            mask = (mask > 127).float().unsqueeze(0)
        
        return img, mask

# =========================
# LOAD DATA
# =========================
def collect_image_mask_paths(base_dir):
    images, masks = [], []
    for folder in ["Normal", "Carries"]:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            continue
        
        for file in os.listdir(folder_path):
            if "-mask" in file.lower() or not file.lower().endswith(".png"):
                continue
            
            img_path = os.path.join(folder_path, file)
            possible_masks = [
                file.replace(".png", "-mask.png"),
                file.replace(".png", "_mask.png"),
            ]
            
            for mask_name in possible_masks:
                potential_mask = os.path.join(folder_path, mask_name)
                if os.path.exists(potential_mask):
                    images.append(img_path)
                    masks.append(potential_mask)
                    break
    
    return images, masks

# =========================
# METRICS
# =========================
def calculate_dice(preds, masks, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    dice = (2. * intersection + 1e-8) / (
        preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-8
    )
    return dice.mean().item()

def calculate_iou(preds, masks, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou.mean().item()

# =========================
# MAIN
# =========================
def main():
    print(f"🎯 BALANCED TRAINING - Proven Configuration")
    print(f"Device: {DEVICE}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LR}")
    
    # Load data
    image_paths, mask_paths = collect_image_mask_paths(DATA_DIR)
    print(f"📊 Total samples: {len(image_paths)}")
    
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.15, random_state=42
    )
    
    print(f"📊 Train: {len(train_imgs)} | Val: {len(val_imgs)}")
    
    train_ds = DentalSegDataset(train_imgs, train_masks, transform=train_transform)
    val_ds = DentalSegDataset(val_imgs, val_masks, transform=val_transform)
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # 🔥 MODEL - Use ResNet34 (often better than EfficientNet for medical)
    print("🔄 Loading model...")
    
    try:
        # Try ResNet34 first - proven for medical imaging
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        ).to(DEVICE)
        print("✅ Using ResNet34 encoder (proven for medical imaging)")
    except:
        # Fallback to EfficientNet-B0
        model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        ).to(DEVICE)
        print("✅ Using EfficientNet-B0 encoder")
    
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 🔥 SIMPLE BUT EFFECTIVE LOSS
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    focal_loss = smp.losses.FocalLoss(mode="binary")
    
    def combined_loss(preds, masks):
        # Simple 70/30 split works well
        return 0.7 * dice_loss(preds, masks) + 0.3 * focal_loss(preds, masks)
    
    # 🔥 CONSERVATIVE OPTIMIZER
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    # 🔥 SIMPLE SCHEDULER - Reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=8,
        verbose=True,
        min_lr=1e-7
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    best_dice = 0.0
    best_iou = 0.0
    epochs_no_improve = 0
    
    print("\n🚀 Starting BALANCED training...\n")
    
    for epoch in range(EPOCHS):
        # TRAINING
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                loss = combined_loss(preds, masks)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            with torch.no_grad():
                train_dice += calculate_dice(torch.sigmoid(preds), masks)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{train_dice / (pbar.n + 1):.4f}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        
        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [VAL]")
            for imgs, masks in pbar:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                
                with torch.cuda.amp.autocast():
                    preds = model(imgs)
                    loss = combined_loss(preds, masks)
                
                val_loss += loss.item()
                
                preds_prob = torch.sigmoid(preds)
                val_dice += calculate_dice(preds_prob, masks)
                val_iou += calculate_iou(preds_prob, masks)
                
                pbar.set_postfix({
                    'dice': f'{val_dice / (pbar.n + 1):.4f}',
                    'iou': f'{val_iou / (pbar.n + 1):.4f}'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        # Step scheduler
        scheduler.step(avg_val_dice)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(
            f"\n📊 Epoch {epoch+1}/{EPOCHS}\n"
            f"   Train - Loss: {avg_train_loss:.4f} | Dice: {avg_train_dice:.4f}\n"
            f"   Val   - Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f} | IoU: {avg_val_iou:.4f}\n"
            f"   LR: {current_lr:.2e}"
        )
        
        # SAVE BEST
        if avg_val_dice > best_dice + 1e-4:
            improvement = avg_val_dice - best_dice
            best_dice = avg_val_dice
            best_iou = avg_val_iou
            epochs_no_improve = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'best_iou': best_iou,
            }, f"{CHECKPOINT_DIR}/best_model_balanced.pth")
            
            print(f"✅ NEW BEST! Dice: {best_dice:.4f} (+{improvement:.4f}) | IoU: {best_iou:.4f}")
        else:
            epochs_no_improve += 1
            print(f"⚠️  No improvement for {epochs_no_improve} epochs")
        
        # EARLY STOPPING
        if epochs_no_improve >= PATIENCE:
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            break
        
        print("-" * 80)
    
    print(f"\n🏆 Training completed!")
    print(f"   Best Dice: {best_dice:.4f}")
    print(f"   Best IoU:  {best_iou:.4f}")
    print(f"   Saved to: {CHECKPOINT_DIR}/best_model_balanced.pth")

if __name__ == '__main__':
    main()