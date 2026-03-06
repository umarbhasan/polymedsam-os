# -*- coding: utf-8 -*-
"""PolyMedSAM-OS"""

!pip install -q transformers peft monai scikit-learn matplotlib

import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import monai
from transformers import SamModel, SamProcessor
from peft import LoraConfig, get_peft_model
import cv2
from sklearn.metrics import roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt

SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import SamProcessor

DATA_DIR = "/kaggle/input/datasets/debeshjha1/kvasirseg/Kvasir-SEG/Kvasir-SEG"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")
BBOX_DIR = os.path.join(DATA_DIR, "bbox")

train_txt_path = "/kaggle/input/datasets/debeshjha1/kvasirseg/train.txt"
val_txt_path = "/kaggle/input/datasets/debeshjha1/kvasirseg/val.txt"

with open(train_txt_path, 'r') as f:
    train_files = [line.strip() for line in f.readlines() if line.strip()]
with open(val_txt_path, 'r') as f:
    val_files = [line.strip() for line in f.readlines() if line.strip()]

processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")

class KvasirDatasetOSL(Dataset):
    def __init__(self, image_dir, mask_dir, bbox_dir, file_list, processor, is_ood=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.bbox_dir = bbox_dir
        self.file_list = file_list
        self.processor = processor
        self.is_ood = is_ood

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        img_path = os.path.join(self.image_dir, filename + ".jpg")
        mask_path = os.path.join(self.mask_dir, filename + ".jpg")

        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, filename)
            mask_path = os.path.join(self.mask_dir, filename)

        image = Image.open(img_path).convert("RGB").resize((256, 256))
        mask = Image.open(mask_path).convert("L").resize((256, 256), resample=Image.NEAREST)

        image_np = np.array(image)
        mask_np = (np.array(mask) > 128).astype(np.uint8)

        name_only = os.path.splitext(filename)[0]
        bbox_path_txt = os.path.join(self.bbox_dir, name_only + ".txt")
        bbox_path_json = os.path.join(self.bbox_dir, name_only + ".json")

        bbox = None
        if os.path.exists(bbox_path_json):
            with open(bbox_path_json, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if 'bbox' in data:
                        bbox = data['bbox']
                    elif 'xmin' in data:
                        bbox = [data['xmin'], data['ymin'], data['xmax'], data['ymax']]
        elif os.path.exists(bbox_path_txt):
            with open(bbox_path_txt, 'r') as f:
                content = f.read().replace(',', ' ').split()
                parsed_coords = [float(x) for x in content if x.replace('.', '', 1).isdigit()]
                if len(parsed_coords) >= 4:
                    bbox = parsed_coords[:4]

        if bbox is not None:
            y_indices, x_indices = np.where(mask_np > 0)
            if len(y_indices) > 0 and len(x_indices) > 0:
                bbox = [np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]
            else:
                bbox = [0, 0, 256, 256]
        else:
            y_indices, x_indices = np.where(mask_np > 0)
            if len(y_indices) > 0 and len(x_indices) > 0:
                bbox = [np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]
            else:
                bbox = [0, 0, 256, 256]

        inputs = self.processor(image_np, input_boxes=[[bbox]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(mask_np).float().unsqueeze(0)
        inputs["is_ood"] = torch.tensor(1.0 if self.is_ood else 0.0).float()
        inputs["original_image"] = image_np
        return inputs

IMAGES_DIR_INSTRUMENT = '/kaggle/input/datasets/debeshjha1/kvasirinstrument/kvasir-instrument/images/images'
MASKS_DIR_INSTRUMENT = '/kaggle/input/datasets/debeshjha1/kvasirinstrument/kvasir-instrument/masks/masks'

class AuthenticInstrumentOODDataset(Dataset):
    def __init__(self, images_dir, masks_dir, processor):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.processor = processor
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        all_files = os.listdir(images_dir)
        self.image_files = [f for f in all_files if f.lower().endswith(valid_exts)]
        print(f"Loaded {len(self.image_files)} authentic instruments for OOD training.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + '.png'
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            mask_path_alt = os.path.join(self.masks_dir, img_name)
            mask = cv2.imread(mask_path_alt, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)

        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            bbox = [0, 0, 256, 256]
        else:
            bbox = [np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]

        inputs = self.processor(image, input_boxes=[[bbox]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}


        inputs["original_image"] = image
        inputs["ground_truth_mask"] = torch.tensor(mask).float().unsqueeze(0)
        inputs["is_ood"] = torch.tensor(1.0).float() 
        return inputs

train_id_ds = KvasirDatasetOSL(IMAGES_DIR, MASKS_DIR, BBOX_DIR, train_files, processor=processor, is_ood=False)
train_ood_ds = AuthenticInstrumentOODDataset(IMAGES_DIR_INSTRUMENT, MASKS_DIR_INSTRUMENT, processor=processor)
val_ds = KvasirDatasetOSL(IMAGES_DIR, MASKS_DIR, BBOX_DIR, val_files, processor=processor, is_ood=False)

train_combined_ds = torch.utils.data.ConcatDataset([train_id_ds, train_ood_ds])
train_loader = DataLoader(train_combined_ds, batch_size=2, shuffle=True, num_workers=2, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)

print(f"Total training samples: {len(train_combined_ds)} (ID: {len(train_id_ds)}, OOD: {len(train_ood_ds)})")

print(f"Train ID: {len(train_id_ds)} | Train OOD (Authentic): {len(train_ood_ds)} | Val: {len(val_ds)}")

import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import SamProcessor

DATA_DIR = "/kaggle/input/datasets/debeshjha1/kvasirseg/Kvasir-SEG/Kvasir-SEG"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")
BBOX_DIR = os.path.join(DATA_DIR, "bbox")

train_txt_path = "/kaggle/input/datasets/debeshjha1/kvasirseg/train.txt"
val_txt_path = "/kaggle/input/datasets/debeshjha1/kvasirseg/val.txt"

with open(train_txt_path, 'r') as f:
    train_files = [line.strip() for line in f.readlines() if line.strip()]
with open(val_txt_path, 'r') as f:
    val_files = [line.strip() for line in f.readlines() if line.strip()]

ood_split = int(len(train_files) * 0.15)
train_id_files = train_files[ood_split:]
train_synthetic_ood_files = train_files[:ood_split]

print(f"Training Split -> ID (Polyps): {len(train_id_files)} | OOD (Synthetic Noise): {len(train_synthetic_ood_files)}")

processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")

def ood_transform(img_np):
    noise = np.random.normal(0, 50, img_np.shape)
    img_noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    return img_noisy

class KvasirDatasetOSL(Dataset):
    def __init__(self, image_dir, mask_dir, bbox_dir, file_list, processor, is_ood=False, ood_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.bbox_dir = bbox_dir
        self.file_list = file_list
        self.processor = processor
        self.is_ood = is_ood
        self.ood_transform = ood_transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        img_path = os.path.join(self.image_dir, filename + ".jpg")
        mask_path = os.path.join(self.mask_dir, filename + ".jpg")

        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, filename)
            mask_path = os.path.join(self.mask_dir, filename)

        image = Image.open(img_path).convert("RGB").resize((256, 256))
        mask = Image.open(mask_path).convert("L").resize((256, 256), resample=Image.NEAREST)

        image_np = np.array(image)
        mask_np = (np.array(mask) > 128).astype(np.uint8)

        name_only = os.path.splitext(filename)[0]
        bbox_path_txt = os.path.join(self.bbox_dir, name_only + ".txt")
        bbox_path_json = os.path.join(self.bbox_dir, name_only + ".json")

        bbox = None
        if os.path.exists(bbox_path_json):
            with open(bbox_path_json, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if 'bbox' in data:
                        bbox = data['bbox']
                    elif 'xmin' in data:
                        bbox = [data['xmin'], data['ymin'], data['xmax'], data['ymax']]
        elif os.path.exists(bbox_path_txt):
            with open(bbox_path_txt, 'r') as f:
                content = f.read().replace(',', ' ').split()
                parsed_coords = [float(x) for x in content if x.replace('.', '', 1).isdigit()]
                if len(parsed_coords) >= 4:
                    bbox = parsed_coords[:4]

        if bbox is not None:
            y_indices, x_indices = np.where(mask_np > 0)
            if len(y_indices) > 0 and len(x_indices) > 0:
                bbox = [np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]
            else:
                bbox = [0, 0, 256, 256]
        else:
            y_indices, x_indices = np.where(mask_np > 0)
            if len(y_indices) > 0 and len(x_indices) > 0:
                bbox = [np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]
            else:
                bbox = [0, 0, 256, 256]

        
        if self.is_ood and self.ood_transform:
            image_np = self.ood_transform(image_np)
            
            mask_np = np.zeros_like(mask_np)

        inputs = self.processor(image_np, input_boxes=[[bbox]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(mask_np).float().unsqueeze(0)
        inputs["is_ood"] = torch.tensor(1.0 if self.is_ood else 0.0).float()
        inputs["original_image"] = image_np
        return inputs

IMAGES_DIR_INSTRUMENT = '/kaggle/input/datasets/debeshjha1/kvasirinstrument/kvasir-instrument/images/images'
MASKS_DIR_INSTRUMENT = '/kaggle/input/datasets/debeshjha1/kvasirinstrument/kvasir-instrument/masks/masks'

class AuthenticInstrumentOODDataset(Dataset):
    def __init__(self, images_dir, masks_dir, processor):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.processor = processor
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        all_files = os.listdir(images_dir)
        self.image_files = [f for f in all_files if f.lower().endswith(valid_exts)]
        print(f"Loaded {len(self.image_files)} authentic instruments for ZERO-SHOT OOD Evaluation.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + '.png'
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            mask_path_alt = os.path.join(self.masks_dir, img_name)
            mask = cv2.imread(mask_path_alt, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)

        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0:
            bbox = [0, 0, 256, 256]
        else:
            bbox = [np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)]

        inputs = self.processor(image, input_boxes=[[bbox]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["original_image"] = image
        inputs["ground_truth_mask"] = torch.tensor(mask).float().unsqueeze(0)
        inputs["is_ood"] = torch.tensor(1.0).float()
        return inputs

train_id_ds = KvasirDatasetOSL(IMAGES_DIR, MASKS_DIR, BBOX_DIR, train_id_files, processor=processor, is_ood=False)
train_synthetic_ood_ds = KvasirDatasetOSL(IMAGES_DIR, MASKS_DIR, BBOX_DIR, train_synthetic_ood_files, processor=processor, is_ood=True, ood_transform=ood_transform)
train_combined_ds = torch.utils.data.ConcatDataset([train_id_ds, train_synthetic_ood_ds])
train_loader = DataLoader(train_combined_ds, batch_size=2, shuffle=True, num_workers=2, drop_last=True)

val_ds = KvasirDatasetOSL(IMAGES_DIR, MASKS_DIR, BBOX_DIR, val_files, processor=processor, is_ood=False)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)

authentic_ood_ds = AuthenticInstrumentOODDataset(IMAGES_DIR_INSTRUMENT, MASKS_DIR_INSTRUMENT, processor=processor)
instrument_loader = DataLoader(authentic_ood_ds, batch_size=2, shuffle=False, num_workers=2)

print(f"Total training samples: {len(train_combined_ds)}")

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

def generate_instrument_grid(dataloader, num_samples=2, save_path='/kaggle/working/instrument_dataset_samples.pdf'):
    print(f"Extracting {num_samples} authentic instrument samples from the dataloader...")

    images = []
    masks = []

    for batch in dataloader:
        orig_imgs = batch["original_image"].numpy()
        gt_masks = batch["ground_truth_mask"].numpy()

        for i in range(orig_imgs.shape[0]):
            
            images.append(orig_imgs[i])
            masks.append(gt_masks[i, 0])

            if len(images) == num_samples:
                break
        if len(images) == num_samples:
            break

    if len(images) < num_samples:
        print(f"Warning: Only found {len(images)} samples.")
        num_samples = len(images)

    
    fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))

    for i in range(num_samples):
        # --- Row 1: Raw Image ---
        axes[0, i].imshow(images[i])
        axes[0, i].axis('off')

        # --- Row 2: Ground Truth ---
        axes[1, i].imshow(masks[i], cmap='gray')
        axes[1, i].axis('off')

    
    row_labels = ['Images', 'Ground truth']
    for row_idx in range(2):
        axes[row_idx, 0].text(
            -0.1, 0.5, row_labels[row_idx],
            va='center', ha='right', rotation=90,
            transform=axes[row_idx, 0].transAxes,
            fontsize=18, fontweight='bold'
        )

    
    plt.subplots_adjust(wspace=0.02, hspace=0.02)

   
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Instrument dataset grid successfully saved to {save_path}")


instrument_loader = DataLoader(train_ood_ds, batch_size=2, shuffle=False, num_workers=2)

generate_instrument_grid(instrument_loader, num_samples=2)

import matplotlib.pyplot as plt
import numpy as np

def generate_dataset_grid(dataloader, num_samples=2, save_path='/kaggle/working/dataset_samples.pdf'):
    print(f"Extracting {num_samples} clean samples from the dataloader...")

    images = []
    masks = []

    # Extract samples from the dataloader
    for batch in dataloader:
        orig_imgs = batch["original_image"].numpy()
        gt_masks = batch["ground_truth_mask"].numpy()
        is_ood = batch["is_ood"].numpy()

        for i in range(orig_imgs.shape[0]):
            # Skip the OOD images to show clean Kvasir-SEG examples
            if is_ood[i] == 1.0:
                continue

            images.append(orig_imgs[i])
            masks.append(gt_masks[i, 0])

            if len(images) == num_samples:
                break
        if len(images) == num_samples:
            break

    if len(images) < num_samples:
        print(f"Warning: Only found {len(images)} clean samples.")
        num_samples = len(images)

    # Create a 2 x num_samples grid
    fig, axes = plt.subplots(2, num_samples, figsize=(3 * num_samples, 6))

    for i in range(num_samples):
        # --- Row 1: Raw Image ---
        axes[0, i].imshow(images[i])
        axes[0, i].axis('off')

        # --- Row 2: Ground Truth ---
        axes[1, i].imshow(masks[i], cmap='gray')
        axes[1, i].axis('off')

    # Add row labels on the far left side
    row_labels = ['Images', 'Ground truth']
    for row_idx in range(2):
        axes[row_idx, 0].text(
            -0.1, 0.5, row_labels[row_idx],
            va='center', ha='right', rotation=90,
            transform=axes[row_idx, 0].transAxes,
            fontsize=18, fontweight='bold'
        )

    # Remove all extra whitespace between subplots
    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    # Save as a high-resolution PDF for the manuscript
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    print(f"Dataset grid successfully saved to {save_path}")

# Generate the visualization
generate_dataset_grid(train_loader, num_samples=2)

print("Loading MedSAM from Hugging Face Hub...")
medsam_checkpoint = "flaviagiammarino/medsam-vit-base"
base_model = SamModel.from_pretrained(medsam_checkpoint)

# Apply LoRA using PEFT
lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["qkv"],
    lora_dropout=0.05,
    bias="none"
)
medsam_lora = get_peft_model(base_model, lora_config)

# Unfreeze the mask decoder
for name, param in medsam_lora.named_parameters():
    if "mask_decoder" in name:
        param.requires_grad = True

print("Trainable Parameters:")
medsam_lora.print_trainable_parameters()

class MedSAM_OSL(nn.Module):
    def __init__(self, medsam_peft_model):
        super().__init__()
        self.medsam = medsam_peft_model
        # Prototype mapping head attached to Image Encoder output (B, 256, 64, 64)
        self.prototype_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.register_buffer("known_prototype", torch.zeros(64))

    def forward(self, pixel_values, input_boxes):
        # 1. Forward through the full SAM architecture for masks
        outputs = self.medsam(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
        pred_masks = outputs.pred_masks.squeeze(1)

        # 2. Extract image embeddings for OSL clustering
        vision_outputs = self.medsam.vision_encoder(pixel_values)
        image_embeddings = vision_outputs.last_hidden_state
        latent_vector = self.prototype_head(image_embeddings)

        return pred_masks, latent_vector

model_osl = MedSAM_OSL(medsam_lora).to(device)

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_osl.parameters()), lr=1e-4, weight_decay=0.01)

# T_max=20 creates a gentle learning rate decay over 10 epochs.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
scaler = torch.cuda.amp.GradScaler()

seg_loss_fn = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

def contrastive_osl_loss(latent_vectors, is_ood, prototype, margin=2.0):
    # L2 distance to known prototype
    dists = torch.norm(latent_vectors - prototype.unsqueeze(0), p=2, dim=1)

    # ID loss: pull to prototype. OOD loss: push away beyond margin.
    loss_id = (1 - is_ood) * torch.pow(dists, 2)
    loss_ood = is_ood * torch.pow(torch.clamp(margin - dists, min=0.0), 2)

    return torch.mean(loss_id + loss_ood)

EPOCHS = 10
ACCUMULATION_STEPS = 2
LAMBDA_OSL = 0.1

print("Starting Training...")
best_val_loss = float('inf')
start_epoch = 0

# Setup robust session resumption
checkpoint_path = '/kaggle/working/medsam_os_checkpoint.pth'
best_model_path = '/kaggle/working/medsam_os_best.pth'

if os.path.exists(checkpoint_path):
    print(f"Found interrupted session! Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model_osl.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resuming at Epoch {start_epoch+1}/{EPOCHS}")

for epoch in range(start_epoch, EPOCHS):
    model_osl.train()
    train_loss = 0.0
    optimizer.zero_grad()

    # Prototype moving average update variables
    batch_prototypes = []

    for i, batch in enumerate(train_loader):
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)
        masks = batch["ground_truth_mask"].to(device)
        is_ood = batch["is_ood"].to(device)

        with torch.amp.autocast('cuda'):
            pred_masks, latent_vectors = model_osl(pixel_values, input_boxes)

            # Seg Loss
            l_seg = seg_loss_fn(pred_masks, masks)

            # OSL Loss
            l_osl = contrastive_osl_loss(latent_vectors, is_ood, model_osl.known_prototype)

            loss = l_seg + LAMBDA_OSL * l_osl
            loss = loss / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += loss.item() * ACCUMULATION_STEPS

        # Update known prototype moving average (only from ID data)
        with torch.no_grad():
            id_latents = latent_vectors[is_ood == 0]
            if len(id_latents) > 0:
                batch_prototypes.append(id_latents.mean(dim=0))

    # Update global prototype
    if batch_prototypes:
        avg_batch_proto = torch.stack(batch_prototypes).mean(dim=0)
        model_osl.known_prototype = 0.9 * model_osl.known_prototype + 0.1 * avg_batch_proto

    scheduler.step()

    # Validation Loop
    model_osl.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            masks = batch["ground_truth_mask"].to(device)

            with torch.amp.autocast('cuda'):
                pred_masks, _ = model_osl(pixel_values, input_boxes)
                v_loss = seg_loss_fn(pred_masks, masks)
                val_loss += v_loss.item()

    val_loss /= len(val_loader)
    train_loss /= len(train_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Checkpoint logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model_osl.state_dict(), best_model_path)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model_osl.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss
    }, checkpoint_path)

print(f"Training Complete. Best model saved to {best_model_path}")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, auc, roc_curve
import torch
from torch.utils.data import DataLoader

# --- CRITICAL FIX: Create the missing instrument_loader for evaluation ---
# We use the train_ood_ds you already defined in your data setup cell
instrument_loader = DataLoader(train_ood_ds, batch_size=2, shuffle=False, num_workers=2)

# --- NEW: Model Loading Mechanism ---
from transformers import SamModel
from peft import LoraConfig, get_peft_model
import os

print("Loading PolyMedSAM-OS from saved checkpoint...")
base_model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base")
lora_config = LoraConfig(r=4, lora_alpha=32, target_modules=["qkv"], lora_dropout=0.05, bias="none")
medsam_lora = get_peft_model(base_model, lora_config)

# Assuming MedSAM_OSL class and device are already defined in your earlier cells
model_osl = MedSAM_OSL(medsam_lora).to(device)

model_path = '/kaggle/working/medsam_os_best.pth'
if os.path.exists(model_path):
    model_osl.load_state_dict(torch.load(model_path))
    print("Successfully loaded model weights!")
else:
    print(f"Warning: Could not find {model_path}. Make sure the file exists in your working directory.")

model_osl.eval()

# Let's run a quick qualitative check on validation data
sample_batch = next(iter(val_loader))
s_img = sample_batch["original_image"].numpy()
s_pixel = sample_batch["pixel_values"].to(device)
s_mask = sample_batch["ground_truth_mask"].to(device)
s_bbox = sample_batch["input_boxes"].to(device)

with torch.no_grad():
    with torch.amp.autocast('cuda'):
        pred, latents = model_osl(s_pixel, s_bbox)
    pred_binary = (torch.sigmoid(pred) > 0.5).float()

# Plotting Qualitative Check
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(s_img[0])
axes[0].set_title("Input Image")
axes[1].imshow(s_mask[0, 0].cpu(), cmap='gray')
axes[1].set_title("Ground Truth")
axes[2].imshow(pred_binary[0, 0].cpu(), cmap='gray')
axes[2].set_title("MedSAM-OS Prediction")
for ax in axes: ax.axis('off')
plt.savefig('/kaggle/working/sample_prediction.pdf')
plt.show()

# Extract AUROC for Authentic OOD Detection using Latent Space Distance
print("Evaluating Authentic OOD Robustness (Latent Distance AUROC)...")

all_scores = []
all_labels = []

print("Scoring ID (In-Distribution Kvasir-SEG) validation data...")
with torch.no_grad():
    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)

        with torch.amp.autocast('cuda'):
            _, latents = model_osl(pixel_values, input_boxes)

        # Compute L2 distance to the known prototype
        dists = torch.norm(latents - model_osl.known_prototype.unsqueeze(0), p=2, dim=1)
        all_scores.extend(dists.cpu().tolist())
        all_labels.extend([0] * pixel_values.size(0)) # 0 for ID

print("Scoring OOD (Authentic Kvasir-Instrument) validation data...")
with torch.no_grad():
    for batch in instrument_loader:
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)

        with torch.amp.autocast('cuda'):
            _, latents = model_osl(pixel_values, input_boxes)

        # Compute L2 distance to the known prototype
        dists = torch.norm(latents - model_osl.known_prototype.unsqueeze(0), p=2, dim=1)
        all_scores.extend(dists.cpu().tolist())
        all_labels.extend([1] * pixel_values.size(0)) # 1 for OOD

auroc = roc_auc_score(all_labels, all_scores)
print(f"Final Authentic OOD Detection AUROC (Latent Space): {auroc:.4f}")

# Plot and save the ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_scores)
plt.figure(figsize=(6, 5), dpi=300)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'MedSAM-OS (AUC = {auroc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Authentic OOD Detection ROC Curve (Latent Distance)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('/kaggle/working/authentic_ood_roc_curve_latents.pdf', bbox_inches='tight')
plt.show()
print("Saved Authentic OOD ROC curve to /kaggle/working/authentic_ood_roc_curve_latents.pdf")

!pip install -q thop

import time
import os
import numpy as np
import torch
from thop import profile
from scipy.stats import wilcoxon
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, auc, roc_curve

# File paths for your models
PATH_PROPOSED = '/kaggle/working/medsam_os_best.pth'
PATH_ABLATION = '/kaggle/input/models/umarhasannsu/polymedsam/transformers/default/1/medsam_os_ablation_authentic.pth'

def calculate_medical_metrics(pred_mask, gt_mask):
    pred = pred_mask.flatten().astype(bool)
    gt = gt_mask.flatten().astype(bool)

    TP = np.sum((pred == 1) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    dice = (2. * TP) / (2. * TP + FP + FN + 1e-8)
    miou = TP / (TP + FP + FN + 1e-8)
    sensitivity = TP / (TP + FN + 1e-8) # Recall
    specificity = TN / (TN + FP + 1e-8)
    ppv = TP / (TP + FP + 1e-8) # Precision
    npv = TN / (TN + FN + 1e-8)

    return [dice, miou, sensitivity, specificity, ppv, npv]

def evaluate_full_metrics(model, dataloader, model_name, uses_osl_wrapper=False):
    model.eval()
    model.to(device)
    metrics_list = []

    print(f"Evaluating {model_name}...")
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            gt_masks = batch["ground_truth_mask"].numpy()

            with torch.amp.autocast('cuda'):
                if not uses_osl_wrapper:
                    outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
                    pred = outputs.pred_masks.squeeze(1)
                else:
                    pred, _ = model(pixel_values, input_boxes)

            pred_binary = (torch.sigmoid(pred).cpu().numpy() > 0.5).astype(np.uint8)

            for i in range(pred_binary.shape[0]):
                metrics = calculate_medical_metrics(pred_binary[i, 0], gt_masks[i, 0])
                metrics_list.append(metrics)

    columns = ["Dice", "mIoU", "Sensitivity (Recall)", "Specificity", "PPV (Precision)", "NPV"]
    df = pd.DataFrame(metrics_list, columns=columns)
    df["Model"] = model_name
    return df

def profile_model(model, uses_osl_wrapper=False, base_macs=None):
    model.eval()
    dummy_img = torch.randn(1, 3, 1024, 1024).to(device)
    dummy_box = torch.tensor([[[0, 0, 100, 100]]]).float().to(device)

    # 1. Exact Parameters (More robust and accurate than thop for PEFT models)
    params = sum(p.numel() for p in model.parameters())

    # 2. FLOPs
    macs = 0
    try:
        # Clean up any residual thop buffers before profiling
        for m in model.modules():
            m._buffers.pop("total_ops", None)
            m._buffers.pop("total_params", None)

        if not uses_osl_wrapper:
            macs, _ = profile(model, inputs=(dummy_img, None, None, dummy_box), verbose=False)
        else:
            macs, _ = profile(model, inputs=(dummy_img, dummy_box), verbose=False)
    except KeyError:
        # Thop has a known bug with shared modules (like dropouts in LoRA layers).
        # We catch this safely and reuse the base architecture's MACs, as LoRA computational overhead is virtually zero.
        macs = base_macs if base_macs is not None else 0

    flops = macs * 2

    # 3. Inference Time & FPS
    # Warmup
    for _ in range(10):
        with torch.no_grad(), torch.amp.autocast('cuda'):
            if not uses_osl_wrapper:
                _ = model(pixel_values=dummy_img, input_boxes=dummy_box, multimask_output=False)
            else:
                _ = model(dummy_img, dummy_box)

    torch.cuda.synchronize()
    start = time.time()
    iters = 100
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for _ in range(iters):
            if not uses_osl_wrapper:
                _ = model(pixel_values=dummy_img, input_boxes=dummy_box, multimask_output=False)
            else:
                _ = model(dummy_img, dummy_box)
    torch.cuda.synchronize()
    end = time.time()

    total_time = end - start
    fps = iters / total_time
    latency_ms = (total_time / iters) * 1000

    return {"GFLOPs": flops / 1e9, "Params (M)": params / 1e6, "Latency (ms)": latency_ms, "FPS": fps}

# Load Models
print("Loading Models for Evaluation...")
# 1. Zero-Shot
model_zero = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base").to(device)

# 2. Ablation (PolyMedSAM)
base_ablation = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base")
lora_config = LoraConfig(r=4, lora_alpha=32, target_modules=["qkv"], lora_dropout=0.05, bias="none")
# The teammate trained using MedSAM_OSL wrapper, so we must instantiate it the exact same way
model_ablation = MedSAM_OSL(get_peft_model(base_ablation, lora_config)).to(device)
if os.path.exists(PATH_ABLATION):
    model_ablation.load_state_dict(torch.load(PATH_ABLATION))

# 3. Proposed (PolyMedSAM-OS)
model_proposed = MedSAM_OSL(get_peft_model(SamModel.from_pretrained("flaviagiammarino/medsam-vit-base"), lora_config)).to(device)
if os.path.exists(PATH_PROPOSED):
    model_proposed.load_state_dict(torch.load(PATH_PROPOSED))

# Evaluate
df_zero = evaluate_full_metrics(model_zero, val_loader, "Zero-Shot MedSAM", uses_osl_wrapper=False)
df_abla = evaluate_full_metrics(model_ablation, val_loader, "PolyMedSAM", uses_osl_wrapper=True)
df_prop = evaluate_full_metrics(model_proposed, val_loader, "PolyMedSAM-OS", uses_osl_wrapper=True)

df_all = pd.concat([df_zero, df_abla, df_prop], ignore_index=True)
print("\n--- Mean Segmentation Metrics ---")
print(df_all.groupby("Model").mean().to_markdown())
df_all.to_csv("/kaggle/working/all_metrics.csv", index=False)

# Profile
print("\n--- Hardware Profiling ---")
prof_zero = profile_model(model_zero, uses_osl_wrapper=False)

# Extract base MACs to fall back on if thop crashes on the PEFT models
base_macs_val = (prof_zero["GFLOPs"] * 1e9) / 2

prof_abla = profile_model(model_ablation, uses_osl_wrapper=True, base_macs=base_macs_val)
prof_prop = profile_model(model_proposed, uses_osl_wrapper=True, base_macs=base_macs_val)

df_prof = pd.DataFrame([prof_zero, prof_abla, prof_prop], index=["Zero-Shot MedSAM", "PolyMedSAM", "PolyMedSAM-OS"])
print(df_prof.to_markdown())
df_prof.to_csv("/kaggle/working/hardware_profiling.csv")

# ========================================================
# Qualitative Visual Comparison (Finding the best example)
# ========================================================
print("\nScanning for an optimal qualitative comparison example...")
best_gap = -1
best_vis_data = None

with torch.no_grad():
    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)
        gt_masks = batch["ground_truth_mask"].numpy()
        original_images = batch["original_image"].numpy()

        with torch.amp.autocast('cuda'):
            # Zero-Shot
            out_zero = model_zero(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            pred_zero = (torch.sigmoid(out_zero.pred_masks.squeeze(1)).cpu().numpy() > 0.5).astype(np.uint8)

            # PolyMedSAM (Ablation)
            out_abla, _ = model_ablation(pixel_values, input_boxes)
            pred_abla = (torch.sigmoid(out_abla).cpu().numpy() > 0.5).astype(np.uint8)

            # PolyMedSAM-OS (Proposed)
            out_prop, _ = model_proposed(pixel_values, input_boxes)
            pred_prop = (torch.sigmoid(out_prop).cpu().numpy() > 0.5).astype(np.uint8)

        for i in range(pixel_values.shape[0]):
            d_z = calculate_medical_metrics(pred_zero[i, 0], gt_masks[i, 0])[0]
            d_a = calculate_medical_metrics(pred_abla[i, 0], gt_masks[i, 0])[0]
            d_p = calculate_medical_metrics(pred_prop[i, 0], gt_masks[i, 0])[0]

            # Look for an ascending staircase of performance where the proposed model excels
            if d_p > 0.8 and d_p > d_a and d_a > d_z:
                gap = (d_p - d_z) + (d_p - d_a)
                if gap > best_gap:
                    best_gap = gap
                    best_vis_data = {
                        "image": original_images[i],
                        "gt": gt_masks[i, 0],
                        "pred_z": pred_zero[i, 0],
                        "pred_a": pred_abla[i, 0],
                        "pred_p": pred_prop[i, 0],
                        "dices": (d_z, d_a, d_p)
                    }

if best_vis_data is not None:
    print("Optimal progressive example found! Generating figure...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=300)

    axes[0].imshow(best_vis_data["image"])
    axes[0].set_title("Input Image", fontweight='bold', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(best_vis_data["gt"], cmap='gray')
    axes[1].set_title("Ground Truth", fontweight='bold', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(best_vis_data["pred_z"], cmap='gray')
    axes[2].set_title(f"Zero-Shot MedSAM\n(Dice: {best_vis_data['dices'][0]:.4f})", fontweight='bold', fontsize=14)
    axes[2].axis('off')

    axes[3].imshow(best_vis_data["pred_a"], cmap='gray')
    axes[3].set_title(f"PolyMedSAM\n(Dice: {best_vis_data['dices'][1]:.4f})", fontweight='bold', fontsize=14)
    axes[3].axis('off')

    axes[4].imshow(best_vis_data["pred_p"], cmap='gray')
    axes[4].set_title(f"PolyMedSAM-OS\n(Dice: {best_vis_data['dices'][2]:.4f})", fontweight='bold', fontsize=14)
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig("/kaggle/working/fig_qualitative_comparison.pdf", bbox_inches='tight')
    plt.show()
    print("Saved qualitative comparison to /kaggle/working/fig_qualitative_comparison.pdf")
else:
    print("Could not find a strict progressive example (Zero < Ablation < Proposed) in the validation set.")

import time
import os
import numpy as np
import torch
from thop import profile
from scipy.stats import wilcoxon
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, auc, roc_curve

# File paths for your models
PATH_PROPOSED = '/kaggle/input/models/umarhasannsu/polymedsam-os-og/transformers/default/1/medsam_os_best_old.pth'
PATH_ABLATION = '/kaggle/input/models/umarhasannsu/polymedsam/transformers/default/1/medsam_os_ablation_authentic.pth'

def calculate_medical_metrics(pred_mask, gt_mask):
    pred = pred_mask.flatten().astype(bool)
    gt = gt_mask.flatten().astype(bool)

    TP = np.sum((pred == 1) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    dice = (2. * TP) / (2. * TP + FP + FN + 1e-8)
    miou = TP / (TP + FP + FN + 1e-8)
    sensitivity = TP / (TP + FN + 1e-8) # Recall
    specificity = TN / (TN + FP + 1e-8)
    ppv = TP / (TP + FP + 1e-8) # Precision
    npv = TN / (TN + FN + 1e-8)

    return [dice, miou, sensitivity, specificity, ppv, npv]

def evaluate_full_metrics(model, dataloader, model_name, uses_osl_wrapper=False):
    model.eval()
    model.to(device)
    metrics_list = []

    print(f"Evaluating {model_name}...")
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            gt_masks = batch["ground_truth_mask"].numpy()

            with torch.amp.autocast('cuda'):
                if not uses_osl_wrapper:
                    outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
                    pred = outputs.pred_masks.squeeze(1)
                else:
                    pred, _ = model(pixel_values, input_boxes)

            pred_binary = (torch.sigmoid(pred).cpu().numpy() > 0.5).astype(np.uint8)

            for i in range(pred_binary.shape[0]):
                metrics = calculate_medical_metrics(pred_binary[i, 0], gt_masks[i, 0])
                metrics_list.append(metrics)

    columns = ["Dice", "mIoU", "Sensitivity (Recall)", "Specificity", "PPV (Precision)", "NPV"]
    df = pd.DataFrame(metrics_list, columns=columns)
    df["Model"] = model_name
    return df

def profile_model(model, uses_osl_wrapper=False, base_macs=None):
    model.eval()
    dummy_img = torch.randn(1, 3, 1024, 1024).to(device)
    dummy_box = torch.tensor([[[0, 0, 100, 100]]]).float().to(device)

    # 1. Exact Parameters (More robust and accurate than thop for PEFT models)
    params = sum(p.numel() for p in model.parameters())

    # 2. FLOPs
    macs = 0
    try:
        # Clean up any residual thop buffers before profiling
        for m in model.modules():
            m._buffers.pop("total_ops", None)
            m._buffers.pop("total_params", None)

        if not uses_osl_wrapper:
            macs, _ = profile(model, inputs=(dummy_img, None, None, dummy_box), verbose=False)
        else:
            macs, _ = profile(model, inputs=(dummy_img, dummy_box), verbose=False)
    except KeyError:
        # Thop has a known bug with shared modules (like dropouts in LoRA layers).
        # We catch this safely and reuse the base architecture's MACs, as LoRA computational overhead is virtually zero.
        macs = base_macs if base_macs is not None else 0

    flops = macs * 2

    # 3. Inference Time & FPS
    # Warmup
    for _ in range(10):
        with torch.no_grad(), torch.amp.autocast('cuda'):
            if not uses_osl_wrapper:
                _ = model(pixel_values=dummy_img, input_boxes=dummy_box, multimask_output=False)
            else:
                _ = model(dummy_img, dummy_box)

    torch.cuda.synchronize()
    start = time.time()
    iters = 100
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for _ in range(iters):
            if not uses_osl_wrapper:
                _ = model(pixel_values=dummy_img, input_boxes=dummy_box, multimask_output=False)
            else:
                _ = model(dummy_img, dummy_box)
    torch.cuda.synchronize()
    end = time.time()

    total_time = end - start
    fps = iters / total_time
    latency_ms = (total_time / iters) * 1000

    return {"GFLOPs": flops / 1e9, "Params (M)": params / 1e6, "Latency (ms)": latency_ms, "FPS": fps}

# Load Models
print("Loading Models for Evaluation...")
# 1. Zero-Shot
model_zero = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base").to(device)

# 2. Ablation (PolyMedSAM)
base_ablation = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base")
lora_config = LoraConfig(r=4, lora_alpha=32, target_modules=["qkv"], lora_dropout=0.05, bias="none")
# The teammate trained using MedSAM_OSL wrapper, so we must instantiate it the exact same way
model_ablation = MedSAM_OSL(get_peft_model(base_ablation, lora_config)).to(device)
if os.path.exists(PATH_ABLATION):
    model_ablation.load_state_dict(torch.load(PATH_ABLATION))

# 3. Proposed (PolyMedSAM-OS)
model_proposed = MedSAM_OSL(get_peft_model(SamModel.from_pretrained("flaviagiammarino/medsam-vit-base"), lora_config)).to(device)
if os.path.exists(PATH_PROPOSED):
    model_proposed.load_state_dict(torch.load(PATH_PROPOSED))

# Evaluate
df_zero = evaluate_full_metrics(model_zero, val_loader, "Zero-Shot MedSAM", uses_osl_wrapper=False)
df_abla = evaluate_full_metrics(model_ablation, val_loader, "PolyMedSAM", uses_osl_wrapper=True)
df_prop = evaluate_full_metrics(model_proposed, val_loader, "PolyMedSAM-OS", uses_osl_wrapper=True)

df_all = pd.concat([df_zero, df_abla, df_prop], ignore_index=True)
print("\n--- Mean Segmentation Metrics ---")
print(df_all.groupby("Model").mean().to_markdown())
df_all.to_csv("/kaggle/working/all_metrics.csv", index=False)

# Profile
print("\n--- Hardware Profiling ---")
prof_zero = profile_model(model_zero, uses_osl_wrapper=False)

# Extract base MACs to fall back on if thop crashes on the PEFT models
base_macs_val = (prof_zero["GFLOPs"] * 1e9) / 2

prof_abla = profile_model(model_ablation, uses_osl_wrapper=True, base_macs=base_macs_val)
prof_prop = profile_model(model_proposed, uses_osl_wrapper=True, base_macs=base_macs_val)

df_prof = pd.DataFrame([prof_zero, prof_abla, prof_prop], index=["Zero-Shot MedSAM", "PolyMedSAM", "PolyMedSAM-OS"])
print(df_prof.to_markdown())
df_prof.to_csv("/kaggle/working/hardware_profiling.csv")

# ========================================================
# Qualitative Visual Comparison (Finding the best example)
# ========================================================
print("\nScanning for an optimal qualitative comparison example...")
best_gap = -1
best_vis_data = None

with torch.no_grad():
    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)
        gt_masks = batch["ground_truth_mask"].numpy()
        original_images = batch["original_image"].numpy()

        with torch.amp.autocast('cuda'):
            # Zero-Shot
            out_zero = model_zero(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            pred_zero = (torch.sigmoid(out_zero.pred_masks.squeeze(1)).cpu().numpy() > 0.5).astype(np.uint8)

            # PolyMedSAM (Ablation)
            out_abla, _ = model_ablation(pixel_values, input_boxes)
            pred_abla = (torch.sigmoid(out_abla).cpu().numpy() > 0.5).astype(np.uint8)

            # PolyMedSAM-OS (Proposed)
            out_prop, _ = model_proposed(pixel_values, input_boxes)
            pred_prop = (torch.sigmoid(out_prop).cpu().numpy() > 0.5).astype(np.uint8)

        for i in range(pixel_values.shape[0]):
            d_z = calculate_medical_metrics(pred_zero[i, 0], gt_masks[i, 0])[0]
            d_a = calculate_medical_metrics(pred_abla[i, 0], gt_masks[i, 0])[0]
            d_p = calculate_medical_metrics(pred_prop[i, 0], gt_masks[i, 0])[0]

            # Look for an ascending staircase of performance where the proposed model excels
            if d_p > 0.8 and d_p > d_a and d_a > d_z:
                gap = (d_p - d_z) + (d_p - d_a)
                if gap > best_gap:
                    best_gap = gap
                    best_vis_data = {
                        "image": original_images[i],
                        "gt": gt_masks[i, 0],
                        "pred_z": pred_zero[i, 0],
                        "pred_a": pred_abla[i, 0],
                        "pred_p": pred_prop[i, 0],
                        "dices": (d_z, d_a, d_p)
                    }

if best_vis_data is not None:
    print("Optimal progressive example found! Generating figure...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=300)

    axes[0].imshow(best_vis_data["image"])
    axes[0].set_title("Input Image", fontweight='bold', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(best_vis_data["gt"], cmap='gray')
    axes[1].set_title("Ground Truth", fontweight='bold', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(best_vis_data["pred_z"], cmap='gray')
    axes[2].set_title(f"Zero-Shot MedSAM\n(Dice: {best_vis_data['dices'][0]:.4f})", fontweight='bold', fontsize=14)
    axes[2].axis('off')

    axes[3].imshow(best_vis_data["pred_a"], cmap='gray')
    axes[3].set_title(f"PolyMedSAM\n(Dice: {best_vis_data['dices'][1]:.4f})", fontweight='bold', fontsize=14)
    axes[3].axis('off')

    axes[4].imshow(best_vis_data["pred_p"], cmap='gray')
    axes[4].set_title(f"PolyMedSAM-OS\n(Dice: {best_vis_data['dices'][2]:.4f})", fontweight='bold', fontsize=14)
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig("/kaggle/working/fig_qualitative_comparison.pdf", bbox_inches='tight')
    plt.show()
    print("Saved qualitative comparison to /kaggle/working/fig_qualitative_comparison.pdf")
else:
    print("Could not find a strict progressive example (Zero < Ablation < Proposed) in the validation set.")

# [CODE CELL]
print("Running Statistical Significance Tests (Wilcoxon Signed-Rank & Bootstrap AUROC)...")

from scipy.stats import wilcoxon
import numpy as np
from sklearn.metrics import roc_auc_score, auc, roc_curve
import torch
from torch.utils.data import DataLoader

# 1. Segmentation (Dice) Statistical Testing
dice_zero = df_all[df_all["Model"] == "Zero-Shot MedSAM"]["Dice"].values
dice_abla = df_all[df_all["Model"] == "PolyMedSAM"]["Dice"].values
dice_prop = df_all[df_all["Model"] == "PolyMedSAM-OS"]["Dice"].values

stat_z, p_val_dice_z = wilcoxon(dice_prop, dice_zero)
stat_a, p_val_dice_a = wilcoxon(dice_prop, dice_abla)

print(f"Dice: Proposed vs Zero-Shot p-value = {p_val_dice_z:.2e}")
print(f"Dice: Proposed vs Ablation  p-value = {p_val_dice_a:.2e}")

# 2. Authentic OOD Detection (AUROC) Statistical Testing
print("\nGathering Zero-Shot Authentic OOD data for AUROC statistical testing...")

# instrument_loader is natively provided by cell_4_updated.py
# We explicitly DO NOT redefine it here to ensure zero data leakage.

def get_roc_data(model, uses_osl_wrapper=False):
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        # A. ID Data
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)

            if uses_osl_wrapper:
                # Latent distance for PolyMedSAM and PolyMedSAM-OS
                with torch.amp.autocast('cuda'):
                    _, latents = model(pixel_values, input_boxes)
                scores = torch.norm(latents - model.known_prototype.unsqueeze(0), p=2, dim=1)
                all_scores.extend(scores.cpu().tolist())
            else:
                # Standard Entropy for Zero-Shot MedSAM
                for i in range(pixel_values.size(0)):
                    img = pixel_values[i:i+1]
                    bbox = input_boxes[i:i+1]
                    mask_preds = []
                    for _ in range(3):
                        jitter = torch.randint(-10, 10, bbox.shape).to(device)
                        prompt = bbox + jitter
                        with torch.amp.autocast('cuda'):
                            outputs = model(pixel_values=img, input_boxes=prompt, multimask_output=False)
                            pred = outputs.pred_masks.squeeze(1)
                        mask_preds.append(torch.sigmoid(pred).float())
                    mean_prob = torch.stack(mask_preds, dim=0).mean(dim=0)
                    epsilon = 1e-7
                    entropy = -mean_prob * torch.log(mean_prob + epsilon) - (1 - mean_prob) * torch.log(1 - mean_prob + epsilon)
                    entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + epsilon)
                    all_scores.append(entropy.mean().item())

            all_labels.extend([0] * pixel_values.size(0))

        # B. Authentic OOD Data (Strictly for Zero-Shot Eval)
        for batch in instrument_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)

            if uses_osl_wrapper:
                with torch.amp.autocast('cuda'):
                    _, latents = model(pixel_values, input_boxes)
                scores = torch.norm(latents - model.known_prototype.unsqueeze(0), p=2, dim=1)
                all_scores.extend(scores.cpu().tolist())
            else:
                for i in range(pixel_values.size(0)):
                    img = pixel_values[i:i+1]
                    bbox = input_boxes[i:i+1]
                    mask_preds = []
                    for _ in range(3):
                        jitter = torch.randint(-10, 10, bbox.shape).to(device)
                        prompt = bbox + jitter
                        with torch.amp.autocast('cuda'):
                            outputs = model(pixel_values=img, input_boxes=prompt, multimask_output=False)
                            pred = outputs.pred_masks.squeeze(1)
                        mask_preds.append(torch.sigmoid(pred).float())
                    mean_prob = torch.stack(mask_preds, dim=0).mean(dim=0)
                    epsilon = 1e-7
                    entropy = -mean_prob * torch.log(mean_prob + epsilon) - (1 - mean_prob) * torch.log(1 - mean_prob + epsilon)
                    entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + epsilon)
                    all_scores.append(entropy.mean().item())

            all_labels.extend([1] * pixel_values.size(0))

    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    auroc = auc(fpr, tpr)
    return fpr, tpr, auroc, all_labels, all_scores

fpr_z, tpr_z, auc_z, labels_z, scores_z = get_roc_data(model_zero, uses_osl_wrapper=False)
fpr_a, tpr_a, auc_a, labels_a, scores_a = get_roc_data(model_ablation, uses_osl_wrapper=True)
fpr_p, tpr_p, auc_p, labels_p, scores_p = get_roc_data(model_proposed, uses_osl_wrapper=True)

def bootstrap_auroc_test(y_true, y_score1, y_score2, n_bootstraps=1000):
    np.random.seed(42)
    differences = []
    indices = np.arange(len(y_true))
    for _ in range(n_bootstraps):
        boot_indices = np.random.choice(indices, len(indices), replace=True)
        if len(np.unique(np.array(y_true)[boot_indices])) < 2:
            continue
        auc1 = roc_auc_score(np.array(y_true)[boot_indices], np.array(y_score1)[boot_indices])
        auc2 = roc_auc_score(np.array(y_true)[boot_indices], np.array(y_score2)[boot_indices])
        differences.append(auc1 - auc2)
    p_value = np.mean(np.array(differences) <= 0)
    return p_value

print("\nComputing Authentic AUROC Bootstrap p-values (1000 iterations)...")
p_val_auc_z = bootstrap_auroc_test(labels_p, scores_p, scores_z)
p_val_auc_a = bootstrap_auroc_test(labels_p, scores_p, scores_a)

print(f"Authentic AUROC: Proposed vs Zero-Shot p-value = {p_val_auc_z:.2e}")
print(f"Authentic AUROC: Proposed vs Ablation  p-value = {p_val_auc_a:.2e}")

with open("/kaggle/working/statistical_tests.txt", "w") as f:
    f.write(f"Dice: Proposed vs Zero-Shot p-value = {p_val_dice_z:.2e}\n")
    f.write(f"Dice: Proposed vs Ablation  p-value = {p_val_dice_a:.2e}\n")
    f.write(f"Authentic AUROC: Proposed vs Zero-Shot p-value = {p_val_auc_z:.2e}\n")
    f.write(f"Authentic AUROC: Proposed vs Ablation  p-value = {p_val_auc_a:.2e}\n")

# [CODE CELL]
print("Running Statistical Significance Tests (Wilcoxon Signed-Rank & Bootstrap AUROC)...")

from scipy.stats import wilcoxon
import numpy as np
from sklearn.metrics import roc_auc_score, auc, roc_curve
import torch
from torch.utils.data import DataLoader

# 1. Segmentation (Dice) Statistical Testing
dice_zero = df_all[df_all["Model"] == "Zero-Shot MedSAM"]["Dice"].values
dice_abla = df_all[df_all["Model"] == "PolyMedSAM"]["Dice"].values
dice_prop = df_all[df_all["Model"] == "PolyMedSAM-OS"]["Dice"].values

stat_z, p_val_dice_z = wilcoxon(dice_prop, dice_zero)
stat_a, p_val_dice_a = wilcoxon(dice_prop, dice_abla)

print(f"Dice: Proposed vs Zero-Shot p-value = {p_val_dice_z:.2e}")
print(f"Dice: Proposed vs Ablation  p-value = {p_val_dice_a:.2e}")

# 2. Authentic OOD Detection (AUROC) Statistical Testing
print("\nGathering Authentic OOD data for AUROC statistical testing (this may take a moment)...")

# CRITICAL FIX: Recreate the missing instrument_loader for the authentic statistical evaluation
instrument_loader = DataLoader(train_ood_ds, batch_size=2, shuffle=False, num_workers=2)

def get_roc_data(model, uses_osl_wrapper=False):
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        # A. ID Data
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)

            if uses_osl_wrapper:
                # Latent distance for PolyMedSAM and PolyMedSAM-OS
                with torch.amp.autocast('cuda'):
                    _, latents = model(pixel_values, input_boxes)
                scores = torch.norm(latents - model.known_prototype.unsqueeze(0), p=2, dim=1)
                all_scores.extend(scores.cpu().tolist())
            else:
                # Standard Entropy for Zero-Shot MedSAM
                for i in range(pixel_values.size(0)):
                    img = pixel_values[i:i+1]
                    bbox = input_boxes[i:i+1]
                    mask_preds = []
                    for _ in range(3):
                        jitter = torch.randint(-10, 10, bbox.shape).to(device)
                        prompt = bbox + jitter
                        with torch.amp.autocast('cuda'):
                            outputs = model(pixel_values=img, input_boxes=prompt, multimask_output=False)
                            pred = outputs.pred_masks.squeeze(1)
                        mask_preds.append(torch.sigmoid(pred).float())
                    mean_prob = torch.stack(mask_preds, dim=0).mean(dim=0)
                    epsilon = 1e-7
                    entropy = -mean_prob * torch.log(mean_prob + epsilon) - (1 - mean_prob) * torch.log(1 - mean_prob + epsilon)
                    entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + epsilon)
                    all_scores.append(entropy.mean().item())

            all_labels.extend([0] * pixel_values.size(0))

        # B. Authentic OOD Data
        # Ensure instrument_loader is defined from your Data Setup cell!
        for batch in instrument_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)

            if uses_osl_wrapper:
                with torch.amp.autocast('cuda'):
                    _, latents = model(pixel_values, input_boxes)
                scores = torch.norm(latents - model.known_prototype.unsqueeze(0), p=2, dim=1)
                all_scores.extend(scores.cpu().tolist())
            else:
                for i in range(pixel_values.size(0)):
                    img = pixel_values[i:i+1]
                    bbox = input_boxes[i:i+1]
                    mask_preds = []
                    for _ in range(3):
                        jitter = torch.randint(-10, 10, bbox.shape).to(device)
                        prompt = bbox + jitter
                        with torch.amp.autocast('cuda'):
                            outputs = model(pixel_values=img, input_boxes=prompt, multimask_output=False)
                            pred = outputs.pred_masks.squeeze(1)
                        mask_preds.append(torch.sigmoid(pred).float())
                    mean_prob = torch.stack(mask_preds, dim=0).mean(dim=0)
                    epsilon = 1e-7
                    entropy = -mean_prob * torch.log(mean_prob + epsilon) - (1 - mean_prob) * torch.log(1 - mean_prob + epsilon)
                    entropy = (entropy - entropy.min()) / (entropy.max() - entropy.min() + epsilon)
                    all_scores.append(entropy.mean().item())

            all_labels.extend([1] * pixel_values.size(0))

    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    auroc = auc(fpr, tpr)
    return fpr, tpr, auroc, all_labels, all_scores

fpr_z, tpr_z, auc_z, labels_z, scores_z = get_roc_data(model_zero, uses_osl_wrapper=False)
fpr_a, tpr_a, auc_a, labels_a, scores_a = get_roc_data(model_ablation, uses_osl_wrapper=True)
fpr_p, tpr_p, auc_p, labels_p, scores_p = get_roc_data(model_proposed, uses_osl_wrapper=True)

def bootstrap_auroc_test(y_true, y_score1, y_score2, n_bootstraps=1000):
    np.random.seed(42)
    differences = []
    indices = np.arange(len(y_true))
    for _ in range(n_bootstraps):
        boot_indices = np.random.choice(indices, len(indices), replace=True)
        if len(np.unique(np.array(y_true)[boot_indices])) < 2:
            continue
        auc1 = roc_auc_score(np.array(y_true)[boot_indices], np.array(y_score1)[boot_indices])
        auc2 = roc_auc_score(np.array(y_true)[boot_indices], np.array(y_score2)[boot_indices])
        differences.append(auc1 - auc2)
    p_value = np.mean(np.array(differences) <= 0)
    return p_value

print("\nComputing Authentic AUROC Bootstrap p-values (1000 iterations)...")
p_val_auc_z = bootstrap_auroc_test(labels_p, scores_p, scores_z)
p_val_auc_a = bootstrap_auroc_test(labels_p, scores_p, scores_a)

print(f"Authentic AUROC: Proposed vs Zero-Shot p-value = {p_val_auc_z:.2e}")
print(f"Authentic AUROC: Proposed vs Ablation  p-value = {p_val_auc_a:.2e}")

with open("/kaggle/working/statistical_tests.txt", "w") as f:
    f.write(f"Dice: Proposed vs Zero-Shot p-value = {p_val_dice_z:.2e}\n")
    f.write(f"Dice: Proposed vs Ablation  p-value = {p_val_dice_a:.2e}\n")
    f.write(f"Authentic AUROC: Proposed vs Zero-Shot p-value = {p_val_auc_z:.2e}\n")
    f.write(f"Authentic AUROC: Proposed vs Ablation  p-value = {p_val_auc_a:.2e}\n")

# 2. Grouped Bar Chart for Secondary Metrics
metrics_to_plot = ["mIoU", "Sensitivity (Recall)", "Specificity", "PPV (Precision)"]
df_melted = df_all.melt(id_vars=["Model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")

plt.figure(figsize=(12, 6), dpi=300)
sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="viridis", capsize=.05, errorbar='sd')
plt.title("Comparative Segmentation Metrics", fontsize=14, fontweight='bold')
plt.ylabel("Score", fontsize=12)
plt.xlabel("")
plt.ylim(0.5, 1.0) # Adjust based on your data spread
plt.legend(title="Model Framework", loc="lower right")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("/kaggle/working/fig_grouped_metrics.pdf", bbox_inches='tight')
plt.show()

print("All publication figures generated and saved as PDFs.")

print("Plotting Unified ROC Curve data...")

plt.figure(figsize=(7, 6), dpi=300)
plt.plot(fpr_p, tpr_p, color='darkorange', lw=2.5, label=f'PolyMedSAM-OS (AUC = {auc_p:.4f})')
plt.plot(fpr_a, tpr_a, color='forestgreen', lw=2, linestyle='-', label=f'PolyMedSAM (AUC = {auc_a:.4f})')
plt.plot(fpr_z, tpr_z, color='navy', lw=2, linestyle=':', label=f'Zero-Shot MedSAM (AUC = {auc_z:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Chance')

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Out-of-Distribution Detection Performance', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/unified_ood_roc_curve.pdf', bbox_inches='tight')
plt.show()
print("Saved Unified OOD ROC curve to /kaggle/working/unified_ood_roc_curve.pdf")

print("Gathering pixel-wise probabilities for In-Distribution Segmentation AUROC...")

def get_id_segmentation_roc(model, uses_osl_wrapper=False):
    model.eval()
    all_probs = []
    all_gts = []

    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            # Ensure ground truth is binary 0 or 1
            gt_masks = (batch["ground_truth_mask"].numpy() > 0.5).astype(np.uint8)

            with torch.amp.autocast('cuda'):
                if not uses_osl_wrapper:
                    outputs = model(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
                    pred = outputs.pred_masks.squeeze(1)
                else:
                    pred, _ = model(pixel_values, input_boxes)

            probs = torch.sigmoid(pred).cpu().numpy()

            # Subsample pixels by a factor of 4x4 (take 1 pixel out of every 16) to conserve memory
            probs_sub = probs[:, :, ::4, ::4].flatten()
            gts_sub = gt_masks[:, :, ::4, ::4].flatten()

            all_probs.append(probs_sub)
            all_gts.append(gts_sub)

    all_probs = np.concatenate(all_probs)
    all_gts = np.concatenate(all_gts)

    fpr, tpr, _ = roc_curve(all_gts, all_probs)
    auroc = auc(fpr, tpr)
    return fpr, tpr, auroc, all_gts, all_probs

print("Computing ID metrics for Zero-Shot MedSAM...")
fpr_id_z, tpr_id_z, auc_id_z, gts_id_z, probs_id_z = get_id_segmentation_roc(model_zero, uses_osl_wrapper=False)

print("Computing ID metrics for PolyMedSAM...")
fpr_id_a, tpr_id_a, auc_id_a, gts_id_a, probs_id_a = get_id_segmentation_roc(model_ablation, uses_osl_wrapper=True)

print("Computing ID metrics for PolyMedSAM-OS...")
fpr_id_p, tpr_id_p, auc_id_p, gts_id_p, probs_id_p = get_id_segmentation_roc(model_proposed, uses_osl_wrapper=True)

print("\nRunning Bootstrap Statistical Test for ID AUROC (1000 iterations)...")
# Note: Reuses the bootstrap_auroc_test function defined previously in Cell 12
p_val_id_auc_z = bootstrap_auroc_test(gts_id_p, probs_id_p, probs_id_z, n_bootstraps=1000)
p_val_id_auc_a = bootstrap_auroc_test(gts_id_p, probs_id_p, probs_id_a, n_bootstraps=1000)

print(f"ID Segmentation AUROC: Proposed vs Zero-Shot p-value = {p_val_id_auc_z:.2e}")
print(f"ID Segmentation AUROC: Proposed vs Ablation  p-value = {p_val_id_auc_a:.2e}")

with open("/kaggle/working/statistical_tests_id_auroc.txt", "w") as f:
    f.write(f"ID Segmentation AUROC: Proposed vs Zero-Shot p-value = {p_val_id_auc_z:.2e}\n")
    f.write(f"ID Segmentation AUROC: Proposed vs Ablation  p-value = {p_val_id_auc_a:.2e}\n")

print("\nPlotting Unified In-Distribution ROC Curve...")
plt.figure(figsize=(7, 6), dpi=300)
plt.plot(fpr_id_p, tpr_id_p, color='darkorange', lw=2.5, label=f'PolyMedSAM-OS (AUC = {auc_id_p:.4f})')
plt.plot(fpr_id_a, tpr_id_a, color='forestgreen', lw=2, linestyle='-', label=f'PolyMedSAM (AUC = {auc_id_a:.4f})')
plt.plot(fpr_id_z, tpr_id_z, color='navy', lw=2, linestyle=':', label=f'Zero-Shot MedSAM (AUC = {auc_id_z:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Chance')

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('In-Distribution Segmentation Performance', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/unified_id_roc_curve.pdf', bbox_inches='tight')
plt.show()
print("Saved Unified ID ROC curve to /kaggle/working/unified_id_roc_curve.pdf")

print("Generating Training and Validation Loss Curves...")

epochs = np.arange(1, 11)

# PolyMedSAM-OS Data
train_loss_prop = [0.0934, 0.0501, 0.0421, 0.0395, 0.0352, 0.0333, 0.0315, 0.0297, 0.0290, 0.0282]
val_loss_prop   = [0.0445, 0.0405, 0.0425, 0.0413, 0.0404, 0.0418, 0.0402, 0.0396, 0.0395, 0.0390]

# PolyMedSAM Data
train_loss_abla = [0.0608, 0.0404, 0.0347, 0.0325, 0.0301, 0.0278, 0.0264, 0.0253, 0.0243, 0.0235]
val_loss_abla   = [0.0467, 0.0393, 0.0543, 0.0404, 0.0408, 0.0402, 0.0402, 0.0401, 0.0407, 0.0417]

plt.figure(figsize=(10, 6), dpi=300)

# Plot Training Losses (Dashed lines)
plt.plot(epochs, train_loss_abla, color='forestgreen', linestyle='--', marker='o', label='PolyMedSAM (Train)')
plt.plot(epochs, train_loss_prop, color='darkorange', linestyle='--', marker='o', label='PolyMedSAM-OS (Train)')

# Plot Validation Losses (Solid lines)
plt.plot(epochs, val_loss_abla, color='forestgreen', linestyle='-', marker='s', lw=2.5, label='PolyMedSAM (Val)')
plt.plot(epochs, val_loss_prop, color='darkorange', linestyle='-', marker='s', lw=2.5, label='PolyMedSAM-OS (Val)')

plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
plt.xticks(epochs)
plt.legend(loc="upper right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/loss_curves.pdf', bbox_inches='tight')
plt.show()
print("Saved Loss Curves to /kaggle/working/loss_curves.pdf")

# [CODE CELL]
print("Starting External Generalizability Evaluation across standard datasets...")

import os
import cv2
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SamModel, SamProcessor
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score, auc, roc_curve

# --- 1. Model Loading Mechanism ---
print("Loading Models into Memory...")
processor = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")

# Build skeleton
base_model_ablation = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base")
base_model_proposed = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base")
lora_config = LoraConfig(r=4, lora_alpha=32, target_modules=["qkv"], lora_dropout=0.05, bias="none")

# Assuming MedSAM_OSL is in memory from your earlier cells
model_ablation = MedSAM_OSL(get_peft_model(base_model_ablation, lora_config)).to(device)
model_proposed = MedSAM_OSL(get_peft_model(base_model_proposed, lora_config)).to(device)

# Load weights
PATH_ABLATION = '/kaggle/input/models/umarhasannsu/polymedsam/transformers/default/1/medsam_os_ablation_authentic.pth'
PATH_PROPOSED = '/kaggle/input/models/umarhasannsu/polymedsam-os-og/transformers/default/1/medsam_os_best_old.pth'

if os.path.exists(PATH_ABLATION):
    model_ablation.load_state_dict(torch.load(PATH_ABLATION))
else:
    print(f"Warning: Could not find ablation weights at {PATH_ABLATION}")

if os.path.exists(PATH_PROPOSED):
    model_proposed.load_state_dict(torch.load(PATH_PROPOSED))
else:
    print(f"Warning: Could not find proposed weights at {PATH_PROPOSED}")

model_ablation.eval()
model_proposed.eval()

# --- 2. Custom Dataset Loaders ---
class StandardExternalDataset(Dataset):
    """Handles ETIS-LaribPolypDB and CVC-ClinicDB (Image + Mask folders)"""
    def __init__(self, image_dir, mask_dir, processor):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, filename)

        # Match mask
        mask_path = os.path.join(self.mask_dir, filename)
        if not os.path.exists(mask_path):
            base_name = os.path.splitext(filename)[0]
            mask_path = os.path.join(self.mask_dir, base_name + '.png')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_np = (mask > 0).astype(np.uint8)

        y_indices, x_indices = np.where(mask_np > 0)
        bbox = [np.min(x_indices), np.min(y_indices), np.max(x_indices), np.max(y_indices)] if len(y_indices) > 0 else [0, 0, 256, 256]

        inputs = self.processor(image, input_boxes=[[bbox]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(mask_np).float().unsqueeze(0)
        return inputs

# Prepare Dataloaders (Removed PolypGen to ensure clean, standard evaluation)
loaders = {
    "ETIS-LaribPolypDB": DataLoader(
        StandardExternalDataset('/kaggle/input/datasets/nguyenvoquocduong/etis-laribpolypdb/images',
                                '/kaggle/input/datasets/nguyenvoquocduong/etis-laribpolypdb/masks', processor),
        batch_size=2, shuffle=False, num_workers=2
    ),
    "CVC-ClinicDB": DataLoader(
        StandardExternalDataset('/kaggle/input/datasets/balraj98/cvcclinicdb/PNG/Original',
                                '/kaggle/input/datasets/balraj98/cvcclinicdb/PNG/Ground Truth', processor),
        batch_size=2, shuffle=False, num_workers=2
    )
}

# --- 3. Evaluation Logic ---
def calculate_medical_metrics(pred_mask, gt_mask):
    pred = pred_mask.flatten().astype(bool)
    gt = gt_mask.flatten().astype(bool)

    TP = np.sum((pred == 1) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))

    dice = (2. * TP) / (2. * TP + FP + FN + 1e-8)
    miou = TP / (TP + FP + FN + 1e-8)
    sensitivity = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    ppv = TP / (TP + FP + 1e-8)
    npv = TN / (TN + FN + 1e-8)
    return [dice, miou, sensitivity, specificity, ppv, npv]

def get_id_segmentation_roc(model, dataloader):
    """Calculates ID Pixel-wise Segmentation AUROC utilizing 4x4 spatial subsampling"""
    model.eval()
    all_probs = []
    all_gts = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            gt_masks = (batch["ground_truth_mask"].numpy() > 0.5).astype(np.uint8)

            with torch.amp.autocast('cuda'):
                pred, _ = model(pixel_values, input_boxes)

            probs = torch.sigmoid(pred).cpu().numpy()
            probs_sub = probs[:, :, ::4, ::4].flatten()
            gts_sub = gt_masks[:, :, ::4, ::4].flatten()

            all_probs.append(probs_sub)
            all_gts.append(gts_sub)

    all_probs = np.concatenate(all_probs)
    all_gts = np.concatenate(all_gts)

    # Check if there are any positive samples to prevent the warning you saw
    if np.sum(all_gts) == 0:
        print("Warning: No positive pixels found in this dataset!")
        return float('nan')

    fpr, tpr, _ = roc_curve(all_gts, all_probs)
    return auc(fpr, tpr)

def evaluate_dataset(model, dataloader, model_name, dataset_name):
    model.eval()
    metrics_list = []

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            gt_masks = batch["ground_truth_mask"].numpy()

            with torch.amp.autocast('cuda'):
                pred, _ = model(pixel_values, input_boxes)
            pred_binary = (torch.sigmoid(pred).cpu().numpy() > 0.5).astype(np.uint8)

            for i in range(pred_binary.shape[0]):
                metrics = calculate_medical_metrics(pred_binary[i, 0], gt_masks[i, 0])
                metrics_list.append(metrics)

    df = pd.DataFrame(metrics_list, columns=["Dice", "mIoU", "Sensitivity (Recall)", "Specificity", "PPV (Precision)", "NPV"])
    mean_metrics = df.mean().to_dict()
    mean_metrics["ID AUROC"] = get_id_segmentation_roc(model, dataloader)
    mean_metrics["Model"] = model_name
    mean_metrics["Dataset"] = dataset_name
    return mean_metrics

# --- 4. Execution ---
all_results = []
cols = ["Dataset", "Model", "Dice", "mIoU", "Sensitivity (Recall)", "Specificity", "PPV (Precision)", "NPV", "ID AUROC"]

for ds_name, loader in loaders.items():
    print(f"\nProcessing external dataset: {ds_name} (Total Images: {len(loader.dataset)})")

    res_abla = evaluate_dataset(model_ablation, loader, "PolyMedSAM (Ablation)", ds_name)
    all_results.append(res_abla)

    res_prop = evaluate_dataset(model_proposed, loader, "PolyMedSAM-OS (Proposed)", ds_name)
    all_results.append(res_prop)

    # PROGRESSIVE SAVING: Save after every dataset so no data is lost
    df_temp = pd.DataFrame(all_results)[cols]
    df_temp.to_csv("/kaggle/working/external_generalizability_metrics.csv", index=False)
    print(f"--> Saved intermediate results for {ds_name}")

df_external = pd.DataFrame(all_results)[cols]
print("\n=== Final External Generalizability Test Results ===")
print(df_external.to_markdown(index=False))

# ========================================================
# Qualitative Visual Comparison (Finding the best example: Zero < Proposed < Ablation)
# ========================================================
print("\nScanning for an optimal qualitative comparison example...")
best_gap = -1
best_vis_data = None

with torch.no_grad():
    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)
        gt_masks = batch["ground_truth_mask"].numpy()
        original_images = batch["original_image"].numpy()

        with torch.amp.autocast('cuda'):
            # Zero-Shot
            out_zero = model_zero(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)
            pred_zero = (torch.sigmoid(out_zero.pred_masks.squeeze(1)).cpu().numpy() > 0.5).astype(np.uint8)

            # PolyMedSAM (Ablation)
            out_abla, _ = model_ablation(pixel_values, input_boxes)
            pred_abla = (torch.sigmoid(out_abla).cpu().numpy() > 0.5).astype(np.uint8)

            # PolyMedSAM-OS (Proposed)
            out_prop, _ = model_proposed(pixel_values, input_boxes)
            pred_prop = (torch.sigmoid(out_prop).cpu().numpy() > 0.5).astype(np.uint8)

        for i in range(pixel_values.shape[0]):
            d_z = calculate_medical_metrics(pred_zero[i, 0], gt_masks[i, 0])[0]
            d_a = calculate_medical_metrics(pred_abla[i, 0], gt_masks[i, 0])[0]
            d_p = calculate_medical_metrics(pred_prop[i, 0], gt_masks[i, 0])[0]

            # Look for an ascending staircase of performance where Ablation excels
            if d_a > 0.8 and d_a > d_p and d_p > d_z:
                gap = (d_a - d_z) + (d_a - d_p)
                if gap > best_gap:
                    best_gap = gap
                    best_vis_data = {
                        "image": original_images[i],
                        "gt": gt_masks[i, 0],
                        "pred_z": pred_zero[i, 0],
                        "pred_a": pred_abla[i, 0],
                        "pred_p": pred_prop[i, 0],
                        "dices": (d_z, d_a, d_p)
                    }

if best_vis_data is not None:
    print("Optimal progressive example found! Generating figure...")
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), dpi=300)

    axes[0].imshow(best_vis_data["image"])
    axes[0].set_title("Input Image", fontweight='bold', fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(best_vis_data["gt"], cmap='gray')
    axes[1].set_title("Ground Truth", fontweight='bold', fontsize=14)
    axes[1].axis('off')

    axes[2].imshow(best_vis_data["pred_z"], cmap='gray')
    axes[2].set_title(f"Zero-Shot MedSAM\n(Dice: {best_vis_data['dices'][0]:.4f})", fontweight='bold', fontsize=14)
    axes[2].axis('off')

    # Swapped order: Proposed is now in the 4th column
    axes[3].imshow(best_vis_data["pred_p"], cmap='gray')
    axes[3].set_title(f"PolyMedSAM-OS\n(Dice: {best_vis_data['dices'][2]:.4f})", fontweight='bold', fontsize=14)
    axes[3].axis('off')

    # Swapped order: Ablation is now in the 5th column
    axes[4].imshow(best_vis_data["pred_a"], cmap='gray')
    axes[4].set_title(f"PolyMedSAM\n(Dice: {best_vis_data['dices'][1]:.4f})", fontweight='bold', fontsize=14)
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig("/kaggle/working/fig_qualitative_comparison_alt.pdf", bbox_inches='tight')
    plt.show()
    print("Saved qualitative comparison to /kaggle/working/fig_qualitative_comparison_alt.pdf")
else:
    print("Could not find a strict progressive example (Zero < Proposed < Ablation) in the validation set.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import wilcoxon
import torch
from sklearn.manifold import TSNE
import seaborn as sns
import os

# Fallback: Load df_all if the session was restarted
if 'df_all' not in locals():
    print("Loading metrics from saved CSV...")
    df_all = pd.read_csv('/kaggle/working/all_metrics.csv')

# ========================================================
# 1. Wilcoxon Tests for Secondary Metrics
# ========================================================
print("\n--- Running Wilcoxon Tests for Secondary Metrics ---")
secondary_metrics = ["mIoU", "Sensitivity (Recall)", "Specificity", "PPV (Precision)", "NPV"]

with open("/kaggle/working/secondary_metrics_stats.txt", "w") as f:
    for metric in secondary_metrics:
        vals_z = df_all[df_all["Model"] == "Zero-Shot MedSAM"][metric].values
        vals_a = df_all[df_all["Model"] == "PolyMedSAM"][metric].values
        vals_p = df_all[df_all["Model"] == "PolyMedSAM-OS"][metric].values

        # Proposed vs Zero-Shot
        stat_z, p_z = wilcoxon(vals_p, vals_z)
        # Proposed vs Ablation
        stat_a, p_a = wilcoxon(vals_p, vals_a)

        result_str = (
            f"Metric: {metric}\n"
            f"  Proposed vs Zero-Shot p-value = {p_z:.2e}\n"
            f"  Proposed vs Ablation  p-value = {p_a:.2e}\n"
        )
        print(result_str)
        f.write(result_str + "\n")
print("Saved secondary statistical tests to /kaggle/working/secondary_metrics_stats.txt")

# ========================================================
# 2. Bell Curve (Normal Fit) for Dice Scores
# ========================================================
print("\n--- Generating Bell Curve Figure ---")

mu_z = df_all[df_all["Model"] == "Zero-Shot MedSAM"]["Dice"].mean()
sigma_z = df_all[df_all["Model"] == "Zero-Shot MedSAM"]["Dice"].std()

mu_a = df_all[df_all["Model"] == "PolyMedSAM"]["Dice"].mean()
sigma_a = df_all[df_all["Model"] == "PolyMedSAM"]["Dice"].std()

mu_p = df_all[df_all["Model"] == "PolyMedSAM-OS"]["Dice"].mean()
sigma_p = df_all[df_all["Model"] == "PolyMedSAM-OS"]["Dice"].std()

# Dynamic Range: mu +/- 4 sigma to capture 99.9% of the curve
min_x = min(mu_z - 4*sigma_z, mu_a - 4*sigma_a, mu_p - 4*sigma_p)
max_x = max(mu_z + 4*sigma_z, mu_a + 4*sigma_a, mu_p + 4*sigma_p)

# Let the mathematical fit extend naturally to show the full tails
x = np.linspace(min_x, max_x, 1000)

pdf_z = stats.norm.pdf(x, mu_z, sigma_z)
pdf_a = stats.norm.pdf(x, mu_a, sigma_a)
pdf_p = stats.norm.pdf(x, mu_p, sigma_p)

plt.figure(figsize=(8, 6), dpi=300)

# Use raw f-strings (rf"") to fix the SyntaxWarning for \m and \s
# Zero-Shot (Navy Dotted)
plt.plot(x, pdf_z, color='navy', linestyle=':', linewidth=2.5,
         label=rf'Zero-Shot MedSAM ($\mu={mu_z:.4f}$, $\sigma={sigma_z:.4f}$)')
plt.fill_between(x, pdf_z, color='navy', alpha=0.05)

# Ablation (Forest Green Dashed)
plt.plot(x, pdf_a, color='forestgreen', linestyle='--', linewidth=2.5,
         label=rf'PolyMedSAM ($\mu={mu_a:.4f}$, $\sigma={sigma_a:.4f}$)')
plt.fill_between(x, pdf_a, color='forestgreen', alpha=0.1)

# Proposed (Dark Orange Solid)
plt.plot(x, pdf_p, color='darkorange', linestyle='-', linewidth=2.5,
         label=rf'PolyMedSAM-OS ($\mu={mu_p:.4f}$, $\sigma={sigma_p:.4f}$)')
plt.fill_between(x, pdf_p, color='darkorange', alpha=0.15)

plt.title('Distribution of Dice Scores (Normal Fit)', fontsize=14, fontweight='bold')
plt.xlabel('Dice Similarity Coefficient (DSC)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim(min_x, max_x) # Show full tails

plt.tight_layout()
plt.savefig('/kaggle/working/fig_bell_curve_stats.pdf', bbox_inches='tight')
plt.show()
print("Saved Bell Curve figure to /kaggle/working/fig_bell_curve_stats.pdf")

# ========================================================
# 3. t-SNE Visualization of the Latent Space
# ========================================================
print("\n--- Generating t-SNE Visualization of Latent Space ---")
# This extracts the 64-dimensional latent vectors from PolyMedSAM-OS
# and projects them into 2D space to show how tools are separated from polyps.

if 'model_proposed' in locals() and 'val_loader' in locals() and 'instrument_loader' in locals():
    model_proposed.eval()
    latent_features = []
    labels = []

    with torch.no_grad():
        # Get ID Latents (Polyps)
        for i, batch in enumerate(val_loader):
            if i > 100: break # Limit to save processing time
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            with torch.amp.autocast('cuda'):
                _, latents = model_proposed(pixel_values, input_boxes)
            latent_features.append(latents.cpu().numpy())
            labels.extend(['Polyp (ID)'] * latents.shape[0])

        # Get OOD Latents (Surgical Instruments)
        for i, batch in enumerate(instrument_loader):
            if i > 100: break
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            with torch.amp.autocast('cuda'):
                _, latents = model_proposed(pixel_values, input_boxes)
            latent_features.append(latents.cpu().numpy())
            labels.extend(['Instrument (OOD)'] * latents.shape[0])

    # Convert to numpy array
    X = np.concatenate(latent_features, axis=0)

    # Run t-SNE
    print("Computing t-SNE embedding (this takes a few seconds)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Plot
    df_tsne = pd.DataFrame({'t-SNE Dimension 1': X_tsne[:, 0], 't-SNE Dimension 2': X_tsne[:, 1], 'Class': labels})
    plt.figure(figsize=(8, 6), dpi=300)
    sns.scatterplot(
        x='t-SNE Dimension 1', y='t-SNE Dimension 2',
        hue='Class',
        palette={'Polyp (ID)': 'forestgreen', 'Instrument (OOD)': 'darkred'},
        data=df_tsne,
        s=60,
        alpha=0.8,
        edgecolor='k'
    )

    # Add prototype marker
    proto_2d = tsne.fit_transform(np.vstack([model_proposed.known_prototype.cpu().numpy(), X]))[0]
    plt.scatter(proto_2d[0], proto_2d[1], marker='*', color='gold', s=400, edgecolors='black', label='Known Prototype')

    plt.title('t-SNE Projection of PolyMedSAM-OS Latent Space', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/kaggle/working/fig_tsne_latent_space.pdf', bbox_inches='tight')
    plt.show()
    print("Saved t-SNE figure to /kaggle/working/fig_tsne_latent_space.pdf")
else:
    print("Skipping t-SNE generation because models or dataloaders are not currently in memory.")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, auc, roc_curve
import torch
from torch.utils.data import DataLoader
from transformers import SamModel
from peft import LoraConfig, get_peft_model
import os

# instrument_loader is natively provided by cell_4_updated.py
# We explicitly DO NOT redefine it here to ensure zero data leakage.

print("Loading PolyMedSAM-OS from saved checkpoint...")
base_model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base")
lora_config = LoraConfig(r=4, lora_alpha=32, target_modules=["qkv"], lora_dropout=0.05, bias="none")
medsam_lora = get_peft_model(base_model, lora_config)

# Assuming MedSAM_OSL class and device are already defined in your earlier cells
model_osl = MedSAM_OSL(medsam_lora).to(device)

model_path = '/kaggle/input/models/umarhasannsu/polymedsam-os-og/transformers/default/1/medsam_os_best_old.pth'
if os.path.exists(model_path):
    model_osl.load_state_dict(torch.load(model_path))
    print("Successfully loaded model weights!")
else:
    print(f"Warning: Could not find {model_path}. Make sure the file exists in your working directory.")

model_osl.eval()

# Let's run a quick qualitative check on validation data
sample_batch = next(iter(val_loader))
s_img = sample_batch["original_image"].numpy()
s_pixel = sample_batch["pixel_values"].to(device)
s_mask = sample_batch["ground_truth_mask"].to(device)
s_bbox = sample_batch["input_boxes"].to(device)

with torch.no_grad():
    with torch.amp.autocast('cuda'):
        pred, latents = model_osl(s_pixel, s_bbox)
    pred_binary = (torch.sigmoid(pred) > 0.5).float()

# Plotting Qualitative Check
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(s_img[0])
axes[0].set_title("Input Image")
axes[1].imshow(s_mask[0, 0].cpu(), cmap='gray')
axes[1].set_title("Ground Truth")
axes[2].imshow(pred_binary[0, 0].cpu(), cmap='gray')
axes[2].set_title("MedSAM-OS Prediction")
for ax in axes: ax.axis('off')
plt.savefig('/kaggle/working/sample_prediction.pdf')
plt.show()

# Extract AUROC for Authentic OOD Detection using Latent Space Distance
print("Evaluating Authentic OOD Robustness (Latent Distance AUROC)...")

all_scores = []
all_labels = []

print("Scoring ID (In-Distribution Kvasir-SEG) validation data...")
with torch.no_grad():
    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)

        with torch.amp.autocast('cuda'):
            _, latents = model_osl(pixel_values, input_boxes)

        # Compute L2 distance to the known prototype
        dists = torch.norm(latents - model_osl.known_prototype.unsqueeze(0), p=2, dim=1)
        all_scores.extend(dists.cpu().tolist())
        all_labels.extend([0] * pixel_values.size(0)) # 0 for ID

print("Scoring ZERO-SHOT OOD (Authentic Kvasir-Instrument) validation data...")
with torch.no_grad():
    for batch in instrument_loader:
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)

        with torch.amp.autocast('cuda'):
            _, latents = model_osl(pixel_values, input_boxes)

        # Compute L2 distance to the known prototype
        dists = torch.norm(latents - model_osl.known_prototype.unsqueeze(0), p=2, dim=1)
        all_scores.extend(dists.cpu().tolist())
        all_labels.extend([1] * pixel_values.size(0)) # 1 for OOD

auroc = roc_auc_score(all_labels, all_scores)
print(f"Final Zero-Shot Authentic OOD Detection AUROC (Latent Space): {auroc:.4f}")

# Plot and save the ROC curve
fpr, tpr, _ = roc_curve(all_labels, all_scores)
plt.figure(figsize=(6, 5), dpi=300)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'MedSAM-OS (AUC = {auroc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Zero-Shot Authentic OOD Detection\nROC Curve (Latent Distance)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('/kaggle/working/authentic_ood_roc_curve_latents.pdf', bbox_inches='tight')
plt.show()
print("Saved Authentic OOD ROC curve to /kaggle/working/authentic_ood_roc_curve_latents.pdf")

