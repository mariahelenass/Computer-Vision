# Wildfire Image Classification with Pretrained ResNet18 Models

This project focuses on the classification of Sentinel-2 satellite images into two categories: `wildfire` and `nowildfire`. We explore different pretraining strategies for ResNet18 models using PyTorch and TorchGeo.

---

## Project Overview

The primary goal is to detect the presence of wildfires in satellite imagery using deep learning techniques. To achieve this, we tested three versions of ResNet18 models pretrained on different datasets:

1. **MoCo (Sentinel-2 RGB) - TorchGeo**
2. **SECO (Sentinel-2 RGB) - TorchGeo**
3. **ImageNet - Torchvision**

All models were fine-tuned on a wildfire image classification dataset composed of 128x128 pixel RGB tiles derived from Sentinel-2 imagery.

---

## 📂 Dataset

- **Source:** Sentinel-2 RGB tiles
- **Classes:** `wildfire`, `nowildfire`
- **Format:** `.png` images
- **Splits:**
  - `train/`
  - `val/`
  - `test/`

Each split contains subfolders for each class.

---

## Methodology

### Model Configuration
- Architecture: `ResNet18`
- Input Size: `128 x 128`
- Final FC layer: Modified for binary classification
- Frozen backbone; only `fc` layer is trainable
- Frameworks: `PyTorch`, `TorchGeo`, `Torchvision`

### Training Settings
- Optimizer: `Adam`
- Loss: `CrossEntropyLoss`
- Batch Size: `32`
- Epochs: `10`
- Transformations: `transforms.ToTensor()`

---
