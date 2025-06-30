# README: Semantic Segmentation with Segment Anything Model (SAM)

## Overview
This repository demonstrates the use of Facebook Research's "Segment Anything Model" (SAM) for semantic segmentation. The project includes loading a pre-trained ViT-H model, uploading an image, and generating automatic segmentation masks.

## Prerequisites
Before you begin, ensure you have the following installed:

1. Python 3.6 or later
2. Required libraries: OpenCV, Matplotlib, NumPy, Torch

You will also need:
- Access to Google Colab (optional for easier execution).
- An internet connection to download the pre-trained model and required dependencies.

## Steps to Run the Code

### Step 1: Install Dependencies
Install the required Python packages and clone the SAM repository:

```bash
!pip install opencv-python matplotlib
!git clone https://github.com/facebookresearch/segment-anything.git
%cd segment-anything
```

### Step 2: Download the Pre-Trained Model Checkpoint
Download the SAM ViT-H model checkpoint and place it in the appropriate directory:

```bash
!mkdir -p models
!wget -O models/sam_vit_h.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### Step 3: Upload an Image
Upload an image file for segmentation:

```python
from google.colab import files
uploaded = files.upload()
```

### Step 4: Load the Model
Load the pre-trained SAM model using Torch:

```python
import torch
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h.pth")
predictor = SamPredictor(sam)
```

### Step 5: Read the Uploaded Image
Load and preprocess the uploaded image:

```python
import cv2
import numpy as np

image_path = list(uploaded.keys())[0]
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

### Step 6: Predict Masks Using Automatic Mask Generator
Generate segmentation masks with the automatic mask generator:

```python
from segment_anything import SamAutomaticMaskGenerator

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
```

### Step 7: Visualize the Segmentation Output
Display the segmented output overlaid on the original image:

```python
import matplotlib.pyplot as plt

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 4))
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.random.random(3).tolist() + [0.6]
        img[m] = color_mask
    plt.imshow(img)

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.title("Segmented Output with SAM")
plt.show()
```

## Expected Output
Once executed, the code will:
1. Load the pre-trained SAM model.
2. Accept an uploaded image.
3. Generate segmentation masks using the Automatic Mask Generator.
4. Display the segmented output with overlaid masks.

## Troubleshooting
- Ensure that all dependencies are installed.
- Verify the model file path.
- If issues persist, confirm that the uploaded image format is supported by OpenCV.

## License
This repository follows the licensing of the original "Segment Anything" project by Facebook Research. Please refer to their repository for licensing terms.
