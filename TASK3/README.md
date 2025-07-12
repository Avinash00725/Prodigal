# Segment Anything (SAM) for Image and Video Segmentation

This project demonstrates how to use the **Segment Anything Model (SAM)** for both **static image segmentation** and **video frame-by-frame segmentation** using Python, OpenCV, and the SAM library.

## Project Structure

```
project-root/
├── sam2.ipynb           ← Jupyter notebook with SAM code for image and video segmentation
└── content/
    ├── image.jpg        ← Input image for static segmentation
    └── video.mp4        ← Input video for frame-by-frame segmentation
```

## Prerequisites

- Install required packages:
  ```bash
  pip install opencv-python matplotlib numpy segment-anything
  ```
- Download the SAM model checkpoint and place it in the project root or specify its path in the notebook.

## Static Image Segmentation

The following code loads and segments a single image (`content/image.jpg`) using SAM:

```python
import cv2
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator

# Load image
image = cv2.imread("../content/image.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize SAM mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# Generate masks
masks = mask_generator.generate(image_rgb)

# Visualize
plt.imshow(image_rgb)
show_anns(masks)  # Custom function to display segmented masks
plt.axis('off')
plt.show()
```

## Video Segmentation (Frame-by-Frame)

To perform **video segmentation** on `content/video.mp4`, the following code reads the video, processes each frame with SAM, and visualizes the results:

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamAutomaticMaskGenerator

# Load video
video_path = "../content/video.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize SAM mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

frame_count = 0
max_frames = 5  # Process only first 5 frames for preview

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Generate masks
    masks = mask_generator.generate(frame_rgb)

    # Visualize the segmentation
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

    plt.figure(figsize=(8, 8))
    plt.imshow(frame_rgb)
    show_anns(masks)
    plt.axis('off')
    plt.title(f"Segmented Frame {frame_count}")
    plt.show()

    frame_count += 1

cap.release()
```

## Notes

- Ensure `content/image.jpg` and `content/video.mp4` exist in the `content/` directory.
- Adjust `max_frames` to process more or fewer frames as needed.
- The `show_anns` function overlays segmented masks with random colors for visualization.
