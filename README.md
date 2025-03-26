# instance_segmentation


# Instance Segmentation using Mask R-CNN

## Overview
This project implements **instance segmentation** using **Mask R-CNN with a ResNet-50 FPN backbone**, a pre-trained model from PyTorch's `torchvision` library. The model detects objects, draws bounding boxes, and applies segmentation masks.

## Features
- Uses **Mask R-CNN** pre-trained on the COCO dataset.
- Detects multiple object classes (e.g., person, car, bus, dog, etc.).
- Applies **bounding boxes and masks** to detected objects.
- Uses **confidence thresholding** to filter weak detections.
- Visualizes **segmentation results** using OpenCV and Matplotlib.



## Dependencies
Ensure you have the following installed:

```bash
pip install torch torchvision opencv-python numpy matplotlib
```

## How to Run
1. **Load the model**: Initializes the Mask R-CNN model with pre-trained weights.
2. **Preprocess an image**: Converts and normalizes input images.
3. **Perform instance segmentation**: Generates bounding boxes, masks, and class labels.
4. **Display results**: Overlays segmentation masks and labels on the image.

### Example Usage
```python
import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load Pretrained Mask R-CNN Model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define COCO class labels (subset for visualization)
COCO_CLASSES = {1: "Person", 2: "Bicycle", 3: "Car", 4: "Motorcycle", 6: "Bus", 7: "Train", 8: "Truck"}

# Load Image
def process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, F.to_tensor(image).unsqueeze(0)

# Perform Segmentation
def segment_image(image_path):
    image, image_tensor = process_image(image_path)
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    return image, predictions

# Visualize Results
def visualize_results(image, predictions, threshold=0.5):
    for i in range(len(predictions['scores'])):
        if predictions['scores'][i] > threshold:
            box = predictions['boxes'][i].numpy().astype(int)
            label_id = predictions['labels'][i].item()
            label = COCO_CLASSES.get(label_id, "Unknown")
            score = predictions['scores'][i].item()
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            cv2.putText(image, f"{label}: {score:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Run the model
image_path = "path/to/image.jpg"
image, predictions = segment_image(image_path)
visualize_results(image, predictions)
```

## Expected Output
The script will display the **original image with bounding boxes and segmentation masks** over detected objects.


## Notes
- Set the correct **image path** before running the script.
- The model detects **COCO dataset objects**.
- Adjust the **confidence threshold** to filter detections.
![Screenshot 2025-03-27 002805](https://github.com/user-attachments/assets/4829d31b-9fb0-432f-8268-d8400f7aa763)

## License
This project is open-source and available under the MIT License.

