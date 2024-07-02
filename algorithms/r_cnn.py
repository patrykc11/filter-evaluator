import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil
import time

model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.roi_heads.nms_thresh = 0.3
model.roi_heads.score_thresh = 0.3
model.eval()

input_folder = 'images/sgu_images'
output_folder = 'images/r_cnn_output_images'

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder, exist_ok=True)

with torch.no_grad():
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path).convert("RGB")
            image_tensor = F.to_tensor(image).unsqueeze(0)

            start_time = time.time()
            predictions = model(image_tensor)
            inference_time = time.time() - start_time

            pred_scores = predictions[0]['scores'].numpy()
            pred_boxes = predictions[0]['boxes'].numpy()
            pred_labels = predictions[0]['labels'].numpy()

            fig, ax = plt.subplots(1)
            ax.imshow(image)

            for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                if label == 1:
                    rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                         edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(box[0], box[1], f'{score:.3f}', bbox=dict(facecolor='white', alpha=0.5))

            ax.axis('off')
            output_image_path = os.path.join(output_folder, filename)
            plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            print(f"Inference time for {filename}: {inference_time:.3f} seconds")