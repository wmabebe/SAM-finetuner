from sklearn.metrics import confusion_matrix
import matplotlib as plt
import tifffile
import random
import torch
import numpy as np
from transformers import SamModel, SamConfig, SamProcessor

from transformers import SamModel

#Get bounding boxes from mask.
def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox

large_test_images = tifffile.imread("datasets/mitochondria/testing.tif")
large_test_image = large_test_images[1]

model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the model architecture with the loaded configuration
my_mito_model = SamModel(config=model_config)
#Update the model by loading the weights from saved file.
my_mito_model.load_state_dict(torch.load("models/mito_model_checkpoint.pth"))

# set the device to cuda if available, otherwise use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
my_mito_model.to(device)

# Create an empty list to store results and mIoU values
all_test_results = []
all_miou_values = []

# Iterate over each test image
for test_image_idx in range(len(large_test_images)):
    # Load a test image
    test_image = large_test_images[test_image_idx]

    # Get bounding box prompt based on ground truth segmentation map
    ground_truth_mask = np.zeros_like(test_image)  # Assuming no ground truth mask is available
    prompt = get_bounding_box(ground_truth_mask)

    # Prepare image + box prompt for the model
    inputs = processor(test_image, input_boxes=[[prompt]], return_tensors="pt")

    # Move the input tensor to the GPU if it's not already there
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Set the model to evaluation mode
    my_mito_model.eval()

    # Forward pass
    with torch.no_grad():
        outputs = my_mito_model(**inputs, multimask_output=False)

    # Apply sigmoid
    test_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    # Convert soft mask to hard mask
    test_seg_prob = test_seg_prob.cpu().numpy().squeeze()
    test_seg = (test_seg_prob > 0.5).astype(np.uint8)

    # Compute mIoU
    ground_truth_flat = ground_truth_mask.flatten()
    test_seg_flat = test_seg.flatten()

    # Using confusion matrix to compute mIoU
    cm = confusion_matrix(ground_truth_flat, test_seg_flat, labels=[0, 1])
    intersection = np.diag(cm)
    union = np.sum(cm, axis=(0, 1)) + 1e-10  # Add a small epsilon to avoid division by zero
    iou = intersection / union
    miou = np.mean(iou)

    # Store the results and mIoU value
    all_test_results.append({
        "test_image": test_image,
        "test_seg_prob": test_seg_prob,
        "test_seg": test_seg
    })
    all_miou_values.append(miou)

# Display mIoU values for all test images
for idx, miou_value in enumerate(all_miou_values):
    print(f"mIoU for Test Image {idx + 1}: {miou_value}")
