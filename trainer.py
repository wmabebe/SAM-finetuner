import os
from transformers import SamModel, SamProcessor
from raffm import RaFFM
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from utility import *
from dataset import SA1BDataset


# Initialize the original SAM model and processor
original_model = SamModel.from_pretrained("facebook/sam-vit-base").to("cuda")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# RaFFM configuration and submodel initialization
elastic_config = {
    "atten_out_space": [1280],
    "inter_hidden_space": [2048],
    "residual_hidden_space": [2048],
}
raffm_model = RaFFM(original_model, elastic_config=elastic_config)
submodel, params, config = raffm_model.random_resource_aware_model()
submodel = submodel.to("cuda")  # Move submodel to GPU

dataset = SA1BDataset("SA1B")
train_dataset = Subset(dataset, indices=range(0, 256, 1))
test_dataset = Subset(dataset, indices=range(256, 320, 1))
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1)

print(f'trainloader : {len(train_dataloader)}')
print(f'trainloader : {len(test_dataloader)}')

# Calculate IoUs and average IoU before training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_ious_original, average_iou_original = inference(original_model, test_dataloader, device)
print('IoUs for original:')
for iou in image_ious_original:
    print(iou)
print(f'Average IoU original: {average_iou_original:.4f}')

# image_ious_before, average_iou_before = inference(submodel, test_dataloader)
# print('IoUs for each image before training submodel:')
# for iou in image_ious_before:
#     print(iou)
# print(f'Average IoU before training submodel: {average_iou_before:.4f}')

# Freeze vision_encoder and prompt_encoder in the submodel
for name, param in submodel.named_parameters():
    if name.startswith("mask_decoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)

# Training loop
# optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, submodel.parameters()), lr=1e-8)
# num_epochs = 100

# for epoch in range(num_epochs):
#     torch.cuda.empty_cache()
#     submodel.train()
#     running_loss = 0.0

#     for images, masks in train_dataloader:
#         images = images.to("cuda")
#         masks = masks.to("cuda")

#         optimizer.zero_grad()

#         # Forward pass
#         outputs = submodel(images)['pred_masks']
#         pred_masks = outputs.squeeze(1)

#         # Select the appropriate channel and squeeze the channel dimension
#         pred_masks = pred_masks[:, 0, :, :].unsqueeze(1)

#         # Resize predicted masks to match target masks
#         pred_masks_resized = torch.nn.functional.interpolate(pred_masks, size=(256, 256), mode='bilinear', align_corners=False)

#         # Compute the loss
#         loss_focal = focal_loss(pred_masks_resized, masks)
#         loss_dice = dice_loss(pred_masks_resized, masks)
#         loss = loss_focal + loss_dice

#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     epoch_loss = running_loss / len(train_dataloader)
#     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# # Calculate IoUs and average IoU after training
# image_ious_after, average_iou_after = inference(submodel, test_dataloader)
# print('IoUs for each image after training:')
# for iou in image_ious_after:
#     print(iou)
# print(f'Average IoU after training: {average_iou_after:.4f}')
