import os
from transformers import SamModel, SamConfig, SamProcessor
from raffm import RaFFM
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from utility import *
from dataset import SA1BDataset, SAMDataset
from torch.optim import Adam
import monai
from statistics import mean


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

# dataset = SA1BDataset("SA1B")
# train_dataset = Subset(dataset, indices=range(0, 256, 1))
# test_dataset = Subset(dataset, indices=range(256, 320, 1))
# train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=1)

# Create an instance of the SAMDataset
train_dataset = load_dataset("datasets/mitochondria/training.tif", "datasets/mitochondria/training_groundtruth.tif")
test_dataset = load_dataset("datasets/mitochondria/testing.tif", "datasets/mitochondria/testing_groundtruth.tif")
train_dataset = SAMDataset(dataset=train_dataset, processor=processor)
test_dataset = SAMDataset(dataset=test_dataset, processor=processor)

# Create a DataLoader instance for the training dataset
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, drop_last=True)

print(f'trainloader : {len(train_dataloader)}')
print(f'trainloader : {len(test_dataloader)}')

# Calculate IoUs and average IoU before training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
miou = test(submodel,test_dataloader,disable_verbose=True)
print(f'pre-train mIoU : {miou}')

# make sure we only compute gradients for mask decoder
for name, param in submodel.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)


# Initialize the optimizer and the loss function
optimizer = Adam(submodel.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
#Try DiceFocalLoss, FocalLoss, DiceCELoss
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')


#Training loop
num_epochs = 5

device = "cuda" if torch.cuda.is_available() else "cpu"
submodel.to(device)

submodel.train()
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader, disable=True):
      # forward pass
      outputs = submodel(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

      # compute loss
      predicted_masks = outputs.pred_masks.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')

# Save the model's state dictionary to a file
torch.save(submodel.state_dict(), "models/mito_submodel_checkpoint.pth")

"""**Inference**"""

# Load the model configuration
model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the model architecture with the loaded configuration
my_mito_submodel = SamModel(config=model_config)
#Update the model by loading the weights from saved file.
my_mito_submodel.load_state_dict(torch.load("models/mito_submodel_checkpoint.pth"))

# set the device to cuda if available, otherwise use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
my_mito_submodel.to(device)


#Testing

miou = test(my_mito_submodel,test_dataloader,disable_verbose=True)
print(f'post-train mIoU : {miou}')

