import torch
from PIL import Image
from torchvision import transforms
from utility import get_image_info, get_ground_truth_masks

class SA1BDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_directory):
        self.image_mask_pairs = get_image_info(dataset_directory, num_images= 320 )  # Adjust num_images as needed
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),  # Resize images to 1024x1024
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for ImageNet
        ])

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.image_mask_pairs[idx]
        image = Image.open(image_path).convert("RGB")
        mask = get_ground_truth_masks(mask_path)[0]  # Assuming there is at least one mask per image

        image = self.transform(image)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        mask = transforms.functional.resize(mask, (256, 256), interpolation=transforms.InterpolationMode.NEAREST)  # Resize mask

        return image, mask