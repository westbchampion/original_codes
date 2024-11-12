import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image

class VOCSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, file_list, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        with open(file_list, "r") as f:
            self.images = [line.strip() for line in f]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx] + ".png"
        mask_name = self.images[idx] + ".png"
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        mask = transforms.ToTensor()(mask)  # 将掩码转换为单通道张量，不做归一化
        
        return image, mask

if __name__ == "__main__":
    images_dir = './VOC2007/JPEGImages'
    masks_dir = './VOC2007/SegmentationClass_256'
    train_list = './VOC2007/ImageSets/Segmentation/train.txt'
    val_list = './VOC2007/ImageSets/Segmentation/val.txt'
    file_list = './VOC2007/ImageSets/Segmentation/train.txt'


    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = VOCSegmentationDataset(images_dir, masks_dir, file_list, transform=transform)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for images, masks in data_loader:
        print("Image batch shape:", images.shape)  # 应该是 (batch_size, 3, H, W)
        print("Mask batch shape:", masks.shape)    # 应该是 (batch_size, 1, H, W)
        break
