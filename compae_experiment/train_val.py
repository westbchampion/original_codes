import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
from dataset import VOCSegmentationDataset 
from sklearn.metrics import jaccard_score, accuracy_score, recall_score 
import numpy as np
# from models import DenseUNet, UNetPlusPlus, AttentionUNet, UNet, ResUNet
from models import U_Net, AttU_Net, ResUNet
from unetplusplus import NestedUNet
from denseunet import DenseUNet
import matplotlib.pyplot as plt
import os


# UNet：源代码
# Attention Unet: https://github.com/LeeJunHyun/Image_Segmentation
# UNet++: https://github.com/4uiiurz1/pytorch-nested-unet
# ResUnet: smp的库
# DenseUnet:https://github.com/stefano-malacrino/DenseUNet-pytorch
transform = transforms.Compose([
    transforms.ToTensor(),
])

images_dir = './VOCdevkit_cell/VOC2007/JPEGImages'
masks_dir = './VOCdevkit_cell/VOC2007/SegmentationClass'
train_list = './VOCdevkit_cell/VOC2007/ImageSets/Segmentation/trainval.txt'
val_list = './VOCdevkit_cell/VOC2007/ImageSets/Segmentation/test.txt'

train_dataset = VOCSegmentationDataset(images_dir, masks_dir, train_list, transform=transform)
val_dataset = VOCSegmentationDataset(images_dir, masks_dir, val_list, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize model
model = ResUNet()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training configuration
num_epochs = 10
train_losses = []
val_losses = []
miou_scores = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.float())
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

    train_losses.append(train_loss / len(train_loader))

    model.eval()
    val_loss = 0
    all_ious, all_accuracies, all_recalls = [], [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks.float())
            val_loss += loss.item()

            preds = (outputs.sigmoid() > 0.5).cpu().numpy().astype(int)
            masks_np = masks.cpu().numpy().astype(int)
            for i in range(preds.shape[0]):
                iou = jaccard_score(masks_np[i].flatten(), preds[i].flatten(), average="macro")
                accuracy = accuracy_score(masks_np[i].flatten(), preds[i].flatten())
                recall = recall_score(masks_np[i].flatten(), preds[i].flatten(), average="macro")
                all_ious.append(iou)
                all_accuracies.append(accuracy)
                all_recalls.append(recall)

    val_losses.append(val_loss / len(val_loader))
    miou = np.mean(all_ious)
    miou_scores.append(miou)
    
    maccuracy = np.mean(all_accuracies)
    mrecal = np.mean(all_recalls)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
          f"Val Loss: {val_loss/len(val_loader):.4f}, mIoU: {miou:.4f}, mAccuracy: {maccuracy:.4f}, mRecall: {mrecal:.4f}")

# Plotting Loss and mIoU
plt.figure(figsize=(10, 5))

# Plot Train and Val Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train and Validation Loss")
plt.legend()

# Plot mIoU
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), miou_scores, label="mIoU", color='g')
plt.xlabel("Epoch")
plt.ylabel("mIoU")
plt.title("Mean IoU (mIoU)")
plt.legend()

# Save the plots
plt.tight_layout()
plt.savefig("./training_plots.png")
plt.show()

# Save the model
os.makedirs('./model', exist_ok=True)
torch.save(model.state_dict(), './model/unetplusplus.pth')
print("Model saved to ./model/unetplusplus.pth")
