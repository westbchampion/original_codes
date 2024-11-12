import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets

from model import GoogLeNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 图像转换操作
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机截取图像的一部分
                                 transforms.RandomHorizontalFlip(),  # 以给定概率随机水平旋转图像，默认概率为0.5
                                 transforms.ToTensor(),  # 将图像转换为Tensor类型
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),  # 归一化处理
    "val": transforms.Compose([transforms.Resize((224, 224)),  # 将图片转换为224*224的尺寸
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}

data_root = os.getcwd()  # 返回当前.py文件所在目录
image_path = data_root + "\\data\\"

# 加载数据集
train_dataset = datasets.ImageFolder(root=image_path + "temp_file_train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)
# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())

# 将dict转换为json文件
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "temp_file_train",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=0)

net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0003)

best_acc = 0.0
save_path = './googleNet_test.pth'
for epoch in range(30):
    # train
    net.train()
    running_loss = 0.0
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits, aux_logits2, aux_logits1 = net(images.to(device))
        # print(type(aux_logits1))
        # print(aux_logits1)
        loss0 = loss_function(logits, labels.to(device))
        loss1 = loss_function(aux_logits1, labels.to(device))
        loss2 = loss_function(aux_logits2, labels.to(device))
        loss = loss0 + loss1 * 0.3 + loss2 * 0.3
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    net.eval()
    acc = 0.0
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))

print('finished training')
