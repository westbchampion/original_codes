import json
import os
import cv2
import torch
from PIL import Image

from torchvision import transforms, datasets

from model import GoogLeNet
import matplotlib.pyplot as plt
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


model = GoogLeNet(num_classes=5, aux_logits=True)
model.cuda()
model_weight_path = "./MAndS/trible_t2.pth"
model.to(device)
missing_key, unexpect_key = model.load_state_dict(torch.load(model_weight_path), strict=False)

model.eval().to(device)

def predict(i):
    img = Image.open(i)
    #print(type(i))
    #img = Image.fromarray(i)
    img = data_transforms(img)
    img = torch.unsqueeze(img, dim=0).to(device)


    try:
        json_file = open('./MAndS/class_indices.json','r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)



    with torch.no_grad():
        output = torch.squeeze(model(img)).to(device)
        predict = torch.softmax(output, dim=0).to(device)
        predict_cla = torch.argmax(predict).cpu().numpy()
    #print(class_indict[str(predict_cla)])
    #print(predict_cla)
    return(class_indict[str(predict_cla)])
    #return(predict_cla)
    #plt.show()

image_path = "MAndS/data/temp_file_test/"
image = os.listdir(image_path)
t = 0
ta = 0
for i in tqdm(image):
    # res = predict(str(i) + "_muscle.jpg")
    #i=cv2.imread(image_path+i)
    res = predict(image_path+i)
    #print(i)
    #print(res)
    if res == i.split('_')[1].split('.')[0]:
        t=t+1
    ta=ta+1
print(t/ta)

# image_path = "data/result_png.jpg"
# predict(image_path)