import json
import os
import cv2
import torch
from PIL import Image

from torchvision import transforms, datasets
from tqdm import tqdm
from model import GoogLeNet
import matplotlib.pyplot as plt


Image.MAX_IMAGE_PIXELS = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


model = GoogLeNet(num_classes=5, aux_logits=True)
model.cuda()
model_weight_path = "/home/hudian/photo_classification/trible_t2.pth"
model.to(device)
missing_key, unexpect_key = model.load_state_dict(torch.load(model_weight_path), strict=False)

model.eval().to(device)

def predict(i):
    #print(i)
    #img = Image.open(i)
    img = Image.fromarray(i)
    img = data_transforms(img)
    img = torch.unsqueeze(img, dim=0).to(device)


    try:
        json_file = open('/home/hudian/photo_classification/class_indices.json','r')
        class_indict = json.load(json_file)
    except Exception as e:
        
        exit(-1)



    with torch.no_grad():
        output = torch.squeeze(model(img)).to(device)
        predict = torch.softmax(output, dim=0).to(device)
        predict_cla = torch.argmax(predict).cpu().numpy()
    #print(class_indict[str(predict_cla)])
    #print(predict_cla)
    #return(class_indict[str(predict_cla)])
    return(predict_cla)
    #plt.show()




    
