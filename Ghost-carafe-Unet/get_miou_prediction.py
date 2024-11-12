from unet import Unet as psp
from torch import nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F  
import numpy as np
import colorsys
import torch
import copy
import os

class miou_Pspnet(psp):
    def detect_image(self, image):
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        image, nw, nh = self.letterbox_image(image,(512,512))
        images = [np.array(image)/255]
        images = np.transpose(images,(0,3,1,2))
        
        with torch.no_grad():
            images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
        
        pr = pr[int((512-nh)//2):int((512-nh)//2+nh), int((512-nw)//2):int((512-nw)//2+nw)]
        
        image = Image.fromarray(np.uint8(pr)).resize((orininal_w,orininal_h),Image.NEAREST)

        return image

psp = miou_Pspnet()

image_ids = open(r"VOCdevkit\VOC2007\ImageSets\Segmentation\val.txt",'r').read().splitlines()

if not os.path.exists("./miou_pr_dir"):
    os.makedirs("./miou_pr_dir")

for image_id in image_ids:
    image_path = "./VOCdevkit/VOC2007/SegmentationClass/"+image_id+".png"
    image = Image.open(image_path)
    image = image.resize((512, 512))
    image.save("./miou_pr_dir copy/" + image_id + ".png")

    image_path = "./VOCdevkit/VOC2007/JPEGImages/"+image_id+".png"

    image = Image.open(image_path)
    image = psp.detect_image(image)
    image = image.resize((512, 512))
    image.save("./miou_pr_dir/" + image_id + ".png")
    print(image_id," done!")
