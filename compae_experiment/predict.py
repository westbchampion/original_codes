import torch
from nets.unet import Unet
import os
import torchvision.transforms as tfs
import torch.nn.functional as F 
import cv2
import numpy as np
import openpyxl


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 2
VOCdevkit_path = '../VOCdevkit_old'
with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/test.txt"), "r") as f:
    test_lines = f.readlines()


def fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 



def test(model):
    ious = []
    recs = []
    pres = []

    # 创建一个新的Excel工作簿
    workbook = openpyxl.Workbook()

    # 获取默认的工作表
    sheet = workbook.active

    # 写入数据
    sheet['A1'] = 'name'
    sheet['B1'] = 'IoU'
    sheet['C1'] = 'Recall'
    sheet['D1'] = 'Precision'

    for annotation_line in test_lines:
        model.eval()
        # model = nn.DataParallel(model)
        model = model.to(device)
        ckp = torch.load('./logs/best_epoch_weights.pth', map_location=device)
        model.load_state_dict(ckp)

        name = annotation_line.split()[0]
        img = cv2.imread(os.path.join(os.path.join(VOCdevkit_path, "VOC2007/JPEGImages"), name + ".png"))
        label = cv2.imread(os.path.join(os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass"), name + ".png"))[:, :, 0]

        fuse = img.copy()
        img = img[:, :, (2, 1, 0)]
        img = tfs.ToTensor()(img)
        img = torch.unsqueeze(img, dim=0)
        img = img.cuda().type(torch.cuda.FloatTensor)
        pred = model(img)

        pred = F.log_softmax(pred, dim=1)
        predicted_classes = torch.argmax(pred, dim=1)
        predicted_classes = torch.squeeze(predicted_classes, dim=0)
        pred = predicted_classes.cpu().detach().numpy()
        print(name)
        
        hist = np.zeros((num_classes, num_classes))
        for lt, lp in zip(label,pred):
            hist += fast_hist(lt.flatten(), lp.flatten(), num_classes)

        IoUs        = per_class_iu(hist)
        PA_Recall   = per_class_PA_Recall(hist)
        Precision   = per_class_Precision(hist)

        iou = np.nanmean(IoUs)
        rec = np.nanmean(PA_Recall)
        pre = np.nanmean(Precision)

        ious.append(iou)
        recs.append(rec)
        pres.append(pre)

        # print('IoUs: %f' % iou)
        # print('PA_Recall: %f' % rec)
        # print('Precision: %f' % pre)

        # 添加一行数据
        sheet.append([name, iou, rec, pre])
        
        pred = pred*255
        label = label*255
        m, n = pred.shape
        # fuse = np.zeros([m, n, 3])
        # for i in range(3):
        #     fuse[:, :, i] = label

        for i in range(m):
            for j in range(n):
                if pred[i, j] == label[i, j] and label[i, j] == 255:
                    fuse[i, j, :] = [255, 0, 0]
                elif pred[i, j] != label[i, j] and label[i, j] == 255:
                    fuse[i, j, :] = [0, 0, 255]
                elif pred[i, j] != label[i, j] and label[i, j] == 0:
                    fuse[i, j, :] = [0, 255, 0]
        
        preds_path = results_path + 'preds/'
        if not os.path.exists(preds_path):
            os.mkdir(preds_path)

        fuses_path = results_path + 'fuses/'
        if not os.path.exists(fuses_path):
            os.mkdir(fuses_path)

        labels_path = results_path + 'labels/'
        if not os.path.exists(labels_path):
            os.mkdir(labels_path)

        cv2.imwrite(preds_path + '%s.png' % name, pred)
        cv2.imwrite(fuses_path + '%s.png' % name, fuse)
        cv2.imwrite(labels_path + '%s.png' % name, label)

    # 保存工作簿
    workbook.save('UNet_result.xlsx')

    m_iou = np.mean(ious)
    m_rec = np.mean(recs)
    m_pre = np.mean(pres)

    print('mean_IoU: %f' % m_iou)
    print('mean_Recall: %f' % m_rec)
    print('mean_Precision: %f' % m_pre)

    with open('UNet_results.txt', 'w') as file:
        file.write('mean_IoU: %f' % m_iou + '\n')
        file.write('mean_Recall: %f' % m_rec + '\n')
        file.write('mean_Precision: %f' % m_pre + '\n')

model = Unet(num_classes=num_classes)
results_path = './results/'
if not os.path.exists(results_path):
    os.mkdir(results_path)
test(model)