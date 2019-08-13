from skimage import img_as_float
import copy
import os
import os.path as osp
import torch.nn as nn
import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch
import torch.hub
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms
import argparse
from guided_grad_cam.GradCam_cuda import *
from torch.utils.data import DataLoader
from models.DLmodel import *
from dataloader.LoadData import *
import argparse

def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
    else:
        print("Device: CPU")
    return device

parser = argparse.ArgumentParser(description='tumor')
parser.add_argument('--folder_no', type=int, metavar='N')
args = parser.parse_args()
folder_no = args.folder_no

#pre info
model_path = './Data_mapped_images/img_fold'+str(folder_no)+'/network_0505_th1.pth'
train_csv_path = './Data_mapped_images/img_fold'+str(folder_no)+'/train/labels_train.csv'
train_root_path = './Data_mapped_images/img_fold'+str(folder_no)+'/train'
output_dir = './heatmaps'
batch_size = 1
topk = 1
cuda = True
classes = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC',
           'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD',
           'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
target_layer = "module.conv3"



#datasets
train_dataset = TumorDatasetTrain(csv_file=train_csv_path, root_dir=train_root_path,
                                  transform=transforms.Compose([ToTensor()]))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

#load model
device = get_device(cuda)
model = Net(33)
model = model.double()
model = nn.DataParallel(model)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()
gcam = GradCAM(model=model)
gbp = GuidedBackPropagation(model=model)

count=0
#generate grad-cam
for batch_idx, diction in enumerate(train_loader):
    images = Variable(diction['image'].cuda(),requires_grad = True)
    # labels = Variable(diction['label'])
    # images = images.to(device)
    # labels = labels.to(device)
    # labels = labels.view(-1)
    images = images.double()
    #debug
    # images = images[48]
    # images = images.reshape(1,1,102,102)
    probs, ids = gcam.forward(images)
    _ = gbp.forward(images)
    for i in range(topk):    
        gcam.backward(idx=ids[i])
        regions = gcam.generate(target_layer=target_layer)
        gbp.backward(idx=ids[i])
        gradients = gbp.generate()

        gcam.save(
            filename=osp.join(
                output_dir,"gradcam",str(folder_no),classes[ids[i]],
                "{}-gradcam-{}.png".format(
                    count ,classes[ids[i]]
                ),
            ),
            gcam=regions,
        )
        gbp.save(
            filename=osp.join(
                output_dir,"guided_gradcam",str(folder_no),classes[ids[i]],
                "{}-guided_gcam-{}.png".format(
                   count ,classes[ids[i]]
                ),
            ),
            data = regions*gradients
        )
        count+=1
        print(count)
    print("batch_no:",batch_idx)
    torch.cuda.empty_cache()