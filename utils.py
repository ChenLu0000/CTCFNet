import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

def flatten(tensor):
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))     
    transposed = tensor.permute(axis_order)
    return transposed.contiguous().view(C, -1)

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5

    def forward(self, output, target):
        output = torch.argmax(output,dim=1).unsqueeze(1).type(torch.cuda.FloatTensor)
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target)
        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)
        dice = intersect / denominator
        dice = torch.mean(dice)
        return 1 - dice

class DetailAggregateLoss(nn.Module):
    def __init__(self,):
        super(DetailAggregateLoss, self).__init__()
        self.roberts_x = torch.tensor(
            [-1, 0, 0, 0, 1, 0, 0, 0, 0],
            dtype=torch.float32
        ).reshape(1,1,3,3).requires_grad_(False).type(torch.cuda.FloatTensor)
        self.roberts_y = torch.tensor(
                [0, -1, 0, 1, 0, 0, 0, 0, 0],
                dtype=torch.float32
            ).reshape(1,1,3,3).requires_grad_(False).type(torch.cuda.FloatTensor)
        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6./10], [3./10], [1./10]],
            dtype=torch.float32).reshape(1, 3, 1, 1).type(torch.cuda.FloatTensor))
        self.dice_loss_func = DiceLoss()

    def forward(self, boundary_logits, gtmasks):
        boundary_targets_x = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.roberts_x, padding=1)
        boundary_targets_x = boundary_targets_x.clamp(min=0)
        boundary_targets_y = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.roberts_y, padding=1)
        boundary_targets_y = boundary_targets_y.clamp(min=0)
        boundary_targets = boundary_targets_x + boundary_targets_y
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2_x = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.roberts_x, stride=2, padding=1)
        boundary_targets_x2_x = boundary_targets_x2_x.clamp(min=0)
        boundary_targets_x2_y = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.roberts_y, stride=2, padding=1)
        boundary_targets_x2_y = boundary_targets_x2_y.clamp(min=0)
        boundary_targets_x2 = boundary_targets_x2_x + boundary_targets_x2_y
        
        boundary_targets_x4_x = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.roberts_x, stride=4, padding=1)
        boundary_targets_x4_x = boundary_targets_x4_x.clamp(min=0)
        boundary_targets_x4_y = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.roberts_y, stride=4, padding=1)
        boundary_targets_x4_y = boundary_targets_x4_y.clamp(min=0)
        boundary_targets_x4 = boundary_targets_x4_x + boundary_targets_x4_y

        boundary_targets_x8_x = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.roberts_x, stride=8, padding=1)
        boundary_targets_x8_x = boundary_targets_x8_x.clamp(min=0)
        boundary_targets_x8_y = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.roberts_y, stride=8, padding=1)
        boundary_targets_x8_y = boundary_targets_x8_y.clamp(min=0)
        boundary_targets_x8 = boundary_targets_x8_x + boundary_targets_x8_y
    
        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')

        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0
        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0
        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0

        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=1)
        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)
        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0
        
        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)
        
        bce_loss = F.binary_cross_entropy_with_logits(torch.argmax(boundary_logits,dim=1).type(torch.cuda.FloatTensor), boudary_targets_pyramid.squeeze(1).type(torch.cuda.FloatTensor))
        dice_loss = self.dice_loss_func(torch.sigmoid(boundary_logits), boudary_targets_pyramid)
        
        return bce_loss,  dice_loss

def get_label_colormap(csv_path='./datasets/name/class_dict.csv'):
    pd_label_color = pd.read_csv(csv_path, sep=',')

    classes = []
    colormap = []
    for i in range(len(pd_label_color.index)):

        tmp = pd_label_color.iloc[i]
        color = []
        color.append(tmp['r'])
        color.append(tmp['g'])
        color.append(tmp['b'])
        colormap.append(color)
        classes.append(tmp['name'])

    return colormap

def image2label(label,label_colormap=get_label_colormap()):
    cm2lbl = np.zeros(256**3)
    for i,  cm in enumerate(label_colormap):
        cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i

    data = np.array(label, dtype='int32')
    idx =(data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    image2 = np.array(cm2lbl[idx], dtype='int64')
    return image2