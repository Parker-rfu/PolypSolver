import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def erosion_to_dilate(output):
        z = output.cpu().detach().numpy()
        z = np.where(z > 0.3, 1.0, 0.0)  #covert segmentation result
        z = torch.tensor(z)    
        kernel = np.ones((4, 4), np.uint8)   # kernal matrix
        maskd = np.zeros_like(output.cpu().detach().numpy())  #result array
        maske = np.zeros_like(output.cpu().detach().numpy())  #result array
        for i in range(output.shape[0]):
            y = z[i].permute(1,2,0)
            erosion = y.cpu().detach().numpy()
            dilate = y.cpu().detach().numpy()
            dilate = np.array(dilate,dtype='uint8')
            erosion = np.array(erosion,dtype='uint8')
            erosion = cv.erode(erosion, kernel, 4)  
            dilate = cv.dilate(dilate, kernel, 4)
            mask1 = torch.tensor(dilate-erosion).unsqueeze(-1).permute(2,0,1)
            mask2 = torch.tensor(erosion).unsqueeze(-1).permute(2,0,1)
            maskd[i] = mask1
            maske[i] = mask2
        maskd = torch.tensor(maskd)
        maskd = maskd.cuda()
        maske = torch.tensor(maske)
        maske = maske.cuda()        
        return maskd,maske