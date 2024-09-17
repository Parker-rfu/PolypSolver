import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.G_function import G
import cv2 as cv
import numpy as np
from model.pvtv2 import pvt_v2_b2
from utils.ED import erosion_to_dilate


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)
    
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
    
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
    
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) 

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

    
class CAM(nn.Module):
    def __init__(self,channels = [512,320,128,64]):
        super(CAM, self).__init__()
        self.Conv_1x1 = nn.Conv2d(channels[0],channels[0],kernel_size=1,stride=1,padding=0)
        self.ConvBlock4 = conv_block(ch_in=channels[0], ch_out=channels[0])
	
        self.Up3 = up_conv(ch_in=channels[0],ch_out=channels[1])
        self.AG3 = Attention_block(F_g=channels[1],F_l=channels[1],F_int=channels[2])
        self.ConvBlock3 = conv_block(ch_in=2*channels[1], ch_out=channels[1])

        self.Up2 = up_conv(ch_in=channels[1],ch_out=channels[2])
        self.AG2 = Attention_block(F_g=channels[2],F_l=channels[2],F_int=channels[3])
        self.ConvBlock2 = conv_block(ch_in=2*channels[2], ch_out=32)
        
        self.CA4 = ChannelAttention(channels[0])
        self.CA3 = ChannelAttention(2*channels[1])
        self.CA2 = ChannelAttention(2*channels[2])
        
        self.SA = SpatialAttention()

    def forward(self, x1, x2, x3):
        d4 = self.Conv_1x1(x1)
        
        # CAM4
        d4 = self.CA4(d4)*d4
        d4 = self.SA(d4)*d4 
        d4 = self.ConvBlock4(d4)
        
        # upconv3
        d3 = self.Up3(d4)
        
        # AG3
        y3 = self.AG3(g=d3,x=x2)
        
        # Concat 3
        d3 = torch.cat((y3,d3),dim=1)
        
        # CAM3
        d3 = self.CA3(d3)*d3
        d3 = self.SA(d3)*d3        
        d3 = self.ConvBlock3(d3)
        
        # upconv2
        d2 = self.Up2(d3)
        
        # AG2
        y2 = self.AG2(g=d2,x=x3)
        
        # Concat 2
        d2 = torch.cat((y2,d2),dim=1)
        
        # CAM2
        d2 = self.CA2(d2)*d2
        d2 = self.SA(d2)*d2
        d2 = self.ConvBlock2(d2)
        return d4,d3,d2
        
        
class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h
    
class MFM(nn.Module):
    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super(MFM, self).__init__()

        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)
        self.g_func = G(self.num_s)
        
    def forward(self, x, edge):
        edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)
        
        
        x_state_reshaped = self.g_func(self.conv_state(x)).view(n, self.num_s, -1)
        # x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out =  (self.conv_extend(x_state))

        return out
class BEM(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel):
        self.init__ = super(BEM, self).__init__()
        

        act_fn         = nn.ReLU(inplace=True)
                
        ## ---------------------------------------- ##
        self.layer0    = BasicConv2d(in_channel1, out_channel // 2, 1)
        self.layer1    = BasicConv2d(in_channel2, out_channel // 2, 1)
        
        self.layer3_1  = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),  nn.BatchNorm2d(out_channel // 2),act_fn)
        self.layer3_2  = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),  nn.BatchNorm2d(out_channel // 2),act_fn)
        
        self.layer5_1  = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),  nn.BatchNorm2d(out_channel // 2),act_fn)
        self.layer5_2  = nn.Sequential(nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),  nn.BatchNorm2d(out_channel // 2),act_fn)
        
        self.layer_out = nn.Sequential(nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(out_channel),act_fn)


    def forward(self, x0, x1):
        
        ## ------------------------------------------------------------------ ##
        x0_1  = self.layer0(x0)
        x1_1  = self.layer1(x1)
        
        x_3_1 = self.layer3_1(torch.cat((x0_1,  x1_1),  dim=1))    
        x_5_1 = self.layer5_1(torch.cat((x1_1,  x0_1),  dim=1))

        x_3_2 = self.layer3_2(torch.cat((x_3_1, x_5_1), dim=1))
        x_5_2 = self.layer5_2(torch.cat((x_5_1, x_3_1), dim=1))
        
        out   = self.layer_out(x0_1 + x1_1 + torch.mul(x_3_2, x_5_2))
        return out
    
class GateFusion(nn.Module):
    def __init__(self, in_planes):
        self.init__ = super(GateFusion, self).__init__()
        
        self.gate_1 = nn.Conv2d(in_planes*2, 1, kernel_size=1, bias=True)
        self.gate_2 = nn.Conv2d(in_planes*2, 1, kernel_size=1, bias=True)
        
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        
        ###
        cat_fea = torch.cat([x1,x2], dim=1)
        
        ###
        att_vec_1  = self.gate_1(cat_fea)
        att_vec_2  = self.gate_2(cat_fea)

        att_vec_cat  = torch.cat([att_vec_1, att_vec_2], dim=1)
        att_vec_soft = self.softmax(att_vec_cat)
        
        att_soft_1, att_soft_2 = att_vec_soft[:, 0:1, :, :], att_vec_soft[:, 1:2, :, :]
        x_fusion = x1 * att_soft_1 + x2 * att_soft_2
        
        return x_fusion  
    
class PolypSolver(nn.Module):
    def __init__(self, channel=32):
        super(PolypSolver, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer1_1 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.CAM = CAM()
        
        self.BEM = BEM(64,128,32)

        self.MFM1 = MFM()
        self.MFM2 = MFM()
        
        self.out_CIM = nn.Conv2d(channel, 1, 1)
        self.out_MFM = nn.Conv2d(channel, 1, 1)
        self.out_CAM = nn.Conv2d(channel, 1, 1)
        self.out_4 = nn.Conv2d(512, 32, 1)
        self.out_3 = nn.Conv2d(320, 32, 1)
        act_fn = nn.ReLU(inplace=True)
        self.low_fusion  = GateFusion(32)
        
        ## ---------------------------------------- ##
        self.layer_edge0 = nn.Sequential(nn.Conv2d(32, 32,  kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel),act_fn)
        self.layer_edge1 = nn.Sequential(nn.Conv2d(32, 32,  kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel),act_fn)
        self.layer_edge2 = nn.Sequential(nn.Conv2d(32, 32,   kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(channel),act_fn)
        
        self.seg1 = nn.Conv2d(32,1,1)
        self.seg2 = nn.Conv2d(32,1,1)
        ##-----------------------------------------##
    def forward(self, x):

        # backbone    # [64, 128, 320, 512]
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]
        
        x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=True)
        cim_feature = self.BEM(x1,x2)

        d4,d3,d2 = self.CAM(x4, x3, x2)
        d3 = self.out_3(d3)
        d4 = self.out_4(d4)
        d4 = F.interpolate(d4, scale_factor=4, mode='bilinear')
        d3 = F.interpolate(d3, scale_factor=2, mode='bilinear')
        cam_feature = d4 + d3 + d2
        mask_d1,mask_e1 = erosion_to_dilate(self.seg1(cam_feature))
        mask_d2,mask_e2 = erosion_to_dilate(self.seg2(cim_feature))
        cam_feature_1 = cam_feature*(1-mask_e1)
        T2_1 = cim_feature
        cam_feature_2 = cam_feature
        T2_2 = cim_feature*(1-mask_e2)
        mfm_feature_1 = self.MFM1(T2_1,cam_feature_1)
        mfm_feature_2 = self.MFM2(cam_feature_2,T2_2)
    
        prediction1 = self.out_CAM(mfm_feature_1+cam_feature)
        prediction2 = self.out_MFM(mfm_feature_2+cam_feature)

        prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear') 
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')
        return prediction1_8, prediction2_8


if __name__ == '__main__':
    model = PolypSolver().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    prediction1, prediction2 = model(input_tensor)
    print(prediction1.size(), prediction2.size())