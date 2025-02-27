import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax, Dropout

from typing import List, Callable
from torch import Tensor

import math


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

        return x
        
class BasicConv2dReLu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2dReLu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MSA(nn.Module):
    def __init__(self, channel):
        super(MSA , self).__init__()

        self.initial_depth_conv = nn.Conv2d(channel, channel, kernel_size=5, padding=2, groups=channel)

        self.depth_convs = nn.ModuleList([
            nn.Conv2d(channel, channel, kernel_size=(1, 7), padding=(0, 3), groups=channel),
            nn.Conv2d(channel, channel, kernel_size=(7, 1), padding=(3, 0), groups=channel),
            nn.Conv2d(channel, channel, kernel_size=(1, 11), padding=(0, 5), groups=channel),
            nn.Conv2d(channel, channel, kernel_size=(11, 1), padding=(5, 0), groups=channel),
            nn.Conv2d(channel, channel, kernel_size=(1, 21), padding=(0, 10), groups=channel),
            nn.Conv2d(channel, channel, kernel_size=(21, 1), padding=(10, 0), groups=channel),
        ])

        self.pointwise_conv = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):

        initial_out = self.initial_depth_conv(inputs)

        spatial_outs = [conv(initial_out) for conv in self.depth_convs]
        spatial_out = sum(spatial_outs)

        spatial_att = self.pointwise_conv(spatial_out)
        out = spatial_att * inputs
        out = self.pointwise_conv(out)
        return out



class DiagonalLowerSplit(nn.Module):
    def __init__(self):
        super(DiagonalLowerSplit, self).__init__()

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        mask = torch.triu(torch.ones(height, width), diagonal=1).to(x.device)
        
        x = x * (1 - mask.unsqueeze(0).unsqueeze(0))  
        
        return x

class DiagonalUpperSplit(nn.Module):
    def __init__(self):
        super(DiagonalUpperSplit, self).__init__()

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        mask = torch.tril(torch.ones(height, width), diagonal=0).to(x.device)
        
        x = x * mask.unsqueeze(0).unsqueeze(0)  
        
        return x

class Diagonal(nn.Module): 
    def __init__(self):
        super(Diagonal, self).__init__()
        self.diagonal_lower_split = DiagonalLowerSplit()
        self.diagonal_upper_split = DiagonalUpperSplit()

    def rotate_tensor_90(self, tensor):
        
        return tensor.rot90(1, dims=(-2, -1))

    def forward(self, x):    
       
        x1 = self.diagonal_lower_split(x)
        x2 = self.diagonal_upper_split(x)
        
        x_rotated = self.rotate_tensor_90(x)
        
        x3 = self.diagonal_lower_split(x_rotated)
        x4 = self.diagonal_upper_split(x_rotated)
        
        x_Diagonal1 = x1 + x4
        x_Diagonal2 = x2 + x3
        
        return x_Diagonal1, x_Diagonal2


class DEA(nn.Module):
    def __init__(self, channel):
        super(DEA, self).__init__()
        self.diagonal = Diagonal()       
        
        self.query_conv = nn.Conv2d(channel, channel, kernel_size=1)
        self.key_conv = nn.Conv2d(channel, channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)

        self.value_conv = nn.Conv2d(channel, channel, kernel_size=1)

        self.gamma_cur = nn.Parameter(torch.ones(1))
        
        self.conv = nn.Sequential(
            BasicConv2dReLu(channel, channel, 3, padding=1),
            nn.Dropout2d(0.1, False),
            BasicConv2dReLu(channel, channel, 1)
        )

    def forward(self, x):
        """
        inputs:
            x : input feature maps (B X C X H X W)
        returns:
            out : attention value + input feature
            attention: C X H x W
        """
        x1, x2 = self.diagonal(x)
        
        proj_query = x1
        proj_key = x2
        proj_value = x

        
        query = self.query_conv(proj_query)
        key = self.key_conv(proj_key)
        value = self.value_conv(proj_value)
        
        
        B, C, H, W = query.size()
        query = query.view(B, C, -1)
        key = key.view(B, C, -1)
        value = value.view(B, C, -1)
        
        
        attention_scores = torch.bmm(query.permute(0, 2, 1), key)
        attention_scores = self.softmax(attention_scores)

        attention_output = torch.bmm(value, attention_scores)
        attention_output = attention_output.view(B, C, H, W)
        out = self.gamma_cur * self.conv(attention_output) + x
             
        return out

        
class CEAF(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(CEAF, self).__init__()

        
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)), 
            nn.ReLU(inplace=True),  
            nn.Linear(int(in_channels / rate), in_channels)  
        )

        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3), 
            nn.BatchNorm2d(int(in_channels / rate)),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),  
            nn.BatchNorm2d(in_channels)  
        )

    
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x

    
    def forward(self, x):
        b, c, h, w = x.shape  
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)  
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)  
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()  
        x = x * x_channel_att  

        x = self.channel_shuffle(x, groups=4) 

        x_spatial_att = self.spatial_attention(x).sigmoid()  

        out = x * x_spatial_att  
       
        return out  


class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        
        self.CEAF1 = CEAF(32)
        self.CEAF2 = CEAF(64)
        self.CEAF3 = CEAF(64)
        self.CEAF4 = CEAF(64)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 32*11*11
        self.decoder4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  
            BasicConv2d(32, 32, 3, padding=1)  
        )
               

        # 32*22*22
        self.decoder3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  
            BasicConv2d(64, 32, 3, padding=1) 
        )
        
        # 32*44*44
        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  
            BasicConv2d(64, 32, 3, padding=1)  
        )
        
        # 32*88*88
        self.decoder1 = nn.Sequential(
            BasicConv2d(64, 32, 3, padding=1)  
        )
        
        self.conv = nn.Conv2d(channel, 1, 1)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        

    def forward(self, x4, x3, x2, x1):

        x4_cea = self.CEAF1(x4)
        x4_decoder = self.decoder4(x4_cea)  # 32*22*22

        x3_cat = torch.cat([x4_decoder, x3], 1)   # 64*22*22
        x3_cea = self.CEAF2(x3_cat)        
        x3_decoder = self.decoder3(x3_cea)

        x2_cat = torch.cat([x3_decoder, x2], 1) # 64*44*44
        x2_cea = self.CEAF3(x2_cat)        
        x2_decoder = self.decoder2(x2_cea)   # 32*88*88        

        x1_cat = torch.cat([x2_decoder, x1], 1) # 64*88*88
        x1_cea = self.CEAF4(x1_cat)        
        x1_decoder = self.decoder1(x1_cea)   # 32*88*88
        
        x = self.conv(x1_decoder) # 1*88*88
        x = self.upsample_4(x) # 1*352*352

        return x
        
        

class DBANet(nn.Module):
    def __init__(self, channel=32):
        super(DBANet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './model/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # input 3x352x352
        self.ChannelNormalization_1 = BasicConv2d(64, channel, 3, 1, 1)  # 64x88x88->32x88x88
        self.ChannelNormalization_2 = BasicConv2d(128, channel, 3, 1, 1) # 128x44x44->32x44x44
        self.ChannelNormalization_3 = BasicConv2d(320, channel, 3, 1, 1) # 320x22x22->32x22x22
        self.ChannelNormalization_4 = BasicConv2d(512, channel, 3, 1, 1) # 512x11x11->32x11x11

        self.DEA1 = DEA(64)
        self.DEA2 = DEA(128)
        self.DEA3 = DEA(320)
        self.DEA4 = DEA(512)  
        
        self.MSA1 = MSA(64)
        self.MSA2 = MSA(128)
        self.MSA3 = MSA(320)
        self.MSA4 = MSA(512)
    
        self.Decoder = Decoder(channel)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0] # 64x88x88
        x2 = pvt[1] # 128x44x44
        x3 = pvt[2] # 320x22x22
        x4 = pvt[3] # 512x11x11
        

        x1_dea = self.DEA1(x1)
        x2_dea = self.DEA2(x2)
        x3_dea = self.DEA3(x3)
        x4_dea = self.DEA4(x4)

        x1_msa = self.MSA1(x1)
        x2_msa = self.MSA2(x2)
        x3_msa = self.MSA3(x3)
        x4_msa = self.MSA4(x4)
       
        x1_all = x1_msa + x1_dea
        x2_all = x2_msa + x2_dea
        x3_all = x3_msa + x3_dea
        x4_all = x4_msa + x4_dea
        
        x1_nor = self.ChannelNormalization_1(x1_all) # 32x88x88
        x2_nor = self.ChannelNormalization_2(x2_all) # 32x44x44
        x3_nor = self.ChannelNormalization_3(x3_all) # 32x22x22
        x4_nor = self.ChannelNormalization_4(x4_all) # 32x11x11
        
        prediction = self.Decoder(x4_nor, x3_nor, x2_nor, x1_nor)


        return prediction, self.sigmoid(prediction)
