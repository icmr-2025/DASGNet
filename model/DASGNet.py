import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax, Dropout
from functools import partial

import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply
from mmengine.model import constant_init
from einops import rearrange
import typing as t


from typing import List, Callable
from torch import Tensor

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,activation='relu'):
        super(BasicConv2d, self).__init__()
        self.activation=activation
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

#MDAM
class MDAM(nn.Module):
    def __init__(self, in_ch, out_ch, spatial_kernel_sizes=[3, 5, 7, 9],channel_kernel_sizes=[7, 11, 21],heads = 8):
        super(MDAM,self).__init__()
        self.dsab = DSAB(in_ch,spatial_kernel_sizes)
        self.dcsab = DCSAB(in_ch,channel_kernel_sizes,heads)
        self.mcb = MCB(in_ch, out_ch)
      
    def forward(self, x):
        x = self.dsab(x)
        x = self.dcsab(x)
        x = self.mcb(x)
        return x
    
class DSAB(nn.Module):
    def __init__(self, in_ch,spatial_kernel_sizes):
        super(DSAB,self).__init__()
        self.in_ch=in_ch 
        self.dconv1d3 = nn.Conv1d(self.in_ch//4, self.in_ch//4, kernel_size=spatial_kernel_sizes[0],
                                   padding=spatial_kernel_sizes[0]// 2, groups=self.in_ch//4)
        self.dconv1d5 = nn.Conv1d(self.in_ch//4, self.in_ch//4, kernel_size=spatial_kernel_sizes[1],
                                      padding=spatial_kernel_sizes[1] // 2, groups=self.in_ch//4)
        self.dconv1d7 = nn.Conv1d(self.in_ch//4, self.in_ch//4, kernel_size=spatial_kernel_sizes[2],
                                      padding=spatial_kernel_sizes[2] // 2, groups=self.in_ch//4)
        self.dconv1d9 = nn.Conv1d(self.in_ch//4, self.in_ch//4, kernel_size=spatial_kernel_sizes[3],
                                      padding=spatial_kernel_sizes[3] // 2, groups=self.in_ch//4)
        self.hnorm = nn.GroupNorm(4, in_ch)
        self.wnorm = nn.GroupNorm(4, in_ch)
        self.sig = nn.Sigmoid()

      
    def forward(self, x):
        b, c, h, w = x.size()
        hx = x.mean(dim=3)
        hx1,hx2,hx3,hx4 = torch.split(hx, self.in_ch//4, dim=1)
        wx = x.mean(dim=2)
        wx1,wx2,wx3,wx4 = torch.split(wx, self.in_ch//4, dim=1)

        hx1=self.dconv1d3(hx1)*self.sig(self.dconv1d3(hx1))
        hx2=self.dconv1d3(hx2)*self.sig(self.dconv1d3(hx2))
        hx3=self.dconv1d7(hx3)*self.sig(self.dconv1d7(hx3))
        hx4=self.dconv1d7(hx4)*self.sig(self.dconv1d7(hx4))
        
        hx1=self.dconv1d5(hx1)*self.sig(self.dconv1d5(hx1))
        hx2=self.dconv1d5(hx2)*self.sig(self.dconv1d5(hx2))
        hx3=self.dconv1d9(hx3)*self.sig(self.dconv1d9(hx3))
        hx4=self.dconv1d9(hx4)*self.sig(self.dconv1d9(hx4))

        hx=torch.cat((hx1,hx2,hx3,hx4),dim=1)
        hx_attn=self.sig(self.hnorm(hx))

        hx_attn = hx_attn.view(b, c, h, 1)

        wx1=self.dconv1d3(wx1)*self.sig(self.dconv1d3(wx1))
        wx2=self.dconv1d3(wx2)*self.sig(self.dconv1d3(wx2))
        wx3=self.dconv1d7(wx3)*self.sig(self.dconv1d7(wx3))
        wx4=self.dconv1d7(wx4)*self.sig(self.dconv1d7(wx4))
        
        wx1=self.dconv1d5(wx1)*self.sig(self.dconv1d5(wx1))
        wx2=self.dconv1d5(wx2)*self.sig(self.dconv1d5(wx2))
        wx3=self.dconv1d9(wx3)*self.sig(self.dconv1d9(wx3))
        wx4=self.dconv1d9(wx4)*self.sig(self.dconv1d9(wx4))

        wx=torch.cat((wx1,wx2,wx3,wx4),dim=1)
        wx_attn=self.sig(self.wnorm(wx))

        wx_attn = wx_attn.view(b, c, 1, w)
        
        x = x * hx_attn * wx_attn
        
        return x

class DCSAB(nn.Module):
    def __init__(self, in_ch,channel_kernel_sizes=[7, 11, 21],heads = 8):
        super(DCSAB,self).__init__()
        self.in_ch=in_ch
        self.heads = heads
        self.head_dim = in_ch // heads
        self.scaler=self.head_dim ** -0.5 
        self.sig =nn.Sigmoid()

        self.downsampling = nn.AvgPool2d(kernel_size=(7, 7), stride=7)
        self.norm = nn.GroupNorm(1, in_ch)

        self.q_h=nn.Conv2d(in_ch, in_ch, kernel_size=(channel_kernel_sizes[0],1), padding=(3, 0),bias=False, groups=in_ch)
        self.k_h=nn.Conv2d(in_ch, in_ch, kernel_size=(channel_kernel_sizes[1],1), padding=(5, 0),bias=False, groups=in_ch)
        self.v_h=nn.Conv2d(in_ch, in_ch, kernel_size=(channel_kernel_sizes[2],1), padding=(10, 0),bias=False, groups=in_ch)
        self.q_w=nn.Conv2d(in_ch, in_ch, kernel_size=(1, channel_kernel_sizes[0]), padding=(0, 3),bias=False, groups=in_ch)
        self.k_w=nn.Conv2d(in_ch, in_ch, kernel_size=(1, channel_kernel_sizes[1]), padding=(0, 5),bias=False, groups=in_ch)
        self.v_w=nn.Conv2d(in_ch, in_ch, kernel_size=(1, channel_kernel_sizes[2]), padding=(0, 10),bias=False, groups=in_ch)
      
    def forward(self, x):
        y = self.downsampling(x)
        _, _, h, w = y.size()

        y = self.norm(y)
        q_h = self.q_h(y)
        k_h = self.k_h(y)
        v_h = self.v_h(y)
        q_w = self.q_w(y)
        k_w = self.k_w(y)
        v_w = self.v_w(y)

        q_h = rearrange(q_h, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        k_h = rearrange(k_h, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        v_h = rearrange(v_h, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)

        attn_h = q_h @ k_h.transpose(-2, -1)
        attn_h = attn_h * self.scaler
        attn_h = attn_h.softmax(dim=-1)
        attn_h = attn_h @ v_h
        attn_h = rearrange(attn_h, 'b heads head_dim (h w) -> b (heads head_dim) h w', h=h, w=w)

        q_w = rearrange(q_w, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        k_w = rearrange(k_w, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        v_w = rearrange(v_w, 'b (heads head_dim) h w -> b heads head_dim (h w)', heads=self.heads,
                      head_dim=self.head_dim)
        
        attn_w = q_w @ k_w.transpose(-2, -1)
        attn_w = attn_w * self.scaler 
        attn_w = attn_w.softmax(dim=-1)
    
        attn_w = attn_w @ v_w
        attn_w = rearrange(attn_w, 'b heads head_dim (h w) -> b (heads head_dim) h w', h=h, w=w)
        attn=attn_h+attn_w
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.sig(attn)
        return attn * x

class MCB(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_sizes=[1,3,5]):
        super(MCB, self).__init__()
        self.conv1 = BasicConv2d(in_planes=in_ch, out_planes=in_ch*6, kernel_size=1)
        self.mcbconv = mcbconv(in_ch*6,kernel_sizes)
        self.pwconv = BasicConv2d(in_ch*6, out_ch, kernel_size=1,activation=None)
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        c1 = self.conv1(x)
        msdc1 = self.mcbconv(c1)
        msdc1 = channel_shuffle(msdc1, 32)
        out = self.pwconv(msdc1)
        return x + out
    
class mcbconv(nn.Module):
    def __init__(self, in_ch, kernel_sizes):
        super(mcbconv, self).__init__()
        self.dconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size,1, kernel_size // 2, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU6(inplace=True)
            )
            for kernel_size in kernel_sizes
        ])
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x1,x2,x3=self.dconvs[0](x),self.dconvs[1](x),self.dconvs[2](x)
        return x1+x2+x3

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

#SERM
class SERM(nn.Module):

    def __init__(self, x1_ch, x2_ch, x3_ch, x4_ch, out_ch):
        super(SERM, self).__init__()
        self.msib = MSIB(x2_ch, x3_ch, x4_ch, x1_ch)
        self.agfb=AGFB(in_ch = x1_ch)
        self.desfb=DESFB(in_ch= x1_ch//2, out_ch= x1_ch//2) 

    def forward(self, x1, x2, x3, x4):
        semantic_feature=self.msib(x2, x3, x4)
        semantic_feature=F.interpolate(semantic_feature, scale_factor=2, mode='bilinear', align_corners=False)
        edge_feature, semantic_feature= self.agfb(x1,semantic_feature)
        out=self.desfb(edge_feature, semantic_feature)
        return out
    
class DESFB(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DESFB,self).__init__()
        self.MaxP = nn.AdaptiveMaxPool2d(1)
        self.AvgP = nn.AdaptiveAvgPool2d(1)
        self.conv1 =nn.Conv2d(in_ch, out_ch, kernel_size = 1, padding = 0, dilation = 1, bias=False)
        self.conv2 =nn.Conv2d(in_ch, out_ch, kernel_size = 1, padding = 0, dilation = 1, bias=False)
        self.Sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 2)
        self.conv3 = BasicConv2d(in_ch, out_ch, 1)

    def forward(self, edge_feature, semantic_feature):
        edge_weight = self.conv1(self.MaxP(edge_feature))
        semantic_weight = self.conv2(self.AvgP(semantic_feature))    

        weight = torch.cat([edge_weight,semantic_weight],2)           
        weight = self.softmax(self.Sigmoid(weight))

        edge_weight = torch.unsqueeze(weight[:,:,0],2)         
        semantic_weight = torch.unsqueeze(weight[:,:,1],2)

        out=edge_feature*edge_weight+semantic_feature*semantic_weight           
        out=self.conv3(out)
        return out
    
class MSIB(nn.Module):
    def __init__(self, x2_ch, x3_ch, x4_ch, out_ch):
        super(MSIB, self).__init__()
        self.conv1 = BasicConv2d(x2_ch, out_ch, kernel_size=1,activation='silu')
        self.conv2 = BasicConv2d(x3_ch, out_ch, kernel_size=1,activation='silu')
        self.conv3 = BasicConv2d(x4_ch, out_ch, kernel_size=1,activation='silu')
        self.conv3d = nn.Conv3d(out_ch, out_ch, kernel_size=(3, 3, 3),padding=1)
        self.bn = nn.BatchNorm3d(out_ch)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.maxpool_3d = nn.MaxPool3d(kernel_size=(3, 1, 1))
        self.avgpool_3d = nn.AvgPool3d(kernel_size=(3, 1, 1))

    def forward(self, x2, x3, x4):
        x2 = self.conv1(x2)
        x3 = self.conv2(x3)
        x3 = F.interpolate(x3, x2.size()[2:], mode='nearest')
        x4 = self.conv3(x4)
        x4 = F.interpolate(x4, x2.size()[2:], mode='nearest')
        x2_3d = torch.unsqueeze(x2, -3)
        x3_3d = torch.unsqueeze(x3, -3)
        x4_3d = torch.unsqueeze(x4, -3)
        x_fuse = torch.cat([x2_3d, x3_3d, x4_3d], dim=2)
        x_fuse_3d = self.conv3d(x_fuse)
        x_fuse_bn = self.bn(x_fuse_3d)
        x_act = self.leakyrelu(x_fuse_bn)
        x_act=x_act+x_fuse
        x = self.maxpool_3d(x_act)+self.avgpool_3d(x_act)
        x = torch.squeeze(x, 2)
        return x

class AGFB(nn.Module):
    def __init__(self, in_ch):
        super(AGFB,self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False)
        self.Sigmoid = nn.Sigmoid()
        self.conv3 = BasicConv2d(in_ch // 2, in_ch // 2, 1)
        self.conv4 = BasicConv2d(in_ch // 2, in_ch // 2, 1)
        self.w1 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.silu = nn.SiLU(inplace=True)
        self.conv5 = nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=1, stride=1, padding=0)
      
    def forward(self, x1, semantic_feature):
        edge_feature = self.conv1(x1)
        semantic_feature = self.conv2(semantic_feature)
        
        edge_feature_sig = self.Sigmoid(edge_feature)          
        semantic_feature_sig = self.Sigmoid(semantic_feature)

        edge_feature = self.conv3(edge_feature)              
        semantic_feature = self.conv4(semantic_feature)

        w1 = self.w1
        w2 = self.w2
        
        weight1 = w1 / (torch.sum(w1, dim=0) + self.epsilon)
        weight2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)

        edge_feature = self.conv5(self.silu(weight1[0]*edge_feature + weight1[1]*(edge_feature * edge_feature_sig) + weight1[2]*((1 - edge_feature_sig) * semantic_feature_sig * semantic_feature)))
        semantic_feature = self.conv6(self.silu(weight2[0]*semantic_feature + weight2[1]*(semantic_feature * semantic_feature_sig) + weight2[2]*((1 - semantic_feature_sig) * edge_feature_sig * edge_feature)))

        return edge_feature, semantic_feature

class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(True)                       
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1,activation=None)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1,activation=None)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1,activation=None)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1,activation=None)
        self.conv_upsample5 = BasicConv2d(channel, channel, 3, padding=1,activation=None)
        self.conv_upsample6 = BasicConv2d(channel, channel, 3, padding=1,activation=None)
        self.conv_upsample7 = BasicConv2d(channel, channel, 3, padding=1,activation=None)
        self.conv_upsample8 = BasicConv2d(channel, channel, 3, padding=1,activation=None)
        self.conv_upsample9 = BasicConv2d(channel, channel, 3, padding=1,activation=None)
        self.conv_upsample10 = BasicConv2d(channel, channel, 3, padding=1,activation=None)
        self.conv_upsample11 = BasicConv2d(channel, channel, 3, padding=1,activation=None)
        self.conv_upsample12 = BasicConv2d(2*channel, 2*channel, 3, padding=1,activation=None)
        self.conv_upsample13 = BasicConv2d(3*channel, 3*channel, 3, padding=1,activation=None)
        self.conv_upsample14 = BasicConv2d(4*channel, 4*channel, 3, padding=1,activation=None)
        self.conv_concat1 = BasicConv2d(2*channel, 2*channel, 3, padding=1,activation=None)
        self.conv_concat2 = BasicConv2d(3*channel, 3*channel, 3, padding=1,activation=None)
        self.conv_concat3 = BasicConv2d(4*channel, 4*channel, 3, padding=1,activation=None)
        self.conv_concat4 = BasicConv2d(5*channel, 5*channel, 3, padding=1,activation=None)
        self.conv4 = BasicConv2d(5*channel, 5*channel, 3, padding=1,activation=None)
        self.conv5 = nn.Conv2d(5*channel, 1, 1)

    def forward(self, x1, x2, x3,x4,x5): 
        x1_1 = x1 
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2     
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3     
        x4_1 = self.conv_upsample4(self.upsample(self.upsample(self.upsample(x1)))) * self.conv_upsample5(self.upsample(self.upsample(x2))) * self.conv_upsample6(self.upsample(x3))*x4   
        x5_1 = self.conv_upsample7(self.upsample(self.upsample(self.upsample(x1)))) * self.conv_upsample8(self.upsample(self.upsample(x2))) *self.conv_upsample9(self.upsample(x3))*self.conv_upsample10(x4)*x5           

        x2_2 = torch.cat((x2_1, self.conv_upsample11(self.upsample(x1_1))), 1) 
        x2_2 = self.conv_concat1(x2_2) 

        x3_2 = torch.cat((x3_1, self.conv_upsample12(self.upsample(x2_2))), 1) 
        x3_2 = self.conv_concat2(x3_2) 

        x4_2 = torch.cat((x4_1, self.conv_upsample13(self.upsample(x3_2))), 1) 
        x4_2 = self.conv_concat3(x4_2) 

        x5_2 = torch.cat((x5_1, self.conv_upsample14(x4_2)), 1) 
        x5_2 = self.conv_concat4(x5_2) 

        x = self.conv4(x5_2) 
        x = self.conv5(x) 
        return x


class DASGNet(nn.Module):
    def __init__(self, channel=32):
        super(DASGNet, self).__init__()

        self.backbone = pvt_v2_b2()
        path = './model/pvt_v2_b2.pth'       
        save_model = torch.load(path) 
        model_dict = self.backbone.state_dict()         
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}           
        model_dict.update(state_dict)                   
        self.backbone.load_state_dict(model_dict)       

        self.ChannelReduction_1 = BasicConv2d(64, channel, 3, 1, 1, activation=None)          
        self.ChannelReduction_2 = BasicConv2d(128, channel, 3, 1, 1, activation=None)
        self.ChannelReduction_3 = BasicConv2d(320, channel, 3, 1, 1, activation=None) 
        self.ChannelReduction_4 = BasicConv2d(512, channel, 3, 1, 1, activation=None) 

        self.mdam1=MDAM(in_ch=32, out_ch=32)
        self.mdam2=MDAM(in_ch=32, out_ch=32)
        self.mdam3=MDAM(in_ch=32, out_ch=32)   
        self.mdam4=MDAM(in_ch=32, out_ch=32)

        self.sba = SERM(x1_ch=64, x2_ch =128, x3_ch=320, x4_ch=512, out_ch=32)

        self.decoder = Decoder(channel)   
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)          
        self.sigmoid = nn.Sigmoid()        


    def forward(self, x):

        pvt = self.backbone(x)
        x1 = pvt[0] 
        x2 = pvt[1]
        x3 = pvt[2] 
        x4 = pvt[3] 

        x5=self.sba(x1,x2,x3,x4)           

        x1_cr = self.ChannelReduction_1(x1) 
        x2_cr = self.ChannelReduction_2(x2) 
        x3_cr = self.ChannelReduction_3(x3) 
        x4_cr = self.ChannelReduction_4(x4) 

        x1=self.mdam1(x1_cr)
        x2=self.mdam2(x2_cr)     
        x3=self.mdam3(x3_cr)
        x4=self.mdam4(x4_cr)

        prediction = self.upsample_4(self.decoder(x4, x3, x2, x1, x5))
        
        return prediction, self.sigmoid(prediction)             