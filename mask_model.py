import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models

import res_model

# pos embed for feature, sincos init
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=196):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))

        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos = torch.arange(max_len).unsqueeze(1)
        pos_embedding = torch.zeros(max_len, d_model)
        pos_embedding[:, 0::2] = torch.sin(pos * div_term)
        pos_embedding[:, 1::2] = torch.cos(pos * div_term)
        self.pos_embedding.data.copy_(pos_embedding)

# attention fusion 
class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=512, dropout_rate=0.1):
        super(MultiheadAttentionBlock, self).__init__()

        self.multihead_attention = nn.MultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)  

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout_rate)  

    def forward(self, x):
    
        attn_output, _ = self.multihead_attention(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))  
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

# action prediction
class Predict(nn.Module):
    def __init__(self, d_model=512 + 128):
        super(Predict, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model//2),
            nn.LeakyReLU(),
            nn.Linear(d_model//2, d_model//4),
            nn.LeakyReLU(),
            nn.Linear(d_model //4, 3),
            nn.Dropout(0.),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.LeakyReLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LeakyReLU(),
            nn.Linear(d_model // 4, 2),
            nn.Dropout(0.),
        )

    def forward(self, x):
        action = self.fc1(x)
        griper = self.fc2(x)

        return action, griper


#Flatten
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# max and mean polling
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

# conv
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x

# Channel Gate
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.LeakyReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

# space gate
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


#CBAM fusion
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out



class Block(nn.Module):
    def __init__(self, indim, outdim=None):
        super(Block, self).__init__()
        self.relu = nn.LeakyReLU()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = None

        self.conv1 = nn.Conv2d(indim, outdim//16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim//16, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(self.relu(x))
        r = self.conv2(self.relu(r))

        return r


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


# two-stream fusion
class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = Block(indim, outdim)
        self.attention = CBAM(outdim)
        self.block2 = Block(outdim, outdim)

    def forward(self, x, f16):
        x = torch.cat([x, f16], 1)
        x = self.block1(x)
        r = self.attention(x)
        x = self.block2(x + r)
        return x

#rgb as input, 3 stage
class ImgEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        network = res_model.resnet50(pretrained=True)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  # 1/2, 64
        self.maxpool = network.maxpool

        self.res2 = network.layer1 # 1/4, 256
        self.layer2 = network.layer2 # 1/8, 512
        self.layer3 = network.layer3 # 1/16, 1024

    def forward(self, f):
        x = self.conv1(f)
        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        f4 = self.res2(x)   # 1/4, 256
        f8 = self.layer2(f4) # 1/8, 512
        f16 = self.layer3(f8) # 1/16, 1024
        return f16
        #return f16, f8, f4


# rgb-m and rgb as input, 3 stage two-strem fusion
# mask = rgbm
class MaskEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = res_model.resnet18(pretrained=True, extra_dim=1)

        self.image_encorder = ImgEncoder()

        self.mask_conv1 = resnet.conv1
        self.mask_bn1 = resnet.bn1
        self.mask_relu = resnet.relu  # 1/2, 64
        self.mask_maxpool = resnet.maxpool

        self.mask_layer1 = resnet.layer1 # 1/4, 64
        self.mask_layer2 = resnet.layer2 # 1/8, 128
        self.mask_layer3 = resnet.layer3 # 1/16, 256

        self.fuser = FeatureFusionBlock(1024 + 256, 512) # two-stream fusion

        self.pos_encoding = PositionalEncoding()

        self.self_attention = MultiheadAttentionBlock()
        

    def forward(self, image, mask):

        f = mask
        x = self.mask_conv1(f)
        x = self.mask_bn1(x)
        x = self.mask_relu(x)   # 1/2, 64
        x = self.mask_maxpool(x)  # 1/4, 64
        x = self.mask_layer1(x)   # 1/4, 64
        x = self.mask_layer2(x) # 1/8, 128
        mask = self.mask_layer3(x) # 1/16, 256

        img_f16 = self.image_encorder(image)

        x = self.fuser(mask, img_f16)

        flattened_tensor = torch.reshape(x, (img_f16.size(0), 512, 196)).transpose(1, 2)
        features_with_pe = flattened_tensor + self.pos_encoding.pos_embedding
        attn_output = self.self_attention(features_with_pe.transpose(0, 1)).transpose(0, 1)

        return  attn_output
        
# our policy two-view
class policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb0 = MaskEncoder()
        self.emb1 = MaskEncoder()

        self.self_attention_view = MultiheadAttentionBlock() # multiview fusion

        self.predict = Predict()

        self.embd_pose = nn.Sequential(
            nn.Linear(4, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 128),
        )

    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward(self, image0, mask0, image1, mask1, pose):

        view0 = self.emb0(image0, mask0)

        view1 = self.emb1(image1, mask1)

        features = torch.cat([view0, view1], dim=1)

        attn_output = self.self_attention_view(features.transpose(0, 1)).transpose(0, 1)

        avg_features = attn_output.mean(dim=1)  

        pose = self.embd_pose(pose.to(torch.float32))
        avg_features = torch.cat([avg_features, pose], dim=1)

        action, binary_class = self.predict(avg_features)

        return action, binary_class


