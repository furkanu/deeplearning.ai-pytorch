import torch
import torch.nn as nn
import torch.nn.functional as F


WEIGHTS = [
  'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3',
  'inception_3a_1x1_conv', 'inception_3a_1x1_bn',
  'inception_3a_pool_conv', 'inception_3a_pool_bn',
  'inception_3a_5x5_conv1', 'inception_3a_5x5_conv2', 'inception_3a_5x5_bn1', 'inception_3a_5x5_bn2',
  'inception_3a_3x3_conv1', 'inception_3a_3x3_conv2', 'inception_3a_3x3_bn1', 'inception_3a_3x3_bn2',
  'inception_3b_3x3_conv1', 'inception_3b_3x3_conv2', 'inception_3b_3x3_bn1', 'inception_3b_3x3_bn2',
  'inception_3b_5x5_conv1', 'inception_3b_5x5_conv2', 'inception_3b_5x5_bn1', 'inception_3b_5x5_bn2',
  'inception_3b_pool_conv', 'inception_3b_pool_bn',
  'inception_3b_1x1_conv', 'inception_3b_1x1_bn',
  'inception_3c_3x3_conv1', 'inception_3c_3x3_conv2', 'inception_3c_3x3_bn1', 'inception_3c_3x3_bn2',
  'inception_3c_5x5_conv1', 'inception_3c_5x5_conv2', 'inception_3c_5x5_bn1', 'inception_3c_5x5_bn2',
  'inception_4a_3x3_conv1', 'inception_4a_3x3_conv2', 'inception_4a_3x3_bn1', 'inception_4a_3x3_bn2',
  'inception_4a_5x5_conv1', 'inception_4a_5x5_conv2', 'inception_4a_5x5_bn1', 'inception_4a_5x5_bn2',
  'inception_4a_pool_conv', 'inception_4a_pool_bn',
  'inception_4a_1x1_conv', 'inception_4a_1x1_bn',
  'inception_4e_3x3_conv1', 'inception_4e_3x3_conv2', 'inception_4e_3x3_bn1', 'inception_4e_3x3_bn2',
  'inception_4e_5x5_conv1', 'inception_4e_5x5_conv2', 'inception_4e_5x5_bn1', 'inception_4e_5x5_bn2',
  'inception_5a_3x3_conv1', 'inception_5a_3x3_conv2', 'inception_5a_3x3_bn1', 'inception_5a_3x3_bn2',
  'inception_5a_pool_conv', 'inception_5a_pool_bn',
  'inception_5a_1x1_conv', 'inception_5a_1x1_bn',
  'inception_5b_3x3_conv1', 'inception_5b_3x3_conv2', 'inception_5b_3x3_bn1', 'inception_5b_3x3_bn2',
  'inception_5b_pool_conv', 'inception_5b_pool_bn',
  'inception_5b_1x1_conv', 'inception_5b_1x1_bn',
  'dense_layer'
]

conv_shape = {
  'conv1': [64, 3, 7, 7],
  'conv2': [64, 64, 1, 1],
  'conv3': [192, 64, 3, 3],
  'inception_3a_1x1_conv': [64, 192, 1, 1],
  'inception_3a_pool_conv': [32, 192, 1, 1],
  'inception_3a_5x5_conv1': [16, 192, 1, 1],
  'inception_3a_5x5_conv2': [32, 16, 5, 5],
  'inception_3a_3x3_conv1': [96, 192, 1, 1],
  'inception_3a_3x3_conv2': [128, 96, 3, 3],
  'inception_3b_3x3_conv1': [96, 256, 1, 1],
  'inception_3b_3x3_conv2': [128, 96, 3, 3],
  'inception_3b_5x5_conv1': [32, 256, 1, 1],
  'inception_3b_5x5_conv2': [64, 32, 5, 5],
  'inception_3b_pool_conv': [64, 256, 1, 1],
  'inception_3b_1x1_conv': [64, 256, 1, 1],
  'inception_3c_3x3_conv1': [128, 320, 1, 1],
  'inception_3c_3x3_conv2': [256, 128, 3, 3],
  'inception_3c_5x5_conv1': [32, 320, 1, 1],
  'inception_3c_5x5_conv2': [64, 32, 5, 5],
  'inception_4a_3x3_conv1': [96, 640, 1, 1],
  'inception_4a_3x3_conv2': [192, 96, 3, 3],
  'inception_4a_5x5_conv1': [32, 640, 1, 1,],
  'inception_4a_5x5_conv2': [64, 32, 5, 5],
  'inception_4a_pool_conv': [128, 640, 1, 1],
  'inception_4a_1x1_conv': [256, 640, 1, 1],
  'inception_4e_3x3_conv1': [160, 640, 1, 1],
  'inception_4e_3x3_conv2': [256, 160, 3, 3],
  'inception_4e_5x5_conv1': [64, 640, 1, 1],
  'inception_4e_5x5_conv2': [128, 64, 5, 5],
  'inception_5a_3x3_conv1': [96, 1024, 1, 1],
  'inception_5a_3x3_conv2': [384, 96, 3, 3],
  'inception_5a_pool_conv': [96, 1024, 1, 1],
  'inception_5a_1x1_conv': [256, 1024, 1, 1],
  'inception_5b_3x3_conv1': [96, 736, 1, 1],
  'inception_5b_3x3_conv2': [384, 96, 3, 3],
  'inception_5b_pool_conv': [96, 736, 1, 1],
  'inception_5b_1x1_conv': [256, 736, 1, 1],
}



class BottleneckedConv(nn.Module):
    def __init__(self, in_channels, out_channels1,
                 out_channels2, kernel_size2, stride2=1, padding2=0):
                                    
        super(BottleneckedConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels1, 1) #bottleneck layer
        self.bn1 = nn.BatchNorm2d(out_channels1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size2, stride2, padding2)
        self.bn2 = nn.BatchNorm2d(out_channels2)
        self.relu2 = nn.ReLU()
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
class Conv_1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
class PoolLayer(nn.Module):
    def __init__(self, pool_type, pool_kernel_size, pool_stride,
                  in_channels, out_channels, padding):
        super(PoolLayer, self).__init__()
        if pool_type == 'max':
            self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride)
        elif pool_type == 'average':
            self.pool = nn.AvgPool2d(pool_kernel_size, pool_stride)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) #channel reduce
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.zeropad = nn.ZeroPad2d(padding)
        
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.zeropad(x)
        
        return x
class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)



class inception_block_1a(nn.Module):
    def __init__(self, in_channels):
        super(inception_block_1a, self).__init__()
        
        self.inception_3a_3x3 = BottleneckedConv(in_channels, 96, 128, 3, padding2=1)
        self.inception_3a_5x5 = BottleneckedConv(in_channels, 16, 32, 5, padding2=2)
        self.inception_3a_pool = PoolLayer('max', 3, 2, in_channels, 32, (3, 4, 3, 4))
        self.inception_3a_1x1 = Conv_1x1(in_channels, 64)
    def forward(self, x):
        x_3x3 = self.inception_3a_3x3(x)
        x_5x5 = self.inception_3a_5x5(x)
        x_pool = self.inception_3a_pool(x)
        x_1x1 = self.inception_3a_1x1(x)
        
        return torch.cat([x_3x3, x_5x5, x_pool, x_1x1], dim=1)



class inception_block_1b(nn.Module):
    def __init__(self, in_channels):
        super(inception_block_1b, self).__init__()
        self.inception_3b_3x3 = BottleneckedConv(in_channels, 96, 128, 3, padding2=1)
        self.inception_3b_5x5 = BottleneckedConv(in_channels, 32, 64, 5, padding2=2)
        self.inception_3b_pool = PoolLayer('average', 3, 3, in_channels, 64, 4)
        self.inception_3b_1x1 = Conv_1x1(in_channels, 64)
    
    def forward(self, x):
        x_3x3 = self.inception_3b_3x3(x)
        x_5x5 = self.inception_3b_5x5(x)
        x_pool = self.inception_3b_pool(x)
        x_1x1 = self.inception_3b_1x1(x)
        
        return torch.cat([x_3x3, x_5x5, x_pool, x_1x1], dim=1)



class inception_block_1c(nn.Module):
    def __init__(self, in_channels):
        super(inception_block_1c, self).__init__()
        self.inception_3c_3x3 = BottleneckedConv(in_channels, 128, 256, 3, 2, 1)
        self.inception_3c_5x5 = BottleneckedConv(in_channels, 32, 64, 5, 2, 2)
        
        self.inception_3c_pool_pool = nn.MaxPool2d(3, 2)
        self.inception_3c_pool_zeropad = nn.ZeroPad2d((0, 1, 0, 1))
    
    def forward(self, x):
        x_3x3 = self.inception_3c_3x3(x)
        x_5x5 = self.inception_3c_5x5(x)
        
        pool = self.inception_3c_pool_pool(x)
        pool = self.inception_3c_pool_zeropad(pool)
        
        return torch.cat([x_3x3, x_5x5, pool], dim=1)



class inception_block_2a(nn.Module):
    def __init__(self, in_channels):
        super(inception_block_2a, self).__init__()
        self.inception_4a_3x3 = BottleneckedConv(in_channels, 96, 192, 3, 1, 1)
        self.inception_4a_5x5 = BottleneckedConv(in_channels, 32, 64, 5, 1, 2)
        self.inception_4a_pool = PoolLayer('average', 3, 3, in_channels, 128, 2)
        self.inception_4a_1x1 = Conv_1x1(in_channels, 256)
        
    def forward(self, x):
        x_3x3 = self.inception_4a_3x3(x)
        x_5x5 = self.inception_4a_5x5(x)
        x_pool = self.inception_4a_pool(x)
        x_1x1 = self.inception_4a_1x1(x)
        
        return torch.cat([x_3x3, x_5x5, x_pool, x_1x1], dim=1)


class inception_block_2b(nn.Module):
    def __init__(self, in_channels):
        super(inception_block_2b, self).__init__()
        self.inception_4e_3x3 = BottleneckedConv(in_channels, 160, 256, 3, 2, 1)
        self.inception_4e_5x5 = BottleneckedConv(in_channels, 64, 128, 5, 2, 2)
        
        self.inception_4e_pool_pool = nn.MaxPool2d(3, 2)
        self.inception_4e_pool_zeropad = nn.ZeroPad2d((0, 1, 0, 1))
    def forward(self, x):
        x_3x3 = self.inception_4e_3x3(x)
        x_5x5 = self.inception_4e_5x5(x)
        
        pool = self.inception_4e_pool_pool(x)
        pool = self.inception_4e_pool_zeropad(pool)
        
        return torch.cat([x_3x3, x_5x5, pool], dim=1)


class inception_block_3a(nn.Module):
    def __init__(self, in_channels):
        super(inception_block_3a, self).__init__()
        self.inception_5a_3x3 = BottleneckedConv(in_channels, 96, 384, 3, 1, 1)
        self.inception_5a_pool = PoolLayer('average', 3, 3, in_channels, 96, 1)
        self.inception_5a_1x1 = Conv_1x1(in_channels, 256)
    
    def forward(self, x):
        x_3x3 = self.inception_5a_3x3(x)
        x_pool = self.inception_5a_pool(x)
        x_1x1 = self.inception_5a_1x1(x)
        
        return torch.cat([x_3x3, x_pool, x_1x1], dim=1)



class inception_block_3b(nn.Module):
    def __init__(self, in_channels):
        super(inception_block_3b, self).__init__()
        self.inception_5b_3x3 = BottleneckedConv(in_channels, 96, 384, 3, 1, 1)
        self.inception_5b_pool = PoolLayer('max', 3, 2, in_channels, 96, 1)
        self.inception_5b_1x1 = Conv_1x1(in_channels, 256)
        
    def forward(self, x):
        x_3x3 = self.inception_5b_3x3(x)
        x_pool = self.inception_5b_pool(x)
        x_1x1 = self.inception_5b_1x1(x)
        
        return torch.cat([x_3x3, x_pool, x_1x1], dim=1)



class faceRecoModel(nn.Module):
    def __init__(self):
        super(faceRecoModel, self).__init__()
        #First block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3) #should output 64x48x48
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        
        #Zero-Padding + MAXPOOL
        self.zeropad1 = nn.ZeroPad2d(1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        #Second block
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        #Zero-Padding
        self.zeropad2 = nn.ZeroPad2d(1)
        
        #Third block
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(192)
        self.relu3 = nn.ReLU()
        
        #Zero-Padding + MAXPOOL
        self.zeropad3 = nn.ZeroPad2d(1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Inception 1: a/b/c
        self.block_1a = inception_block_1a(192)
        self.block_1b = inception_block_1b(256)
        self.block_1c = inception_block_1c(320)
        
        # Inception 2: a/b
        self.block_2a = inception_block_2a(640)
        self.block_2b = inception_block_2b(640)
        
        # Inception 3: a/b
        self.block_3a = inception_block_3a(1024)
        self.block_3b= inception_block_3b(736)
        
        # Top layer
        self.average_pool = nn.AvgPool2d(3, 1)
        self.flatten = Flatten()
        self.dense_layer = nn.Linear(736, 128)
        
        
    def forward(self, x):
        #First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        #Zero-Padding + MAXPOOL
        x = self.zeropad1(x)
        x = self.pool1(x)
        
        #Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        #Zero-Padding
        x = self.zeropad2(x)
        
        #Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        #Zero-Padding + MAXPOOL
        x = self.zeropad3(x)
        x = self.pool3(x)
        
        x = self.block_1a(x)
        x = self.block_1b(x)
        x = self.block_1c(x)
        x = self.block_2a(x)
        x = self.block_2b(x)
        x = self.block_3a(x)
        x = self.block_3b(x)
        
        # Top layer
        x = self.average_pool(x)
        x = self.flatten(x)
        x = self.dense_layer(x)
        
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)
        
        return x


import numpy as np
from numpy import genfromtxt
import os
def load_weights():
    # Set weights path
    dirPath = './weights'
    fileNames = filter(lambda f: not f.startswith('.'), os.listdir(dirPath))
    paths = {}
    weights_dict = {}

    for n in fileNames:
        paths[n.replace('.csv', '')] = dirPath + '/' + n

    for name in WEIGHTS:
        if 'conv' in name:
            conv_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
            conv_w = np.reshape(conv_w, conv_shape[name])
            #conv_w = np.transpose(conv_w, (2, 3, 1, 0))
            conv_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
            weights_dict[name] = [conv_w, conv_b]     
        elif 'bn' in name:
            bn_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
            bn_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
            bn_m = genfromtxt(paths[name + '_m'], delimiter=',', dtype=None)
            bn_v = genfromtxt(paths[name + '_v'], delimiter=',', dtype=None)
            weights_dict[name] = [bn_w, bn_b, bn_m, bn_v]
        elif 'dense' in name:
            dense_w = genfromtxt(dirPath+'/dense_w.csv', delimiter=',', dtype=None)
            dense_w = np.reshape(dense_w, (128, 736))
            #dense_w = np.transpose(dense_w, (1, 0))
            dense_b = genfromtxt(dirPath+'/dense_b.csv', delimiter=',', dtype=None)
            weights_dict[name] = [dense_w, dense_b]

    return weights_dict



def find_state_dict_keys(layer_dict, weight_key):
    keys = []
    for key, val in layer_dict.items():
        if (weight_key.startswith('in') and not val.startswith('bl')) or            (not weight_key.startswith('in') and val.startswith('bl')):
                continue
        if weight_key in val:
            keys.append(key)
            
    return keys



def fill_conv(weights_dict, state_dict, weight_key, conv_dict):
    #print('*'*40 + 'Running fill_conv' + '*'*40)
    conv_weight, conv_bias = weights_dict[weight_key]
    keys = find_state_dict_keys(conv_dict, weight_key)
    
    state_dict[keys[0]] = torch.from_numpy(conv_weight)
    state_dict[keys[1]] = torch.from_numpy(conv_bias)
    
    #for key in keys:
    #    print('original:', weight_key, 'matched:', key)




def fill_bn(weights_dict, state_dict, weight_key, bn_dict):
    #print('*'*40 + 'Running fill_bn' + '*'*40)
    weight, bias, mean, variance = weights_dict[weight_key]
    keys = find_state_dict_keys(bn_dict, weight_key)
    
    state_dict[keys[0]] = torch.from_numpy(weight)
    state_dict[keys[1]] = torch.from_numpy(bias)
    state_dict[keys[2]] = torch.from_numpy(mean)
    state_dict[keys[3]] = torch.from_numpy(variance)
            
    #for key in keys:
    #    print('original:', weight_key, 'matched:', key)



def fill_dense(weights_dict, state_dict, weight_key, dense_dict):
    #print('*'*40 + 'Running fill_dense' + '*'*40)
    weight, bias = weights_dict[weight_key]
    keys = [key for key, val in dense_dict.items() if weight_key in val]
    
    state_dict[keys[0]] = torch.from_numpy(weight)
    state_dict[keys[1]] = torch.from_numpy(bias)
    
    #for key in keys:
    #    print('original:', weight_key, 'matched:', key)




def fill_state_dict(weights_dict, conv_dict, bn_dict, dense_dict, state_dict):
    for name in weights_dict.keys():
        if 'conv' in name:
            fill_conv(weights_dict, state_dict, name, conv_dict)
        elif 'bn' in name:
            fill_bn(weights_dict, state_dict, name, bn_dict)
        elif 'dense' in name:
            fill_dense(weights_dict, state_dict, name, dense_dict)
            



def load_weights_from_FaceNet(model):
    weights_dict = load_weights()
    
    state_dict = {}
    state_dict_keys = model.state_dict().keys()

    conv_dict = {x:x.replace('.', '_') for x in state_dict_keys if 'conv' in x}
    bn_dict = {x:x.replace('.', '_') for x in state_dict_keys if 'bn' in x}
    dense_dict = {x:x.replace('.', '_') for x in state_dict_keys if 'dense' in x}

    fill_state_dict(weights_dict, conv_dict, bn_dict, dense_dict, state_dict)
    
    model.load_state_dict(state_dict, strict=True)


