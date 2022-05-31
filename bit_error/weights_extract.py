    
import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from models_extension import resnet18
from models_extension import LeNet5
from models_extension import vgg16
# from resnet_approx1 import resnet18
import random
import torchvision
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
# import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import Module
import pandas as pd
from csv import writer
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import torchvision.models as models
import numpy as np
import random
from float_to_positf import fposit


model = LeNet5()
model.to('cpu')
checkpoint = torch.load(f'/home/arpita/Approximate-Computing-for-Neural-Networks/models/lenet_mnist', map_location = torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
d = model.state_dict()

check_words = ['bias', 'bn', 'downsample', 'classifier']

# for name, param in model.named_parameters():
#     print('name: ', name)
#     print(type(param))
#     print('param.shape: ',param.shape )
#     # print('param.required_grad: ', param.required_grad)
#     print('=======')
# # model.save_weights(savedfile = 'resnet18.weights', cutoff = 0)
# wt = open('resnet18_weights.csv', 'w')
# bs = open('resnet18_biases.csv', 'w')

random.seed()
bit_error_prob = float(sys.argv[1])

posit_length = 9
posit_regime = 2
posit_es = 4
posit = fposit(posit_length,  posit_regime, posit_es)

with torch.no_grad():
    for name, param in model.named_parameters():
        temp = any(i in name for i in check_words)
        if(temp == False):
            # print("Hi")
            weights = param.detach().numpy()
            weight_size = np.shape(weights)
            # print(weight_size)
            weights = weights.flatten()

            for i in range(len(weights)):

                bit_flip_true = random.choices([0,1],weights=[1-bit_error_prob, bit_error_prob])
                # print(bit_flip_true)
                if(bit_flip_true[0] == 1):
                    # print('yes')
                    bit_pos = random.randint(0,posit_length-1)
                    weight_old = weights[i]
                    weights[i] = posit.posit_error(weights[i], bit_pos)
                    # print(weights[i], weight_old)
                # else: print('no')
            weights = np.reshape(weights, weight_size)
            # print(np.shape(weights))
            weights = torch.tensor(weights, requires_grad=True)
            d[name] = weights
            param = torch.nn.Parameter(weights)
            # print(param)
            # compare = param == param_old
            # print(compare)

            # print(len(param_old), len(weights), len(param))
            # print(compare)
    # model.state_dict = d
        # break
model.load_state_dict(d)
torch.save({'model_state_dict':model.state_dict()}, '/home/arpita/Approximate-Computing-for-Neural-Networks/models/lenet_mnist_be.pth')

    
    # pd.DataFrame(weights).to_csv(wt)
    # weights = weights.flatten()
    # writer_object = writer(wt)
    # writer_object.writerow(weights)
    # print(name, ' ', np.size(weights))
    # pd.DataFrame(weights.T.reshape(2,-1)).to_csv('renet18_weights.csv')
# print(model.conv1)
# wt.close()