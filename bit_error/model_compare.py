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
#from resnet_approx1 import resnet18
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

model1 = LeNet5()
model1.to('cpu')
checkpoint1 = torch.load(f'/home/arpita/Approximate-Computing-for-Neural-Networks/models/lenet_mnist', map_location = torch.device('cpu'))
model1.load_state_dict(checkpoint1['model_state_dict'])

model2 = LeNet5()
model2.to('cpu')
checkpoint2 = torch.load(f'/home/arpita/Approximate-Computing-for-Neural-Networks/models/lenet_mnist_be09.pth', map_location = torch.device('cpu'))
model2.load_state_dict(checkpoint2['model_state_dict'])

sum1 = 0
sum2 = 0

for name, param in model1.named_parameters():
    weights1 = param.detach().numpy()
    weights1 = weights1.flatten()
    sum1 = sum1 + np.sum(weights1)
    # break

for name, param in model2.named_parameters():
    weights2 = param.detach().numpy()
    weights2 = weights2.flatten()
    sum2 = sum2 + np.sum(weights2)
    # break
# a = np.array(a, dtype=object)
# b = np.array(b, dtype=object)
# print(a,b)
# compare = a == b
print(sum1, sum2)
# if weights1.all() == weights2.all():
#     print('equal')
# else:
#     print('unequal')

