import random
import numpy as np 
import csv
import pandas as pd 
from csv import writer
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
import random
import torchvision
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import Module
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import torchvision.models as models
import numpy as np
from float_to_positf import fposit



# def dec_to_ieee754(x):
    
#     sign_bit = '0'
#     if(x<0):
#         sign_bit = '1'
    
#     abs_value = abs(x)

#     start_power = 0
#     two_base = abs_value

#     while(int(two_base)!=1):
#         two_base = abs_value/pow(2,start_power)
#         if(abs(x)<1):
#             start_power -= 1
#         else:
#             start_power += 1
#         # print(two_base)
       
#     # print(two_base)
#     if(start_power>0):
#         exp = 127 + start_power-1
#     elif(start_power<0):
#         exp = 127 + start_power +1
#     else:
#         exp = 127    
#     # return(exp)
    
#     fraction = two_base - 1
#     # print(fraction)
#     frac_str = str()
#     mant_str = str()

#     while (fraction and len(frac_str)<23):

#         fraction *=2

#         if(fraction >=1):
#             frac_update_int = 1
#             fraction-=1
#         else:
#             frac_update_int = 0
        
#         frac_str += str(frac_update_int)
#         # print(fraction)

#     exp_str = bin(exp)[2 :]
#     if(len(exp_str)<8):
#         exp_str = '0'+exp_str

#     if(len(frac_str)<23):
#         frac_str = frac_str + ('0'*(23-len(frac_str)))

#     return sign_bit, exp_str, frac_str




# def ieee754_to_dec(sign_bit_str, exp_str, mant_str):

#     sign = (-1)**int(sign_bit_str)

#     biased_exp = int(exp_str, 2)
#     exp_actual = biased_exp - 127

#     power = -1
#     mant_int = 0
#     for i in mant_str:

#         mant_int += (int(i)*pow(2,power))
#         power -=1
    
#     mant_int +=1

#     x = sign*mant_int*pow(2,exp_actual)

#     return x



model = resnet18()
model.to('cpu')
checkpoint = torch.load(f'/home/arpita/Approximate-Computing-for-Neural-Networks/models/resnet18-5c106cde.pth', map_location = torch.device('cpu'))
model.load_state_dict(checkpoint)

original = r'/home/arpita/Approximate-Computing-for-Neural-Networks/bit_error/resnet18_weights.csv'
copied = r'/home/arpita/Approximate-Computing-for-Neural-Networks/bit_error/resnet18_weights_error.csv'
shutil.copyfile(original, copied) 

skip_rows = [0]
param_names = []
param_size = []
for name, param in model.named_parameters():
    weights = param.detach().numpy()
    weights = weights.flatten()
    row_no = np.size(weights) +1
    skip_rows.append(skip_rows[len(skip_rows)-1]+row_no)
    param_names.append(name)
    param_size.append(np.size(weights))

posit_length = 12
posit_regime = 2
posit_es = 4
posit = fposit(posit_length,  posit_regime, posit_es)

count = -1

df = pd.read_csv('resnet18_weights.csv', skiprows = skip_rows, header = None)


random.seed()
bit_error_prob = 0.01

with open('resnet18_weights_error.csv', 'a') as f_object:
    writer_object = writer(f_object)

    for index, row in df.iterrows():

        if row[0] == 0:
            count = count +1
            writer_object.writerow([0])

        if param_names[count].find('conv') != -1:

            bit_flip_true = random.choices([0,1],weights=[1-bit_error_prob, bit_error_prob])
            if(bit_flip_true == 1):
                bit_pos = random.randint(0,12)
                weight_error = posit.posit_error(row[1], bit_pos)
                add_row = [row[0], weight_error]
                writer_object.writerow(add_row)

            else:
                add_row = [row[0], row[1]]
                writer_object.writerow(add_row)
        
        else:

            add_row = [row[0], row[1]]
            writer_object.writerow(add_row)
f_object.close()


            

            


# posit = fposit(6,2,2)
# a=-1.48
# print(posit.extract(a))
# print(posit.float2posit(a))
# print(posit.posit2float(posit.float2posit(a)))
# print(posit.posit_error(a,2))

        
         
    
# m = max(num)
# print(num)
# print(m)

# for name, param in model.named_parameters():
#     name_list = name.split(".")
#     random.seed()
#     if 'weight' in name_list:
#         for i, weights in enumerate(wt_original.iterrows()):
#             if i == layer_count:
#                 # weights_list = weights.split(",")
#                 print(weights)
#                 # weights_list = []
#                 # for j in weights:
                #     # weights_list.append(float(j))
                #     print(j)

                # print(weights_list)

                
                # break
            # bit_flip_true = random.choices([0,1],weights=[1-bit_error_prob, bit_error_prob])
            # if(bit_flip_true):
    # layer_count = layer_count + 1
    # break
            




        


# sign_bit, exp_str, frac_str = dec_to_ieee754(weight)
# float_point = sign_bit+exp_str+frac_str

# 
# if(bit_flip_true==1):
#     bit_index = random.randint(0,31)
#     if(float_point[bit_index] == '0'):
#         float_point[bit_index] = '1'
#     else:
#         float_point[bit_index] = '0'

# sign_bit = float_point[0]
# exp_str = float_point[1:8]
# frac_str = float_point[9:]

# weight_












# x = 1.5
# s,e,m = dec_to_ieee754(x)
# print(s,e,m)
# ag = ieee754_to_dec(s,e,m)
# print(ag)
