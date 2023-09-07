import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch
import medmnist
from medmnist import INFO, Evaluator
data_flag = 'bloodmnist'
info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
# train = torchvision.datasets.FER2013('./',split = 'test', transform = data_transform)

#train = torchvision.datasets.MNIST('./dataset/', train=True, download=True)
train = DataClass(split='train', transform=data_transform, download=True)
# image, target = train._load_data()
fig = plt.figure()
ind = [0,14,9,6,1,3,44,2,5]
image = train.imgs[14]
# image = image[3:9,4:10,:]
# print(image)
plt.imshow(image)
# plt.xticks([])
# plt.yticks([])
# plt.show()
# j = 0
for i,c in enumerate(range(100)):
    image, target = train.imgs[i], train.labels[c]
# #     j = j+1
    plt.subplot(10,10,i+1)
#     # plt.tight_layout()
#     # images = torch.xstack((image, image, image),2)
    plt.imshow(image)
#     plt.xticks([])
#     plt.yticks([])
    plt.title("title:{}".format(target))
    # if target == 4:
    #     plt.imshow(image)
    #     print(i)
    #     break
plt.show()