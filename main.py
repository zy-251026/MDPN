from numpy import random
import numpy
import torch
import torch.nn as nn
import time
import torchvision
import resnext
from fit import fit_model
from fit2 import fit2_model
from model.model_tnns import tnn_Model
from model.model_tnns3 import tnn3_Model
from loaddata import loaddata
from model.AlexNet import AlexNet
from model.ResNet152 import resnet152
from model.dd import ddNet
from model.bls import blsNet
from model.DSAN import DSAN
from model.model_mdnm import MDNM_Model
from model.bilstm import BiLSTM, SpatialAttention, SpectralAttention
from model.discriminator import Discriminator
import medmnist
import argparse
from model.tlr import CifarNet
parser = argparse.ArgumentParser()
parser.add_argument('--device',default='cuda:0',type=str)
# parser.add_argument('--M',default=11,type=int)
parser.add_argument('--epochs',default=50,type=int)
parser.add_argument('--batch_size',default=64,type=int)
parser.add_argument('--data_num',default=0,type=int)
parser.add_argument('--model_num',default=0, type=int)
parser.add_argument('--dataset', type=str, default='indian',
                        choices=['indian','pavia','houston','salina','ksc'],
                        help='dataset name')
                        
parser.add_argument('--network', type=str, default='ssgrn',
                        choices=['segrn','sagrn','ssgrn','fcn'],
                        help='network name')
parser.add_argument('--norm', type=str, default='std',
                        choices=['std','norm'],
                        help='nomalization mode')
parser.add_argument('--mi', type=int, default=-1,
                        help='min normalization range')
parser.add_argument('--ma', type=int, default=1,
                        help='max normalization range')
parser.add_argument('--sync_bn', type=str, default='True',
                        choices=['True', 'False'],help='synchronized batchNorm')
parser.add_argument('--use_apex', type=str, default='False',
                        choices=['True', 'False'],help='mixed-precision training')
parser.add_argument('--opt_level', type=str, default='O1',
                        choices=['O0', 'O1','O2'], help='mixed-precision')
parser.add_argument('--input_mode', type=str, default='part',
                        choices=['whole', 'part'],help='input setting')
parser.add_argument('--input_size', type=int)
parser.add_argument('--overlap_size', type=int, default=16,
                        help='size of overlap')
parser.add_argument('--experiment-num', type=int, default=1,
                        help='experiment trials number')
parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
parser.add_argument('--batch-size', type=int, default=61,
                        help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=4,
                        help='input batch size for validation')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
parser.add_argument('--workers', type=int, default=2,
                        help='workers num')
parser.add_argument('--ignore_label', type=int, default=255,
                        help='ignore label')
parser.add_argument('--print_freq', type=int, default=3,
                        help='print frequency')
parser.add_argument("--resume", type=str, help="model path.")
    # model setting
parser.add_argument('--sa_groups', type=int, default=256, help='spatial group number')
parser.add_argument('--se_groups', type=int, default=256, help='spectral group number')

args = parser.parse_args()

device = torch.device(args.device)
num_epochs = args.epochs
batch_size = args.batch_size
model_num = args.model_num

alldatalist = ['mnist','fashionmnist','cifar10','cifar100','pathmnist','chestmnist','dermamnist','octmnist','pneumoniamnist','retinamnist','breastmnist','bloodmnist','tissuemnist','organamnist','organcmnist','organsmnist', 'emnist','fer2013','emnistd', 'emnistl']
ddatalist = ['organmnist3d','nodulemnist3d','adrenalmnist3d','fracturemnist3d','vesselmnist3d','synapsemnist3d']
model_name = ['MDPN','Alexnet','resnet','resnext','dd','bls','dsan','bi-lstm','tlr','dnm','sdenet']
dataname = alldatalist[args.data_num]
train_loader, val_loader, test_loader, channel, classes, img_size = loaddata(dataname, batch_size)
args.input_size = [img_size, img_size]
print(train_loader)
for runtime in range(5):
    print('Model:'+model_name[model_num]+'  runtime:'+str(runtime))
    if model_num == 0:
        if channel==1:
            model = tnn_Model(50, classes, img_size).to(device)
        else:
            model = tnn3_Model(50, classes, img_size).to(device)
    if model_num == 1:
        model = AlexNet(classes, channel, img_size)
    elif model_num == 2:
        model = resnet152(classes, channel, img_size)
    elif model_num == 3:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d')
        model.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif model_num == 4:
        model = ddNet(classes, channel, img_size)
    elif model_num == 5:
        model = blsNet(10, 8000, classes, channel, img_size)
    elif model_num == 6:
        model = DSAN(classes, channel, img_size)
    elif model_num == 7:
        model = nn.Sequential(SpatialAttention(channel, 9),SpectralAttention(),BiLSTM(128, 2, 0.5, classes))
    elif model_num == 8:
        model = CifarNet(classes, channel, img_size)
    elif model_num == 9:
        model = MDNM_Model(classes, channel, img_size)
    elif model_num == 10:
        model = Discriminator(channel, channel, classes, img_size)
    model.to(device)
    LR = 0.001
    # for k in range(10):
        # model[str(k)] = MDNM_Model(28, 10, sel2, stride)
        # weight = torch.randn(30, 28 * 28)
        # theta = torch.randn(30, 28 * 28)
        # model[str(k)] = IDNMW_Model(weight, theta, 10, 0.5)
        # model[str(k)] = tnn_Model(weight, theta)
        # model[str(k)] = IDNM2_Model(weight, theta, 10, 0.5)
    # optimizer = {}
    # for k in range(10):
    #     optimizer[str(k)] = torch.optim.Adam(model[str(k)].parameters(), lr=LR)
        # torch.optim.lr_scheduler.ExponentialLR(optimizer[str(k)], gamma=0.93)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    # input_shape = (-1, 1, 28, 28)
    # input_shape = (-1, 3, 32, 32)
    # input_shape = (-1, 28*28)
    # input_shape = (-1, 3*28*28)
    #input_shape = (-1, 28, 28)
    if channel==1:
        if model_num == 0:
            input_shape = (-1, img_size, img_size)
        else:    
            input_shape = (-1, 1, img_size, img_size)
    else:
        input_shape = (-1, 3, img_size, img_size)
    # input_shape = (-1, 3 * 32 * 32)
    # input_shape = (-1,28)
    training_loss, training_accuracy, validation_loss, validation_accuracy = fit2_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, val_loader, test_loader, args.model_num, args.data_num, runtime, device)
    # training_loss, training_accuracy, validation_loss, validation_accuracy = fit_singlemodel(model, loss_func, optimizer, input_shape, num_epochs, train_loader, test_loader, ii)

    # plt.plot(range(num_epochs), training_loss, 'b-', label='Training Loss')
# plt.plot(range(num_epochs), validation_loss, 'g-', label='Validation Loss')
# plt.title('Training and Validation loss')
# plt.xlabel('Number of epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
#
# plt.plot(range(num_epochs), training_accuracy, 'b-', label='Training Accuracy')
# plt.plot(range(num_epochs), validation_accuracy, 'g-', label='Validation Accuracy')
# plt.title('Training and Validation accuracy')
# plt.xlabel('Number of epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()2
