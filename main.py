from numpy import random
import numpy
import torch
import torch.nn as nn
import time
import torchvision
from fit import fit_model
# from fitsingle import fit_singlemodel
# from model_lenet5 import CNN_Model
# from model_ann3 import ANN_Model
# from model_dnm import DNM_Model
from model_mdpn import mdpn_Model
from model_mdpn3 import mdpn3_Model
from model_mdpn_modify import mdpn_Model_modify
from model_mdpn3_modify import mdpn3_Model_modify
from loaddata import loaddata
import medmnist
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device',default='cuda:0',type=str)
arser.add_argument('--M',default=10,type=int)
parser.add_argument('--epochs',default=1,type=int)
parser.add_argument('--batch_size',default=64,type=int)
parser.add_argument('--data_num',default=0,type=int)

args = parser.parse_args()
device = torch.device(args.device)
num_epochs = args.epochs
batch_size = args.batch_size

# torch.set_num_threads(8)
# (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

alldatalist = ['mnist','fashionmnist','cifar10','cifar100','pathmnist','chestmnist','dermamnist','octmnist','pneumoniamnist','retinamnist','breastmnist','bloodmnist','tissuemnist','organamnist','organcmnist','organsmnist','emnist', 'fer2013']
ddatalist = ['organmnist3d','nodulemnist3d','adrenalmnist3d','fracturemnist3d','vesselmnist3d','synapsemnist3d']
# learnrate = [0.1,0.05,0.01,0.005,0.0025,0.001,0.0008,0.0006,0.0004,0.0001]
dataname = alldatalist[args.data_num]
train_loader, val_loader, test_loader, channel, classes, imgsize = loaddata(dataname, batch_size)
for runtime in range(1):
    if channel==1:
        model = tnn_Model(10, classes, imgsize)
    else:
        model = tnn3_Model(10, classes, imgsize)
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
        input_shape = (-1, imgsize, imgsize)
    else:
        input_shape = (-1, 3, imgsize, imgsize)
    # input_shape = (-1, 3 * 32 * 32)
    # input_shape = (-1,28)
    training_loss, training_accuracy, validation_loss, validation_accuracy = fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, val_loader, test_loader, M, args.data_num, runtime, device)
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
# plt.show()
