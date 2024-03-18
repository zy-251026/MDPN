import torch
from torch.autograd import Variable
import time
import pandas
import os

def fit2_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, val_loader, test_loader, name_num, dataset, runtime,device):
    learnrate = [0.1,0.05,0.01,0.005,0.0025,0.001,0.0008,0.0006,0.0004,0.0001]
    model_name = ['MDPN','Alexnet','resnet','resnext','dd','bls','dsan','bi-lstm','tlr','dnm','ssgrn']
    namelist = ['mnist','fashionmnist','cifar10','cifar100','pathmnist','chestmnist','dermamnist','octmnist','pneumoniamnist','retinamnist','breastmnist','bloodmnist','tissuemnist','organamnist','organcmnist','organsmnist','emnist',  'fer2013', 'emnistd', 'emnistl']
    alldata = []
    testdata = []
    name = namelist[dataset]
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    all_pred = []
    all_label = []
    totalacc = -99999
    total_loss = 0
    path = 'data/' + model_name[name_num] + '/' + name
    os.makedirs(path, exist_ok=True) 
    path1 = path + '/proc' + str(runtime+1) + '.csv'
    path2 = path + '/model' + str(runtime+1) + '.pth'
    path3 = path + '/pred' + str(runtime+1) + '.csv'
    path4 = path + '/label' + str(runtime+1) + '.csv'
    path5 = path + '/test' + str(runtime+1) + '.csv'
    start = time.time()
    for epoch in range(num_epochs):
        correct_train = 0
        total_train = 0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            train = Variable(images.view(input_shape)).to(device)
            labels = Variable(labels).to(device)
            optimizer.zero_grad()
            outputs = model(train)
            if len(labels.size()) != 1:
                if labels.size(1) == 1:
                    labels = labels.view(labels.size(0))
                else:
                    labels = torch.argmax(labels, dim=1)
            train_loss = loss_func(outputs, labels)
            train_loss.backward()
            optimizer.step()
            predicted = torch.max(outputs.data, 1)[1]
            total_train += len(labels)
            correct_train += (predicted == labels).float().sum()
        train_accuracy = 100 * correct_train/ float(total_train)
        training_accuracy.append(train_accuracy.tolist())
        training_loss.append(train_loss.data.tolist())

        correct_val = 0
        total_val = 0
        model.eval()
        for images, labels in val_loader:
            val = Variable(images.view(input_shape)).to(device)
            labels = labels.to(device)
            outputs = model(val)
            if len(labels.size()) != 1:
                if labels.size(1) == 1:
                    labels = labels.view(labels.size(0))
                else:
                    labels = torch.argmax(labels, dim=1)
            val_loss = loss_func(outputs, labels)
            predicted = torch.max(outputs.data, 1)[1]
            total_val += len(labels)
            correct_val += (predicted == labels).float().sum()
        val_accuracy = 100 * correct_val / float(total_val)
        validation_accuracy.append(val_accuracy.tolist())
        validation_loss.append(val_loss.data.tolist())
    
        correct_test = 0
        total_test = 0
        # model.eval()
        for images, labels in test_loader:
            test = Variable(images.view(input_shape)).to(device)
            labels = labels.to(device)
            outputs = model(test)
            if len(labels.size()) != 1:
                if labels.size(1) == 1:
                    labels = labels.view(labels.size(0))
                else:
                    labels = torch.argmax(labels, dim=1)
            test_loss = loss_func(outputs, labels)
            predicted = torch.max(outputs.data, 1)[1]
            total_test += len(labels)
            correct_test += (predicted == labels).float().sum()
            all_pred.append(predicted.tolist())
            all_label.append(labels.tolist())
        
        test_accuracy = 100 * correct_test / float(total_test)
        test_accuracy = test_accuracy.tolist()
        test_loss = test_loss.data.tolist()
        print(
            'Epoch: {}/{} Training_loss: {:.6f} Training_acc: {:.6f}% Validation_loss: {:.6f} Validation_acc: {:.6f}% Test_loss: {:.6f} Test_acc: {:.6f}%'.format(
                epoch + 1, num_epochs, train_loss.data, train_accuracy, val_loss.data, val_accuracy, test_loss, test_accuracy))
    end = time.time()
    print('Time_consumption: {:.6f}'.format(end-start))
    torch.save(model, path2)
    testdata.append(test_accuracy)
    testdata.append(test_loss)
    alldata.append(training_loss)
    alldata.append(validation_loss)
    alldata.append(training_accuracy)
    alldata.append(validation_accuracy)
    d1 = pandas.DataFrame(alldata)
    d2 = pandas.DataFrame(all_pred)
    d3 = pandas.DataFrame(all_label)
    d4 = pandas.DataFrame(testdata)
    d1.to_csv(path1, header=False, index=False)
    d2.to_csv(path3, header=False, index=False)
    d3.to_csv(path4, header=False, index=False)
    d4.to_csv(path5, header=False, index=False)
    return training_loss, training_accuracy, validation_loss, validation_accuracy
