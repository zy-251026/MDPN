import torch
import torchvision
import medmnist
from medmnist import INFO, Evaluator

def loaddata(name, batchsize):
    data_flag = name
    if name == 'mnist':
        train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('dataset', train=True, download=True, 
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batchsize, shuffle=True)
        sets = torchvision.datasets.MNIST('dataset', train=False, download=True, 
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        val_set, test_set = torch.utils.data.random_split(sets, [3000, 7000])
        val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size=batchsize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size=batchsize, shuffle=True)
        n_channels = 1
        n_classes = 10
        img_size = 28
    elif name == 'fashionmnist':
        train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST('dataset', train=True, download=True, 
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor()
                       ])),
        batch_size=batchsize, shuffle=True)
        sets = torchvision.datasets.FashionMNIST('dataset', train=False, download=True, 
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor()
                       ]))
        val_set, test_set = torch.utils.data.random_split(sets, [3000, 7000])
        val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size=batchsize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size=batchsize, shuffle=True)
        n_channels = 1
        n_classes = 10
        img_size = 28
    elif name == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10('dataset', train=True, download=True, 
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                       ])),
        batch_size=batchsize, shuffle=True)
        sets = torchvision.datasets.CIFAR10('dataset', train=False, download=True, 
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                       ]))
        val_set, test_set = torch.utils.data.random_split(sets, [3000, 7000])
        val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size=batchsize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size=batchsize, shuffle=True)
        n_channels = 3
        n_classes = 10
        img_size = 32
    elif name == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100('dataset', train=True, download=True, 
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor()
                       ])),
        batch_size=batchsize, shuffle=True)
        sets = torchvision.datasets.CIFAR100('dataset', train=False, download=True, 
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor()
                       ]))
        val_set, test_set = torch.utils.data.random_split(sets, [3000, 7000])
        val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size=batchsize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size=batchsize, shuffle=True)
        n_channels = 3
        n_classes = 100
        img_size = 32
    elif name == 'imagenet':
        train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageNet('dataset', split='train', download=True, 
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batchsize, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageNet('dataset', split='val', download=True, 
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batchsize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageNet('dataset', split='test', download=True, 
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batchsize, shuffle=True)
    elif name == 'emnist':
        train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.EMNIST('dataset', split = 'mnist', train=True, download=True, 
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor()
                       ])),
        batch_size=batchsize, shuffle=True)
        sets = torchvision.datasets.EMNIST('dataset',split = 'mnist', train=False, download=True, 
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor()
                       ]))
        val_set, test_set = torch.utils.data.random_split(sets, [3000, 7000])
        val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size=batchsize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size=batchsize, shuffle=True)
        n_channels = 1
        n_classes = 10
        img_size = 28
    elif name == 'fer2013':
        sets = torchvision.datasets.FER2013('dataset', split = 'train', 
                       transform=torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        train_set, val_set, test_set = torch.utils.data.random_split(sets, [20000, 2709, 6000])
        train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size=batchsize, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size=batchsize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size=batchsize, shuffle=True)
        n_channels = 1
        n_classes = 7
        img_size = 48
    else:
        info = INFO[data_flag]
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        data_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[.5], std=[.5])
        ])
        train_loader = torch.utils.data.DataLoader(
        DataClass(split='train', transform=data_transform, download=True),
        batch_size=batchsize, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
        DataClass(split='val', transform=data_transform, download=True),
        batch_size=batchsize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
        DataClass(split='test', transform=data_transform, download=True),
        batch_size=batchsize, shuffle=True)
        img_size = 28
    return train_loader, val_loader, test_loader, n_channels, n_classes, img_size
