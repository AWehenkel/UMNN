import torch
from torchvision import datasets, transforms
from lib.transform import AddUniformNoise, ToTensor, HorizontalFlip, Transpose, Resize

def dataloader(dataset, batch_size, cuda, conditionnal=False):

    if dataset == 'CIFAR10':
        data = datasets.CIFAR10('./CIFAR10', train=True, download=True,
                       transform=transforms.Compose([
                           AddUniformNoise(0.05),
                           Transpose(),
                           ToTensor()
                       ]))

        data_hflip = datasets.CIFAR10('./CIFAR10', train=True, download=True,
                           transform=transforms.Compose([
                           HorizontalFlip(), 
                           AddUniformNoise(0.05),
                           Transpose(),
                           ToTensor()
                       ]))
        data = torch.utils.data.ConcatDataset([data, data_hflip])

        train_data, valid_data = torch.utils.data.random_split(data, [90000, 10000])

        test_data = datasets.CIFAR10('./CIFAR10', train=False, download=True,
                        transform=transforms.Compose([
                            AddUniformNoise(0.05),
                            Transpose(),
                            ToTensor()
                       ]))

    elif dataset == 'MNIST':
        data = datasets.MNIST('./MNIST', train=True, download=True,
                   transform=transforms.Compose([
                       AddUniformNoise(),
                       ToTensor()
                   ]))


        train_data, valid_data = torch.utils.data.random_split(data, [50000, 10000])
 
        test_data = datasets.MNIST('./MNIST', train=False, download=True,
                    transform=transforms.Compose([
                        AddUniformNoise(),
                        ToTensor()
                    ]))



    elif len(dataset) == 6 and dataset[:5] == 'MNIST':
        data = datasets.MNIST('./MNIST', train=True, download=True,
                              transform=transforms.Compose([
                                  AddUniformNoise(),
                                  ToTensor()
                              ]))
        label = int(dataset[5])
        idx = data.train_labels == label
        data.train_labels = data.train_labels[idx]
        data.train_data = data.train_data[idx]

        train_data, valid_data = torch.utils.data.random_split(data, [5000, idx.sum() - 5000])

        test_data = datasets.MNIST('./MNIST', train=False, download=True,
                                   transform=transforms.Compose([
                                       AddUniformNoise(),
                                       ToTensor()
                                   ]))
        idx = test_data.test_labels == label
        test_data.test_labels = test_data.test_labels[idx]
        test_data.test_data = test_data.test_data[idx]
    elif dataset == 'MNIST32':
        data = datasets.MNIST('./MNIST', train=True, download=True,
                              transform=transforms.Compose([
                                  Resize(),
                                  AddUniformNoise(),
                                  ToTensor()
                              ]))

        train_data, valid_data = torch.utils.data.random_split(data, [50000, 10000])


        test_data = datasets.MNIST('./MNIST', train=False, download=True,
                                   transform=transforms.Compose([
                                       Resize(),
                                       AddUniformNoise(),
                                       ToTensor()
                                   ]))
    elif len(dataset) == 8 and dataset[:7] == 'MNIST32':
        data = datasets.MNIST('./MNIST', train=True, download=True,
                              transform=transforms.Compose([
                                  Resize(),
                                  AddUniformNoise(),
                                  ToTensor()
                              ]))

        label = int(dataset[7])
        idx = data.train_labels == label
        data.train_labels = data.train_labels[idx]
        data.train_data = data.train_data[idx]

        train_data, valid_data = torch.utils.data.random_split(data, [5000, idx.sum() - 5000])

        test_data = datasets.MNIST('./MNIST', train=False, download=True,
                                   transform=transforms.Compose([
                                       Resize(),
                                       AddUniformNoise(),
                                       ToTensor()
                                   ]))
        idx = test_data.test_labels == label
        test_data.test_labels = test_data.test_labels[idx]
        test_data.test_data = test_data.test_data[idx]
    else:  
        print ('what network ?', args.net)
        sys.exit(1)

    #load data 
    kwargs = {'num_workers': 0, 'pin_memory': True} if cuda > -1 else {}

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size, shuffle=True, **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size, shuffle=True, **kwargs)
 
    test_loader = torch.utils.data.DataLoader(test_data,
        batch_size=batch_size, shuffle=True, **kwargs)
    
    return train_loader, valid_loader, test_loader
