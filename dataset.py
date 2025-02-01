from torchvision.datasets import MNIST
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
import torch
import numpy as np

TRAIN_DATASET = MNIST(root='dataset', train=True, download=True)
TEST_DATASET = MNIST(root='dataset', train=False, download=True)

def get_train_split(partition_id, num_partitions) :
    dataset_size = len(TRAIN_DATASET)
    split_size = dataset_size // num_partitions
    indices = list(range(dataset_size))
    start_idx = partition_id * split_size
    end_idx = start_idx + split_size

    return Subset(TRAIN_DATASET, indices[start_idx:end_idx])

def get_test():
    return TEST_DATASET

def load_data(partition_id, num_partitions):
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5), (0.5))]
    )
    train_split = get_train_split(partition_id, num_partitions)

    train_split.dataset.transform = pytorch_transforms
    TEST_DATASET.transform = pytorch_transforms

    # Split trainloader into train and validation subsets
    trainloader = DataLoader(train_split, batch_size=64, shuffle=True)
    
    # train_size = int(0.8 * len(trainloader.dataset))
    # val_size = len(trainloader.dataset) - train_size
    
    # train_subset, val_subset = torch.utils.data.random_split(trainloader.dataset, [train_size, val_size])

    # trainloader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    
    # valloader = torch.utils.data.DataLoader(val_subset, batch_size=64, shuffle=False)

    testloader = DataLoader(TEST_DATASET, batch_size=64, shuffle=False)

    return trainloader, testloader

def get_train_split_non_iid(partition_id, num_partitions) :
    
    digit_partitions = np.array_split(np.arange(10), num_partitions)
    partition = digit_partitions[partition_id]
    
    indices = [idx for idx, target in enumerate(TRAIN_DATASET.targets) if target.item() in partition]
    return Subset(TRAIN_DATASET, indices)

def load_data_non_iid(partition_id, num_partitions):
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5), (0.5))]
    )
    train_split = get_train_split_non_iid(partition_id, num_partitions)

    train_split.dataset.transform = pytorch_transforms
    TEST_DATASET.transform = pytorch_transforms

     # Split trainloader into train and validation subsets
    trainloader = DataLoader(train_split, batch_size=64, shuffle=True)
    
    train_size = int(0.8 * len(trainloader.dataset))
    val_size = len(trainloader.dataset) - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(trainloader.dataset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=64, shuffle=False)

    testloader = DataLoader(TEST_DATASET, batch_size=64, shuffle=False)

    return trainloader, valloader, testloader

if __name__ == "__main__" :
    trainloader, valloader, testloader = load_data_non_iid(0, 2)
    print(len(trainloader.dataset)) 
    trainloader, valloader, testloader = load_data_non_iid(1, 2)
    print(len(trainloader.dataset)) 