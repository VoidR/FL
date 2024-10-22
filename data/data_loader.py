import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def load_and_split_data(dataset_name, num_clients, samples_per_client, batch_size=32):
    """
    加载数据集并将其分割给多个客户端,允许总样本数超过数据集大小
    
    参数:
    - dataset_name: 数据集名称 ('mnist', 'cifar10', 或 'cifar100')
    - num_clients: 客户端数量
    - samples_per_client: 每个客户端的样本数量(可以是整数或列表)
    - batch_size: 批次大小
    
    返回:
    - 客户端训练数据加载器列表
    - 测试数据加载器
    """
    
    # 定义数据转换
    if dataset_name == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name == 'cifar10':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset_name == 'cifar100':
        train_transform = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([125.3, 123.0, 113.9]) / 255.0,
                np.array([63.0, 62.1, 66.7]) / 255.0),
        ])
    else:
        raise ValueError("不支持的数据集名称")
    
    # 加载数据集
    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST('~/.torch', train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST('~/.torch', train=False, transform=val_transform)
    elif dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10('~/.torch', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10('~/.torch', train=False, transform=val_transform)
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100('~/.torch', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100('~/.torch', train=False, transform=val_transform)
    else:
        raise ValueError("不支持的数据集名称")

    
    if isinstance(samples_per_client, int):
        if samples_per_client == 0:
            samples_per_client = [len(train_dataset) // num_clients] * num_clients
        else:
            samples_per_client = [samples_per_client] * num_clients
    assert len(samples_per_client) == num_clients, "客户端数量与样本分配不匹配"
    
    dataset_size = len(train_dataset)

    # 创建客户端数据加载器
    client_loaders = []
    for samples in samples_per_client:
        # 对于每个客户端,随机选择指定数量的样本(允许重复)
        indices = torch.randint(dataset_size, (samples,))
        client_dataset = Subset(train_dataset, indices)
        client_loaders.append(DataLoader(client_dataset, batch_size=batch_size, shuffle=True))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return client_loaders, test_loader

def get_data_info(dataset_name):
    """
    获取数据集的基本信息
    
    参数:
    - dataset_name: 数据集名称
    
    返回:
    - 输入维度
    - 类别数量
    """
    if dataset_name == 'mnist':
        input_dim = (1, 28, 28)
        num_classes = 10
    elif dataset_name == 'cifar10':
        input_dim = (3, 32, 32)
        num_classes = 10
    elif dataset_name == 'cifar100':
        input_dim = (3, 32, 32)
        num_classes = 100
    else:
        raise ValueError("不支持的数据集名称")
    
    return input_dim, num_classes
