import torch
from data.data_loader import load_and_split_data, get_data_info
from models.models import get_model
from clients.fedavg_client import FedAvgClient
from server.fedavg_server import FedAvgServer
from evaluation.evaluator import Evaluator

import json
import os
import argparse
import time
import numpy as np

def run_experiment(config):

    SEED = config['seed']
    
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    print("加载并分割数据...")
    client_loaders, test_loader = load_and_split_data(config['dataset'], config['num_clients'], config['samples_per_client'], config['batch_size'])
    
    # 获取数据集信息
    input_dim, num_classes = get_data_info(config['dataset'])
    
    # 初始化全局模型
    global_model = get_model(config['model'], input_dim[0], num_classes).to(device)
    
    # 初始化客户端
    clients = [FedAvgClient(i, loader, get_model(config['model'], input_dim[0], num_classes).to(device), 
                            device, config['learning_rate'], config['local_epochs']) 
               for i, loader in enumerate(client_loaders)]
    
    # 初始化服务器
    server = FedAvgServer(global_model, clients)
    
    # 初始化评估器
    evaluator = Evaluator(global_model, test_loader, device)
    
    # 记录实验结果
    results = []
    
    # 联邦学习过程
    print("开始联邦学习过程...")
    
    # 打印训练参数--按照可视化格式，包括各客户端数据
    print("*"*40)
    print(f"数据集: {config['dataset']}")
    print(f"模型: {config['model']}")
    print(f"客户端数量: {config['num_clients']}")
    print(f"每个客户端的样本数量: {len(client_loaders[0].dataset)}")
    print(f"学习率: {config['learning_rate']}")
    print(f"本地训练轮数: {config['local_epochs']}")
    print(f"联邦学习轮数: {config['num_rounds']}")
    print(f"随机种子: {config['seed']}")
    print("*"*40)

    for round in range(config['num_rounds']):
        round_start_time = time.time()
        

        # 分发
        server.distribute()

        # 统计客户端训练时间
        client_train_time = time.time()
        # 客户端训练
        client_models = server.client_train()
        client_train_time = time.time() - client_train_time

        # 统计聚合时间
        aggregate_time = time.time()
        # 聚合
        server.aggregate(client_models)
        aggregate_time = time.time() - aggregate_time
        
        # 评估
        metrics = evaluator.evaluate()
        latest_metrics = evaluator.get_latest_metrics()
        
        # 计算通信成本 (这里使用模型参数大小作为估计)
        communication_cost = sum(p.numel() for p in global_model.parameters()) * 4  # 假设每个参数是4字节


        # 记录本轮结果
        round_time = time.time() - round_start_time

        costs = {
            'communication_cost': communication_cost,
            'client_train_time': client_train_time/len(clients), # 每个客户端的平均训练时间
            'aggregate_time': aggregate_time,
            'round_time': round_time
        }
        round_result = {
            'round': round,
            'metrics': latest_metrics,
            'costs': costs
        }
        results.append(round_result)
        
        print(f"轮次 {round}: 准确率 = {latest_metrics['accuracy']:.4f}, 损失 = {latest_metrics['loss']:.4f}, 用时 = {round_time:.2f}秒")
    
    # 保存实验结果
    save_results(config, results)

def save_results(config, results):
    # 生成目录名
    dataset_dir = f"results/{config['dataset']}/clients{config['num_clients']}"

    # 创建结果目录
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    # 生成文件名
    filename = f"{dataset_dir}/{time_str}_{config['seed']}.json"
    
    # 保存结果
    with open(filename, 'w') as f:
        json.dump({'config': config, 'results': results}, f, indent=2)
    
    print(f"实验结果已保存到 {filename}")

def main():
    parser = argparse.ArgumentParser(description='联邦学习实验')
    parser.add_argument('-d','--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'cifar100'], help='数据集名称')
    parser.add_argument('--model', type=str, default='simple_cnn', choices=['simple_cnn', 'resnet18', 'resnet20', 'resnet32', 'resnet44', 'resnet56'], help='模型名称')
    parser.add_argument('--num_clients', type=int, default=5, help='客户端数量')
    parser.add_argument('--samples_per_client', type=int, default=10000, help='每个客户端的样本数量')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='学习率')
    parser.add_argument('--local_epochs', type=int, default=1, help='本地训练轮数')
    parser.add_argument('--num_rounds', type=int, default=200, help='联邦学习轮数')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    args = parser.parse_args()
    if args.seed == 0:
        args.seed = int(time.time()%1000000)
    config = vars(args)

    run_experiment(config)

if __name__ == "__main__":
    main()

