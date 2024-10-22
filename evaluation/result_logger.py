import json
import os
import matplotlib.pyplot as plt

class ResultLogger:
    def __init__(self, experiment_name, output_dir='results'):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.results = []
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def log_round(self, round_num, metrics, communication_cost):
        result = {
            'round': round_num,
            'metrics': metrics,
            'communication_cost': communication_cost
        }
        self.results.append(result)

    def save_results(self):
        file_path = os.path.join(self.output_dir, f'{self.experiment_name}results.json')
        with open(file_path, 'w') as f:
            json.dump(self.results, f, indent=2)

    def plot_metrics(self):
        rounds = [r['round'] for r in self.results]
        metrics = self.results[0]['metrics'].keys()
        for metric in metrics:
            values = [r['metrics'][metric] for r in self.results]
            plt.figure(figsize=(10, 5))
            plt.plot(rounds, values)
            plt.title(f'{metric.capitalize()} over rounds')
            plt.xlabel('Round')
            plt.ylabel(metric.capitalize())
            plt.savefig(os.path.join(self.output_dir, f'{self.experiment_name}{metric}.png'))
            plt.close()
        # 绘制通信成本
        comm_costs = [r['communication_cost'] for r in self.results]
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, comm_costs)
        plt.title('Communication cost over rounds')
        plt.xlabel('Round')
        plt.ylabel('Communication cost')
        plt.savefig(os.path.join(self.output_dir, f'{self.experiment_name}communication_cost.png'))
        plt.close()
