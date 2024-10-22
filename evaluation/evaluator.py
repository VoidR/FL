import torch
import time
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Evaluator:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.metrics = defaultdict(list)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        criterion = torch.nn.CrossEntropyLoss()

        start_time = time.time()
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        evaluation_time = time.time() - start_time

        # 计算性能指标
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)

        # 记录指标
        self.metrics['loss'].append(total_loss / len(self.test_loader))
        self.metrics['accuracy'].append(accuracy)
        self.metrics['precision'].append(precision)
        self.metrics['recall'].append(recall)
        self.metrics['f1'].append(f1)
        self.metrics['evaluation_time'].append(evaluation_time)

        return self.metrics

    def get_latest_metrics(self):
        return {k: v[-1] for k, v in self.metrics.items()}

    def get_all_metrics(self):
        return self.metrics
