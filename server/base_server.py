import torch

class BaseServer:
    def __init__(self, global_model, clients):
        self.global_model = global_model
        self.clients = clients

    def aggregate(self):
        raise NotImplementedError("子类必须实现aggregate方法")

    def distribute(self):
        raise NotImplementedError("子类必须实现distribute方法")
