
from tensorboardX import SummaryWriter
import torch

class TensorboardLogger(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_counts = {}

    def add_scalar(self, path, value, count = None, period=10):
        if count is None:
            count = self.log_counts.get(path, 0)
        if count % period == 0:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            super().add_scalar(path, value, count)
        self.log_counts[path] = count + 1
    
    def add_histogram(self, path, value, count = None, period=10):
        if count is None:
            count = self.log_counts.get(path, 0)
        if count % period == 0:
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            super().add_histogram(path, value, count)
        self.log_counts[path] = count + 1


        