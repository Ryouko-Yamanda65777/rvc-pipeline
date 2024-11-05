
import torch
from multiprocessing import cpu_count

class Configs:
    def __init__(self, device="cuda:0", is_half=True):
        self.device = device
        self.is_half = is_half
        self.n_cpu = cpu_count()
        self.gpu_name, self.gpu_mem = None, None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        # Implement GPU or CPU detection logic here
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory / 1024 / 1024 / 1024
            )
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        if self.is_half:
            x_pad, x_query, x_center, x_max = 3, 10, 60, 65
        else:
            x_pad, x_query, x_center, x_max = 1, 6, 38, 41

        if self.gpu_mem and self.gpu_mem <= 4:
            x_pad, x_query, x_center, x_max = 1, 5, 30, 32

        return x_pad, x_query, x_center, x_max
