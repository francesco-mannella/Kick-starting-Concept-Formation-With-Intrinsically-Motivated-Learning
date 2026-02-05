import os

class StorageManager:

    def __init__(self):
        self.simulations_dir = "simulations"
        self.storage_dir = "storage"
        self.site_dir = "www"
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.site_dir, exist_ok=True)

    def update(self, epoch):
        self.epoch_dir = f"{self.storage_dir}/{epoch:06d}"
        os.makedirs(self.epoch_dir, exist_ok=True)
