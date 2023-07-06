from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

import torch


class AnomalyDataset(Dataset):
    def __init__(self, path_config, step, window_size, mode="train", use_cuda=True):
        super().__init__()
        self.mode = mode
        self.step = step
        self.window_size = window_size

        self._load_data(path_config)

        self.device = torch.device('cpu')
        if torch.cuda.is_available() and use_cuda:
            self.device = torch.device('cuda')

        torch.set_default_device(self.device)

    def _load_data(self, path_config):
        train_data = np.load(path_config.train)
        test_data = np.load(path_config.test)

        scaler = StandardScaler()
        scaler.fit(train_data)

        self.train_data = scaler.transform(train_data)
        self.test_data = scaler.transform(test_data)
        self.test_label = np.load(path_config.test_label)

        print(f"train_data: {self.train_data.shape} / test_data: {self.test_data.shape}")

    def __len__(self):
        total_len = self.train_data.shape[0]
        if self.mode == "test":
            total_len = self.test_data.shape[0]

        return (total_len - self.window_size) // self.step + 1

    def __getitem__(self, item):
        index = item * self.step

        data = self.train_data
        if self.mode == "test":
            data = self.test_data

        # 1. train mode에선 label을 사용하지 않기 때문에  None을 반환하도록 했지만
        # default_collate가 None을 batch tensor로 변환할 수 없어 오류 발생
        # 2. dtype=float64 -> float32로 변환하지 않으면 float32인 linear 함수에 입력했을 때 타입 오류 발생
        # 자동 축소는 안되는 듯
        return torch.tensor(data[index : index + self.window_size], dtype=torch.float32), \
            torch.tensor(self.test_label[index : index + self.window_size], dtype=torch.float32)


def get_data_loader(path_config, batch_size, step=1, window_size=100, mode='train', use_cuda=True):
    dataset = AnomalyDataset(path_config, step, window_size, mode, use_cuda)

    # TODO: modify data loader
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0,
                             drop_last=True)
    return data_loader
