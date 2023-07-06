import numpy as np

from model import AnomalyTransformer
from data_loader import get_data_loader
import torch.nn.functional as F
import torch
from torch.optim import Adam
import os

from torch.linalg import norm

from config import train_config
from association_based import AssociationBased


class Solver:
    def __init__(self, config):

        self.train_loader = get_data_loader(
            path_config=config.path,
            batch_size=config.batch_size,
            window_size=config.window_size,
            mode="train",
            use_cuda=config.use_cuda
        )

        self.test_loader = get_data_loader(
            path_config=config.path,
            batch_size=config.batch_size,
            window_size=config.window_size,
            mode="test",
            use_cuda=config.use_cuda
        )

        self.model = AnomalyTransformer(
            N=config.window_size,
            d_model=config.d_model,
            head_num=config.head_num,
            layers=config.layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            use_cuda=config.use_cuda
        )

        self.lambda_ = config.lambda_
        self.optimizer = Adam(self.model.parameters(), lr=config.lr)
        self.epoch = config.epoch

        self.model_save_path = self._build_model_save_path(config.path.save_model)
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            verbose=config.verbose,
            delta=config.delta
        )

        self.anomaly_ratio = config.anomaly_ratio

    def _build_model_save_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def train(self):
        self.model.train()
        for n_epoch in range(self.epoch):
            min_losses, max_losses = [], []
            for step, (batch_data, _) in enumerate(self.train_loader):
                output, prior_list, series_list = self.model(batch_data)

                association_based = AssociationBased(output, prior_list, series_list, self.lambda_)
                min_loss, max_loss = association_based.minimax_loss(batch_data)
                min_losses.append(min_loss.item())
                max_losses.append(max_loss.item())

                self.optimizer.zero_grad()
                min_loss.backward(retain_graph=True)
                max_loss.backward()
                self.optimizer.step()
                # print(f"{step}th batch | min_loss: {min_loss.item():.2f} max_loss: {max_loss.item():.2f}")

                # gc.collect()
                # torch.cuda.empty_cache()

            train_min_loss, train_max_loss = np.average(min_losses), np.average(max_losses)
            with torch.no_grad():
                val_min_loss, val_max_loss = self.validation()
            print(f"Epoch: {n_epoch} | Train Min Loss: {train_min_loss:.2f} Train Max Loss: {train_max_loss:.2f} "
                  f"Val Min Loss: {val_min_loss:.2f} Val Max Loss: {val_max_loss:.2f}")

            self.early_stopping(val_min_loss, val_max_loss, self.model, self.model_save_path)
            if self.early_stopping.early_stop:
                print("Early Stopping")
                break

    def validation(self):
        self.model.eval()
        min_losses, max_losses = [], []
        for batch_data, _ in self.test_loader:
            output, prior_list, series_list = self.model(batch_data)

            association_based = AssociationBased(output, prior_list, series_list, self.lambda_)
            min_loss, max_loss = association_based.minimax_loss(batch_data)

            min_losses.append(min_loss.item())
            max_losses.append(max_loss.item())

        return np.average(min_losses), np.average(max_losses)

    def test(self):
        self.model.load_state_dict(
            torch.load(os.path.join(str(self.model_save_path), '_checkpoint.pth'))
        )
        self.model.eval()

        test_labels = []
        anomaly_scores = []
        for batch_data, label in self.test_loader:
            output, prior_list, series_list = self.model(batch_data)

            association_based = AssociationBased(output, prior_list, series_list, self.lambda_)
            score = association_based.anomaly_score(batch_data)
            anomaly_scores.append(score)

            test_labels.append(label)

        test_scores = torch.cat(anomaly_scores, dim=0).reshape(-1)
        thresh = self.get_threshold()

        # TODO: 예측값 비교부터
        test_labels = torch.cat(test_labels, dim=0).reshape(-1)
        anomaly_pred = (test_scores > thresh).astype(int)


    def get_threshold(self):
        anomaly_scores = []
        for batch_data, _ in self.train_loader:
            output, prior_list, series_list = self.model(batch_data)

            association_based = AssociationBased(output, prior_list, series_list, self.lambda_)
            score = association_based.anomaly_score(batch_data)
            anomaly_scores.append(score)

        total_scores = torch.cat(anomaly_scores, dim=0).reshape(-1)
        thresh = torch.quantile(total_scores, self.anomaly_ratio, interpolation='midpoint').item()

        return thresh

class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose

        self.counter = 0
        self.best_min_loss = np.Inf
        self.best_max_loss = np.Inf
        self.delta = delta

        self.early_stop = False

    def __call__(self, min_loss, max_loss, model, path):
        if self._is_smaller_than_best(min_loss, max_loss):
            if self.verbose:
                print(f'min loss decreased ({self.best_min_loss:.2f} --> {min_loss:.2f}).  Saving model ...')
            torch.save(model.state_dict(), os.path.join(path, '_checkpoint.pth'))
            self.best_min_loss = min_loss
            self.best_max_loss = max_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_smaller_than_best(self, min_loss, max_loss):
        return min_loss < self.best_min_loss + self.delta and max_loss < self.best_max_loss + self.delta


if __name__ == "__main__":
    solver = Solver(train_config)
    solver.train()

