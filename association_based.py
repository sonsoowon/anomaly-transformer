import torch.nn.functional as F
import torch

from torch.linalg import norm

class AssociationBased:
    def __init__(self, output, prior_list, series_list, lamba_):
        self.output = output
        self.prior_list = prior_list
        self.series_list = series_list
        self.lambda_ = lamba_

    def association_discrepancy(self, prior_list, series_list):  # (L, B, N, N) -> (B, N)
        layer_ass_disc = lambda pl, sl: self._rowwise_kl_div(pl, sl) + self._rowwise_kl_div(sl, pl)
        all_ass_disc = torch.stack([layer_ass_disc(prior_layer, series_layer)
                                    for prior_layer, series_layer, in zip(prior_list, series_list)], dim=0)
        return torch.mean(all_ass_disc, dim=0)

    def minimax_loss(self, x):
        return self.min_loss(x), self.max_loss(x)

    def min_loss(self, x):
        series_list = [s_layer.detach() for s_layer in self.series_list]
        return self.loss_function(x, self.output, self.prior_list, series_list, self.lambda_)

    def max_loss(self, x):
        prior_list = [p_layer.detach() for p_layer in self.prior_list]
        return self.loss_function(x, self.output, prior_list, self.series_list, self.lambda_)

    def loss_function(self, x, output, prior_list, series_list, lambda_):
        rec_loss = self._rowwise_mse_loss(x, output)
        ass_disc = self.association_discrepancy(prior_list, series_list)
        return norm(rec_loss, ord='fro') - lambda_ * norm(ass_disc, ord=1)

    def anomaly_score(self, x):  # -> (B, N, 1)
        window_size = x.shape[1]

        ass_disc = F.softmax(-self.association_discrepancy(self.prior_list, self.series_list), dim=-1)
        assert ass_disc.shape[-1] == window_size

        rec_error = self._rowwise_mse_loss(self.output, x)
        assert rec_error.shape[-1] == window_size

        return torch.mul(ass_disc, rec_error).unsqueeze(-1)

    @staticmethod
    def _rowwise_mse_loss(pred, true):  # (B, N, d_model) -> (B, N)
        loss_pointwise = (pred - true) ** 2
        return torch.mean(loss_pointwise, dim=-1)

    @staticmethod
    def _rowwise_kl_div(pred, true):  # (B, N, N) -> (B, N)
        loss_pointwise = pred * (pred.log() - true)
        return torch.sum(loss_pointwise, dim=-1) / pred.shape[-1]