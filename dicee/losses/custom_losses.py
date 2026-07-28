from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
import os
from torch import tensor
from typing import Dict, List, Optional


class DefaultBCELoss(nn.Module):
    def __init__(self):
        super(DefaultBCELoss, self).__init__()

    def forward(self, pred, target, current_epoch):

        criterion = torch.nn.BCEWithLogitsLoss()
        final_loss = criterion(pred, target)

        return final_loss

class WeightedBCELoss(nn.Module):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()

    def forward(self, pred, target, current_epoch):

        gamma = 10
        confidence = torch.abs(2 * torch.sigmoid(pred) - 1)
        weights = torch.exp(-gamma * (1 - confidence))
        weights = torch.clamp(weights, min=0.5, max=1.0)

        weights = weights.detach()

        criterion = torch.nn.BCEWithLogitsLoss(weight=weights)
        final_loss = criterion(pred, target)

        return final_loss

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothness_ratio=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothness_ratio = smoothness_ratio
        self.log_dir = "./smoothing_rates"

    def forward(self, pred, target, current_epoch):

        criterion = torch.nn.BCEWithLogitsLoss()
        final_loss = criterion(pred, target)

        return final_loss

class AdaptiveLabelSmoothingLoss(nn.Module):
    def __init__(self, min_smoothing_factor=0.01,
                 max_smoothing_factor=0.2,
                 smoothing_factor_step=0.01,
                 initial_smoothing_factor=0.1,
                 ):

        super(AdaptiveLabelSmoothingLoss, self).__init__()
        self.min_smoothing_factor = min_smoothing_factor
        self.max_smoothing_factor = max_smoothing_factor
        self.smoothing_factor_step = smoothing_factor_step
        self.smoothing_factor = initial_smoothing_factor
        self.prev_loss = None
        self.eps = 1e-14
        self.log_dir ="./smoothing_rates"

    def forward(self, logits, target, current_epoch):

        pred = F.log_softmax(logits, dim=-1) # scores converted to be used in KL

        num_classes = logits.size(-1)
        #smoothed_target = (1 - self.smoothing_factor) * target + self.smoothing_factor / num_classes
        smoothed_target = (1 - self.smoothing_factor) * target + self.smoothing_factor * (1 - target) / (num_classes - 1)

        kl_loss = F.kl_div(pred, smoothed_target, reduction="batchmean")
        loss = kl_loss

        if self.prev_loss is not None:
            loss_diff = loss.item() - self.prev_loss
            if loss_diff >= 0.0:
                self.smoothing_factor = min(self.smoothing_factor + self.smoothing_factor_step, self.max_smoothing_factor)
                #self.gamma = min(self.gamma + self.gamma_step, self.max_gamma)
            elif loss_diff <= 0.0:
                self.smoothing_factor = max(self.smoothing_factor - self.smoothing_factor_step, self.min_smoothing_factor)

        self.prev_loss = loss.item()

        return loss

class LabelRelaxationLoss(nn.Module):
    def __init__(self, alpha=0.0):
        super(LabelRelaxationLoss, self).__init__()
        self.alpha = alpha
        # Greater zero threshold
        self.gz_threshold = 0.1
        self.eps = 1e-14

    def forward(self, pred, target, current_epoch):
        probs = torch.softmax(pred, dim=-1)

        pred = pred.softmax(dim=-1)
        pred = torch.clamp(pred, min=self.eps, max=1.0)
        # Construct credal set
        with torch.no_grad():
            sum_y_hat_prime = torch.sum((torch.ones_like(target) - target) * pred, dim=-1)
            pred_hat = self.alpha * pred / torch.unsqueeze(sum_y_hat_prime, dim=-1)
            target_credal = torch.where(target > self.gz_threshold, torch.ones_like(target) - self.alpha, pred_hat)

        # Calculate divergence
        divergence = torch.sum(F.kl_div(pred.log(), target_credal, log_target=False, reduction="none"), dim=-1)
        pred = torch.sum(pred * target, dim=-1)
        result = torch.where(torch.gt(pred, 1. - self.alpha), torch.zeros_like(divergence), divergence)
        final_loss = torch.mean(result)

        return final_loss

class AdaptiveLabelRelaxationLoss(nn.Module):
    def __init__(self, min_alpha=0.01, max_alpha=0.2, alpha_step=0.01, initial_alpha=0.1):
        super(AdaptiveLabelRelaxationLoss, self).__init__()
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.alpha_step = alpha_step
        self.alpha = initial_alpha
        self.prev_loss = None
        self.eps = 1e-14
        self.gz_threshold = 0.1

    def forward(self, pred, target, current_epoch):
        probs = torch.softmax(pred, dim=-1)

        pred = pred.softmax(dim=-1)
        pred = torch.clamp(pred, min=self.eps, max=1.0)

        with torch.no_grad():
            sum_y_hat_prime = torch.sum((torch.ones_like(target) - target) * pred, dim=-1)
            pred_hat = self.alpha * pred / torch.unsqueeze(sum_y_hat_prime, dim=-1)
            target_credal = torch.where(target > self.gz_threshold, torch.ones_like(target) - self.alpha, pred_hat)

            divergence = torch.sum(F.kl_div(pred.log(), target_credal, log_target=False, reduction="none"), dim=-1)
            predc = torch.sum(pred * target, dim=-1)
            filtered_loss = torch.where(torch.gt(predc, 1. - self.alpha), torch.zeros_like(divergence),
                                        divergence)
            mean_final_loss = torch.mean(filtered_loss)

        if self.prev_loss is not None:
            loss_diff = mean_final_loss - self.prev_loss
            if loss_diff > 0:
                self.alpha = min(self.alpha + self.alpha_step, self.max_alpha)
            elif loss_diff < 0:
                self.alpha = max(self.alpha - self.alpha_step, self.min_alpha)

        self.prev_loss = mean_final_loss

        with torch.no_grad():
            pred_hat = self.alpha * pred / torch.unsqueeze(sum_y_hat_prime, dim=-1)
            target_credal = torch.where(target > self.gz_threshold, torch.ones_like(target) - self.alpha, pred_hat)

        divergence = torch.sum(F.kl_div(pred.log(), target_credal, log_target=False, reduction="none"), dim=-1)
        predc = torch.sum(pred * target, dim=-1)
        result = torch.where(torch.gt(predc, 1. - self.alpha), torch.zeros_like(divergence), divergence)
        final_loss = torch.mean(result)

        return final_loss


class ConfidenceBasedAdaptiveLabelRelaxationLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(ConfidenceBasedAdaptiveLabelRelaxationLoss, self).__init__()
        self.alpha = alpha
        # Greater zero threshold
        self.gz_threshold = 0.1
        self.eps = 1e-14

    def forward(self, pred, target, current_epoch):
        pred = pred.softmax(dim=-1)
        pred = torch.clamp(pred, min=self.eps, max=1.0)

        pred_confidence_mean = pred.mean().item()
        new_alpha = self.alpha * (1 - pred_confidence_mean)
        self.alpha = new_alpha

        # Construct credal set
        with torch.no_grad():
            sum_y_hat_prime = torch.sum((torch.ones_like(target) - target) * pred, dim=-1)
            pred_hat = self.alpha * pred / torch.unsqueeze(sum_y_hat_prime, dim=-1)
            target_credal = torch.where(target > self.gz_threshold, torch.ones_like(target) - self.alpha, pred_hat)

        # Calculate divergence
        divergence = torch.sum(F.kl_div(pred.log(), target_credal, log_target=False, reduction="none"), dim=-1)
        pred = torch.sum(pred * target, dim=-1)
        result = torch.where(torch.gt(pred, 1. - self.alpha), torch.zeros_like(divergence), divergence)
        final_loss = torch.mean(result)

        return final_loss


class CombinedLSandLR(nn.Module):
    def __init__(self, smoothness_ratio=0.0, alpha=0.0):
        super(CombinedLSandLR, self).__init__()
        self.smoothness_ratio = smoothness_ratio
        self.alpha = alpha

    def forward(self, pred, target, current_epoch):
        final_loss = 0
        if current_epoch < 20:
            criterion = LabelSmoothingLoss(smoothness_ratio=self.smoothness_ratio)
            final_loss = criterion(pred, target, current_epoch)
        else:
            criterion = LabelRelaxationLoss(alpha=self.alpha)
            final_loss = criterion(pred, target, current_epoch)
        return final_loss


class CombinedAdaptiveLSandAdaptiveLR(nn.Module):
    def __init__(self):
        super(CombinedAdaptiveLSandAdaptiveLR, self).__init__()
        self.adaptive_label_smoothing = AdaptiveLabelSmoothingLoss()
        self.adaptive_label_relaxation = AdaptiveLabelRelaxationLoss()
        self.criterion = ''

    def forward(self, pred, target, current_epoch):
        final_loss = 0
        if current_epoch < 100:
            final_loss = self.adaptive_label_smoothing(pred, target, current_epoch)
        else:
            final_loss = self.adaptive_label_relaxation(pred, target, current_epoch)
        return final_loss


class AggregatedLSandLR(nn.Module):
    def __init__(self, smoothness_ratio=0.1, alpha=0.1):
        super(AggregatedLSandLR, self).__init__()
        self.smoothness_ratio = smoothness_ratio
        self.alpha = alpha

    def forward(self, pred, target, current_epoch):
        final_loss = 0

        Smoothing_criterion = LabelSmoothingLoss(smoothness_ratio=self.smoothness_ratio)
        Smoothing_loss = Smoothing_criterion(pred, target, current_epoch)

        Relaxation_criterion = LabelRelaxationLoss(alpha=self.alpha)
        Relaxation_loss = Relaxation_criterion(pred, target, current_epoch)

        w = 0.4
        final_loss = (w * Smoothing_loss) + ((1- w) * Relaxation_loss)

        return final_loss

"""
class GradientBasedLSLR(nn.Module):
    def __init__(self, smoothness_ratio=0.0, alpha=0.0, check_interval=10, dynamic_threshold_ratio=0.015):
        super(GradientBasedLSLR, self).__init__()
        self.smoothness_ratio = smoothness_ratio
        self.alpha = alpha
        self.check_interval = check_interval
        self.dynamic_threshold_ratio = dynamic_threshold_ratio
        self.mode = 'smooth'
        self.grad_norm_history = []

        self.LabelSmoothingLoss = LabelSmoothingLoss()
        self.LabelRelaxationLoss = LabelRelaxationLoss()


    def forward(self, pred, target, current_epoch, gradient_norm):
        if len(self.grad_norm_history) == self.check_interval:
            self.grad_norm_history.pop(0)
        self.grad_norm_history.append(gradient_norm)

        avg_norm = sum(self.grad_norm_history) / len(self.grad_norm_history) + 1e-14 # avoid division by zero

        if current_epoch != 0:
            if avg_norm < self.dynamic_threshold_ratio and self.mode == 'smooth':
                self.mode = 'relax'

        #final_loss = 0
        if self.mode == 'smooth':
            final_loss = self.LabelSmoothingLoss(pred, target, current_epoch, gradient_norm)
        else:
            final_loss = self.LabelRelaxationLoss(pred, target, current_epoch, gradient_norm)

        return final_loss


class GradientBasedAdaptiveLSLR(nn.Module):
    def __init__(self, smoothness_ratio=0.0, alpha=0.0, check_interval=10,
                 variability_threshold=0.09):
        super(GradientBasedAdaptiveLSLR, self).__init__()
        self.smoothness_ratio = smoothness_ratio
        self.alpha = alpha
        self.check_interval = check_interval
        self.variability_threshold = variability_threshold
        self.mode = 'smooth'
        self.grad_norm_history = []
        self.adaptive_label_smoothing = AdaptiveLabelSmoothingLoss()
        self.adaptive_label_relaxation = AdaptiveLabelRelaxationLoss()

    def update_dynamic_threshold(self, current_epoch):
        if len(self.grad_norm_history) < self.check_interval:
            return

        std_dev = np.std(self.grad_norm_history)
        print(std_dev, self.variability_threshold)
        if std_dev < self.variability_threshold and self.mode == 'smooth':
            self.mode = 'relax'

    def forward(self, pred, target, current_epoch, gradient_norm):
        if len(self.grad_norm_history) == self.check_interval:
            self.grad_norm_history.pop(0)
        self.grad_norm_history.append(gradient_norm)

        if current_epoch % self.check_interval == 0:
            self.update_dynamic_threshold(current_epoch)

        if self.mode == 'smooth':
            return self.adaptive_label_smoothing(pred, target, current_epoch, gradient_norm)
        else:
            return self.adaptive_label_relaxation(pred, target, current_epoch, gradient_norm)

"""

class ACLS(nn.Module):

    def __init__(self,
                 pos_lambda: float = 1.0,
                 neg_lambda: float = 0.1,
                 alpha: float = 0.1,
                 margin: float = 10.0,
                 num_classes: int = 200,
                 ignore_index: int = -100):
        super().__init__()
        self.pos_lambda = pos_lambda
        self.neg_lambda = neg_lambda
        self.alpha = alpha
        self.margin = margin
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "reg"

    def get_reg(self, inputs, targets):
        max_values, indices = inputs.max(dim=1)
        max_values = max_values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        indicator = (max_values.clone().detach() == inputs.clone().detach()).float()

        batch_size, num_classes = inputs.size()
        num_pos = batch_size * 1.0
        num_neg = batch_size * (num_classes - 1.0)

        neg_dist = max_values.clone().detach() - inputs

        pos_dist_margin = F.relu(max_values - self.margin)
        neg_dist_margin = F.relu(neg_dist - self.margin)

        pos = indicator * pos_dist_margin ** 2
        neg = (1.0 - indicator) * (neg_dist_margin ** 2)

        reg = self.pos_lambda * (pos.sum() / num_pos) + self.neg_lambda * (neg.sum() / num_neg)
        return reg

    def forward(self, inputs, targets, current_epoch):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))  # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        loss_ce = self.cross_entropy(inputs, targets)

        loss_reg = self.get_reg(inputs, targets)
        loss = loss_ce + self.alpha * loss_reg

        return loss
    
#My Work

class UNITEI(nn.Module):
    """
    UNITE-I loss (pattern-aware & noise-resilient) for mini-batch training.
    """

    def __init__(
        self,
        gamma: float = 5,     
        sigma: float = 1000,  
        lambda_q: float = 0.3,  # Regularization term
        clamp_scores: bool = True,
        use_tanh_scale: bool = True,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.sigma = float(sigma)
        self.lambda_q = float(lambda_q)
        self.clamp_scores = clamp_scores
        self.use_tanh_scale = use_tanh_scale
        self.eps = 1e-12

    def _normalize_scores(self, pred: torch.Tensor) -> torch.Tensor:
        # To keep gamma meaningful and gradients don't die or explode.
        x = pred
        if self.use_tanh_scale:
            # squashes to (-1,1), then scales to (-gamma, gamma)
            x = torch.tanh(x) * self.gamma
        if self.clamp_scores:
            x = torch.clamp(x, min=-self.gamma, max=self.gamma)
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor, current_epoch = None):

        target = target.float()
        pred = self._normalize_scores(pred)

        tau_pos = F.relu(self.gamma - pred)  # only where target==1 used
        tau_neg = F.relu(pred + self.gamma)  # only where target==0 used #changed - to +

        #Quadratic penalties
        loss_pos = self.sigma * (tau_pos ** 2)
        loss_neg = self.sigma * (tau_neg ** 2)

        # Constraint penalties with Q(x) = sigmoid(x)
        # Pos: Q(gamma - f) >= Q(tau) → violation = ReLU(Q(tau) - Q(gamma - f)) where f=pred
        # Neg: Q(-gamma - f) >= Q(tau) → violation = ReLU(Q(tau) - Q(-gamma - f))
        Q_tau_pos = torch.sigmoid(tau_pos)
        Q_tau_neg = torch.sigmoid(tau_neg)
        Q_pos = torch.sigmoid(self.gamma - pred)
        Q_neg = torch.sigmoid(-self.gamma - pred) #changed from pred - self.gamma to -self.gamma - pred

        pos_violation = F.relu(Q_tau_pos - Q_pos)
        neg_violation = F.relu(Q_tau_neg - Q_neg)

        # Mask by labels and combine
        pos_mask = target
        neg_mask = 1.0 - target

        loss = (
            pos_mask * (loss_pos + self.lambda_q * pos_violation)
            + neg_mask * (loss_neg + self.lambda_q * neg_violation)
        )

        return loss.mean()


class RoBoSS(nn.Module):

    def __init__(self, a_roboss=5.0, lambda_roboss=1.5, margin=1.0, normalize_scores=False):
        super().__init__()
        self.a_roboss = float(a_roboss)
        self.lambda_roboss = float(lambda_roboss)
        self.margin = float(margin)
        self.normalize_scores = normalize_scores
        self.eps = 1e-8

    def _normalize_scores(self, pred):
        if self.normalize_scores:
            return torch.tanh(pred)
        return pred

    def forward(self, pred, target, current_epoch = None):
        pred_dtype = pred.dtype
        target = target.float()
        pred = self._normalize_scores(pred.float())
        
        # Convert [0, 1] to [-1, 1]
        if target.min() >= 0 and target.max() <= 1:
            target = 2 * target - 1
        
        u = self.margin - (target * pred)
        
        term1 = (self.a_roboss * u) + 1.0
        # Clamp exponent to avoid inf/NaN gradients in mixed precision.
        exp_arg = torch.clamp(-self.a_roboss * u, min=-50.0, max=50.0)
        term2 = torch.exp(exp_arg)

        loss_values = self.lambda_roboss * (1.0 - (term1 * term2))
        
        # Apply condition: Loss is 0 if u <= 0 (Correctly classified)
        # assign a fixed loss of 0 for all samples with u < 0
        loss = torch.where(u > 0, loss_values, torch.zeros_like(loss_values))

        return loss.mean().to(pred_dtype)


class AGCELoss(nn.Module):

    def __init__(self, agce_a=0.1, agce_q=1.0, eps=1e-8, scale=1.0):
        super(AGCELoss, self).__init__()
        self.agce_a = float(agce_a)
        self.agce_q = float(agce_q)
        self.eps = eps
        self.scale = scale
        
        # According to Corollary 1 in the paper:
        # - If q > 1: Loss is "Completely Asymmetric" (Robust).
        # - If q <= 1: Robustness depends on 'a' being large enough.
        if self.agce_q <= 1:
            print(f"Warning: q={agce_q} (<=1). Ensure 'a' is tuned high enough for noise robustness.")

    def forward(self, pred, labels, current_epoch = None):
        pred_dtype = pred.dtype
        pred = pred.float()
        y = labels.float()
        y_hat = torch.sigmoid(pred)
        u = y * y_hat + (1 - y) * (1 - y_hat)
        u = torch.clamp(u, min=self.eps, max=1.0 - self.eps)

        q = float(self.agce_q)
        if abs(q) < 1e-4:
            # Numerically stable q -> 0 limit:
            # ((a+1)^q - (a+u)^q) / q -> log((a+1)/(a+u))
            loss = torch.log((self.agce_a + 1.0) / (self.agce_a + u))
        else:
            term1 = (self.agce_a + 1.0) ** q
            term2 = (self.agce_a + u) ** q
            loss = (term1 - term2) / q

        return (loss.mean() * self.scale).to(pred_dtype)


class AULoss(nn.Module):
    def __init__(self, aul_a=1.5, aul_p=0.9, eps=1e-7, scale=1.0):
        super(AULoss, self).__init__()
        self.aul_a = float(aul_a)
        self.aul_p = float(aul_p)
        self.eps = eps
        self.scale = scale

        # assert self.aul_a > 1.0, "Parameter 'aul_a' must be > 1.0 for AUL."

    def forward(self, pred, labels, current_epoch = None):
        pred_dtype = pred.dtype
        pred = pred.float()
        y = labels
        #Map scores/logits to probabilities [0, 1]
        y_hat = torch.sigmoid(pred)
        
        # Probability of the CORRECT classification)
        # p_target = y * P(y=1) + (1-y) * P(y=0)
        u = y * y_hat + (1 - y) * (1 - y_hat)

        # Clamp for numerical stability in power operations
        u = torch.clamp(u, min=self.eps, max=1.0 - self.eps)
        
        # 3. Apply AUL Formula: ((aul_a - p_target)^aul_p - (aul_a - 1)^aul_p) / aul_p
        term1 = (self.aul_a - u) ** self.aul_p
        term2 = (self.aul_a - 1) ** self.aul_p
        
        loss = (term1 - term2) / self.aul_p
        
        return (loss.mean() * self.scale).to(pred_dtype)


class AELoss(nn.Module):
    def __init__(self, a_ael=0.5, eps=1e-7, scale=1.0):
        super(AELoss, self).__init__()
        self.a_ael = float(a_ael)
        self.eps = eps
        self.scale = scale

        # Validation based on paper constraints
        assert self.a_ael > 0, "Parameter 'a' must be > 0."

    def forward(self, pred, labels, current_epoch=None):
        pred_dtype = pred.dtype
        pred = torch.sigmoid(pred.float())
        labels = labels.float()
        p_target = labels * pred + (1 - labels) * (1 - pred)

        p_target = torch.clamp(p_target, min=self.eps, max=1.0)
        
        loss = torch.exp(-p_target / self.a_ael)
        
        return (loss.mean() * self.scale).to(pred_dtype)


class EwLoss(nn.Module):
    """
    Corrected Implementation of EwLoss (Exponential Weighted Loss) from
    "EwLoss: A Exponential Weighted Loss Function for Knowledge Graph Embedding Models"
    (Shen et al., IEEE Access, 2023).

    The loss re-weights the negative samples with an exponential coefficient so that
    hard-to-distinguish negatives contribute more to the objective.

    Paper Reference:
      - "Assigns weights... to decrease the weights of easily discriminated samples" [cite: 205]
      - Formula: EwLoss = -log σ(γ - f_pos) - Σ_j (p_j)^α * log σ(f_neg_j - γ) [cite: 249]
    Parameters
    ----------
    margin : float, default=1.0
        γ in the paper.
    alpha : float, default=0.9
        Exponential weight coefficient applied to the normalized negative scores.
    temperature : float, default=1.0
        Softmax temperature used when computing p_j.
    negative_sample_size : int or None, default=512
        Number of negative scores to consider per (h, r) pair.
    normalize_scores : bool, default=False
        Whether to normalize logits with tanh before computing the loss.
    """
    def __init__(self,
                 margin: float = 3.0,
                 alpha: float = 0.1,
                 temperature: float = 1.0,
                 negative_sample_size: int = 512,
                 sample_mode: str = "random",
                 normalize_scores: bool = False,
                 pos_weight: float = 1.0,
                 neg_weight: float = 1.0,
                 positive_threshold: float = 0.5,
                 eps: float = 1e-12):
        super().__init__()
        self.margin = float(margin)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.negative_sample_size = None
        if negative_sample_size is not None and negative_sample_size > 0:
            self.negative_sample_size = int(negative_sample_size)
        self.sample_mode = sample_mode if sample_mode in {"random", "topk"} else "random"
        self.normalize_scores = normalize_scores
        self.pos_weight = float(pos_weight)
        self.neg_weight = float(neg_weight)
        self.positive_threshold = float(positive_threshold)
        self.eps = float(eps)

    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        if self.normalize_scores:
            return torch.tanh(scores)
        return scores

    def _sample_negatives(self, scores: torch.Tensor) -> torch.Tensor:
        if self.negative_sample_size is None or scores.numel() <= self.negative_sample_size:
            return scores
        
        # Optimization: For hard negative mining, we might want the smallest distances
        if self.sample_mode == "topk":
            # If scores are distances, "hardest" are the smallest values
            topk_vals, _ = torch.topk(scores, self.negative_sample_size, largest=False)
            return topk_vals
        
        # Random sampling (default)
        idx = torch.randperm(scores.numel(), device=scores.device)[:self.negative_sample_size]
        return scores[idx]

    def forward(self, pred: torch.Tensor, target: torch.Tensor, current_epoch=None) -> torch.Tensor:
        """
        Args:
            pred: Prediction scores (Distances: Lower is better/closer).
            target: Binary labels (1 for positive, 0 for negative).
        """
        target = target.float()
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        
        pred = pred.float()
        pred = self._normalize_scores(pred)
        positive_mask = target >= self.positive_threshold

        batch_loss = pred.new_tensor(0.0)
        valid_rows = 0

        for row_pred, row_mask in zip(pred, positive_mask):
            pos_scores = row_pred[row_mask]
            neg_scores = row_pred[~row_mask]

            if pos_scores.numel() == 0 or neg_scores.numel() == 0:
                continue

            neg_scores = self._sample_negatives(neg_scores)

            # --- 1. Positive Term ---
            # Minimize distance: push pos_scores < margin
            pos_term = F.softplus(pos_scores - self.margin).mean()

            # --- 2. Calculate Weights ---
            # In distance metrics, Hard Negatives = Small Distance.
            # Easy Negatives = Large Distance.
            # Softmax(x) creates high prob for large x.
            # To weight Hard Negatives higher, we must negate the distance.
            # Eq 17: p(h') = exp(f(h')) / sum(...) where f is a score (inverse of distance)
            scaled_neg_for_weights = -neg_scores / max(self.temperature, self.eps)
            
            probs = torch.softmax(scaled_neg_for_weights, dim=0)
            weights = torch.pow(probs + self.eps, self.alpha)

            # --- 3. Negative Term ---
            # Maximize distance: push neg_scores > margin
            # Eq 18: - sum( weights * log sigmoid(neg_scores - gamma) )
            # Note: logsigmoid(x) is always negative. We want to maximize it towards 0.
            log_sig = F.logsigmoid(neg_scores - self.margin)
            
            # We use a weighted average (dividing by weights.sum()) for numerical stability
            # though the paper Eq 18 implies a direct sum.
            neg_term = -(weights * log_sig).sum() / (weights.sum() + self.eps)

            row_loss = self.pos_weight * pos_term + self.neg_weight * neg_term
            batch_loss = batch_loss + row_loss
            valid_rows += 1

        if valid_rows == 0:
            return pred.new_tensor(0.0, requires_grad=True)

        return batch_loss / valid_rows


class SynMarginLoss(nn.Module):

    def __init__(self,
                 mode: str = "projection",
                 margin: float = 0.1,
                 temperature: float = 1.0,
                 detach_negative: bool = True,
                 eps: float = 1e-8):
        super().__init__()
        assert mode in {"projection", "difference"}, "mode must be projection or difference"
        self.mode = mode
        self.margin = margin
        self.temperature = temperature
        self.detach_negative = detach_negative
        self.eps = eps
        self.embedding_layer = None

    def bind_embedding_layer(self, embedding_layer: nn.Embedding) -> None:
        """
        Attach the entity embedding layer so that the loss can retrieve target vectors.
        """
        self.embedding_layer = embedding_layer

    def forward(self, pred: torch.Tensor, target: torch.Tensor, current_epoch=None):
        if self.embedding_layer is None:
            raise RuntimeError("SynMarginLoss requires entity embeddings. Call bind_embedding_layer first.")

        entity_vectors = self.embedding_layer.weight
        num_entities = entity_vectors.shape[0]

        pred = self._reshape_scores(pred, num_entities, "predictions")
        target = self._reshape_scores(target, num_entities, "targets")

        prob = torch.softmax(pred / self.temperature, dim=-1)
        pred_emb = prob @ entity_vectors
        pred_emb = F.normalize(pred_emb, p=2, dim=-1, eps=self.eps)

        target_weights = torch.clamp(target, min=0.0)
        mass = target_weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        target_probs = target_weights / mass
        target_emb = target_probs @ entity_vectors
        target_emb = F.normalize(target_emb, p=2, dim=-1, eps=self.eps)

        neg = self._synthesize_negative(pred_emb, target_emb)
        if self.detach_negative:
            neg = neg.detach()

        pos_sim = (pred_emb * target_emb).sum(dim=-1)
        neg_sim = (pred_emb * neg).sum(dim=-1)
        loss = torch.relu(self.margin + neg_sim - pos_sim)
        return loss.mean()

    def _reshape_scores(self, tensor: torch.Tensor, num_entities: int, name: str) -> torch.Tensor:
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() > 2:
            tensor = tensor.reshape(tensor.shape[0], -1)

        if tensor.dim() != 2:
            raise ValueError(f"SynMarginLoss expects 2-D {name}, got shape {tensor.shape}")

        if tensor.shape[1] != num_entities:
            if tensor.numel() % num_entities == 0:
                tensor = tensor.view(-1, num_entities)
            else:
                raise RuntimeError(
                    f"SynMarginLoss requires {name} dimension to match number of entities "
                    f"({num_entities}), got shape {tensor.shape}. "
                    f"Please use a scoring technique that produces scores for every entity (e.g., KvsAll/AllvsAll/1vsAll)."
                )
        return tensor

    def _synthesize_negative(self, pred_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        if self.mode == "difference":
            neg = pred_emb - target_emb
        else:
            dot = (pred_emb * target_emb).sum(dim=-1, keepdim=True)
            neg = pred_emb - dot * target_emb
            zero_mask = neg.norm(dim=-1, keepdim=True) <= self.eps
            if zero_mask.any():
                fallback = pred_emb - target_emb
                neg = torch.where(zero_mask, fallback, neg)
        neg = F.normalize(neg, p=2, dim=-1, eps=self.eps)
        return neg


class WaveLoss(nn.Module):
    def __init__(self, wave_a = 1.5, lambda_param = 0.5, eps = 1e-8):
        super().__init__()
        self.wave_a = wave_a
        self.lambda_param = lambda_param
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, current_epoch = None):
        pred_dtype = pred.dtype
        target = target.float()
        pred = pred.float()

        if target.min() >= 0 and target.max() <= 1:
            target = 2 * target - 1 

        u = 1.0 - (target * pred)

        u_sqr = u * u
        
        exp_arg = torch.clamp(self.wave_a * u, min=-50.0, max=50.0)
        exp_term = torch.exp(exp_arg)
        
        denom = 1.0 + self.lambda_param * u_sqr * exp_term + self.eps

        wave_loss = (1.0 / self.lambda_param) * (1.0 - 1.0 / denom)

        return wave_loss.mean().to(pred_dtype)

class NSSALoss(nn.Module):
    """
    Negative Sampling Self-Adversarial (NSSA) loss.
    """

    def __init__(
        self,
        nssa_alpha = 1.0,
        positive_threshold = 0.5, #Semantic threshlod for postive classes
    ):
        super().__init__()
        self.nssa_alpha = nssa_alpha
        self.positive_threshold = positive_threshold

    # def forward(
    #     self,
    #     pred,
    #     target,
    #     current_epoch=None
    # ):

    #     pos_mask = target > self.positive_threshold
    #     neg_mask = ~pos_mask

    #     # Handle 1D targets (single row) vs batched 2D targets.
    #     if target.dim() == 1:
    #         if not (pos_mask.any() and neg_mask.any()):
    #             return pred.new_tensor(0.0, requires_grad=True)
    #     else:
    #         # Masking invalid rows
    #         valid_mask = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    #         if not valid_mask.any():
    #             return pred.new_tensor(0.0, requires_grad=True)

    #         pred = pred[valid_mask]
    #         pos_mask = pos_mask[valid_mask]
    #         neg_mask = neg_mask[valid_mask]

    #     pos_scores = pred.masked_select(pos_mask)
    #     pos_loss = -F.logsigmoid(pos_scores).mean()

    #     neg_scores = pred.masked_select(neg_mask)
    #     weights = F.softmax(neg_scores * self.nssa_alpha, dim=0).detach()
    #     neg_loss = -(weights * F.logsigmoid(-neg_scores)).sum()

    #     return (pos_loss + neg_loss) / 2
    def forward(self, pred, target, current_epoch=None):
    
        pos_mask = target > self.positive_threshold
        neg_mask = ~pos_mask
        
        pos_scores = pred[pos_mask]
        pos_loss = -F.logsigmoid(pos_scores).mean()
        
        neg_loss = 0
        batch_size = pred.size(0)
        
        for i in range(batch_size):
            row_neg_scores = pred[i][neg_mask[i]]
            if row_neg_scores.numel() == 0:
                continue
                
            weights = F.softmax(row_neg_scores * self.nssa_alpha, dim=0).detach()
            neg_loss += -(weights * F.logsigmoid(-row_neg_scores)).sum()
            
        return (pos_loss + (neg_loss / batch_size)) / 2

class FocalLoss(nn.Module):

    def __init__(self, gamma = 2.0, alpha = 0.25):
        super().__init__() 

        self.gamma = gamma 
        self.alpha = alpha 

    def forward(self, pred, target, current_epoch = None):

        p = torch.sigmoid(pred).clamp(1e-6, 1 - 1e-6)

        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        pt = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss

        return loss.mean()


class LocalTripleLoss(nn.Module):
    """
    Local triple loss with per-triple confidence updates (CKRL local confidence)
    """

    requires_x_batch = True

    def __init__(
        self,
        margin = 1.0,
        alpha = 0.9,
        beta = 1e-4,
        min_conf = 0.0,
        max_conf = 1.0,
        positive_threshold = 0.5,
        use_max_negative = True,
        score_is_distance = False
    ):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.min_conf = min_conf
        self.max_conf = max_conf
        self.positive_threshold = positive_threshold
        self.use_max_negative = use_max_negative
        self.score_is_distance = score_is_distance
        self._confidence = {}


    def forward(self, pred, target, current_epoch = None, x_batch = None):

        if pred.dim() > 1:
            pred = pred.reshape(-1)
            target = target.reshape(-1)

        pos_mask = target > self.positive_threshold
        pos_scores = pred[pos_mask]
        neg_scores = pred[~pos_mask]

        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return pred.new_tensor(0.0, requires_grad = True) 

        pos_count = pos_scores.numel()
        neg_count = neg_scores.numel()
        if neg_count % pos_count == 0: #Checks if negatives are evenly aligned with positives
            neg_ratio = neg_count // pos_count 
            neg_scores = neg_scores.view(neg_ratio, pos_count) #neg_scores.shape == [neg_ratio, pos_count]
            if self.use_max_negative:
                neg_per_pos = neg_scores.max(dim = 0).values 
            else:
                neg_per_pos = neg_scores.mean(dim = 0)
        else:
            neg_value = neg_scores.max() if self.use_max_negative else neg_scores.mean()
            neg_per_pos = neg_value.expand_as(pos_scores)
            #Every positive is compared against the same negative score

        if self.score_is_distance:
            e_pos = pos_scores 
            e_neg = neg_per_pos 
        else:
            e_pos = -pos_scores
            e_neg = -neg_per_pos #larger the better assumption

        delta = self.margin + e_pos - e_neg 
        hinge = F.relu(delta) 

        pos_triples = x_batch[pos_mask].detach().cpu().tolist()
        #Initially all triples have a confidence score of 1.0
        conf_values = [self._confidence.get(tuple(triple), 1.0) for triple in pos_triples] 
        conf = pred.new_tensor(conf_values)

        loss = (conf * hinge).mean()

        q_values = (-delta).detach() #Energy value - Equation 5
        for triple, q_val, conf_val in zip(pos_triples, q_values.tolist(), conf_values):
            #Equation 6
            if q_val <= 0:
                new_conf = max(conf_val * self.alpha, self.min_conf) #Bad priples punished
            else:
                new_conf = min(conf_val + self.beta, self.max_conf) #caps good triples back to 1.0
            self._confidence[tuple(triple)] = new_conf 
        
        return loss 

def compute_prior_path_confidence(path_set, rel_path_prior, path_prior, epsilon = 1e-6):
    """
    Compute prior path confidence (PP) for a triple.

    path_set: iterable of (path_id, reliability) where reliability is R(h,p,t)
    rel_path_prior: dict mapping path_id -> P(r, p)
    path_prior: dict mapping path_id -> P(p)
    """
    pp = 0.0
    for path_id, reliability in path_set:
        p_rp = rel_path_prior.get(path_id, 0.0)
        p_p = path_prior.get(path_id, 0.0)
        q_pp = epsilon + (1.0 - epsilon) * (p_rp / max(p_p, epsilon))
        pp += q_pp * reliability
    return pp


class LocalTripleWithPriorPathLoss(nn.Module):
    """
    local triple confidence (LT) with prior path confidence (PP).
    """

    requires_x_batch = True

    def __init__(
        self,
        margin = 1.0,
        alpha = 0.9,
        beta = 1e-4,
        min_conf = 0.0,
        max_conf = 1.0,
        positive_threshold = 0.5,
        use_max_negative = True,
        score_is_distance = False,
        lambda_1_lt = 1.5,
        lambda_2_pp = 0.1,
        prior_confidence_map = None,
    ):
        super().__init__()
        self.local_loss = LocalTripleLoss(
            margin = margin,
            alpha = alpha,
            beta = beta,
            min_conf = min_conf,
            max_conf = max_conf,
            positive_threshold = positive_threshold,
            use_max_negative = use_max_negative,
            score_is_distance = score_is_distance,
        )
        self.lambda_1_lt = float(lambda_1_lt)
        self.lambda_2_pp = float(lambda_2_pp)
        self._prior_confidence = prior_confidence_map or {}

    def set_prior_confidence_map(self, confidence_map): #stores the PP lookup table inside the loss
        self._prior_confidence = confidence_map or {}

    def _get_prior_confidence(self, pos_triples, device, dtype): #retrieves PP values from the table for the current batch of triples
        if not self._prior_confidence:
            return torch.zeros(len(pos_triples), device = device, dtype = dtype)
        values = [self._prior_confidence.get(tuple(triple), 0.0) for triple in pos_triples]
        return torch.tensor(values, device = device, dtype = dtype)

    def forward(self, pred, target, current_epoch = None, x_batch = None):

        if pred.dim() > 1:
            pred = pred.reshape(-1)
            target = target.reshape(-1)

        pos_mask = target > self.local_loss.positive_threshold
        pos_scores = pred[pos_mask]
        neg_scores = pred[~pos_mask]

        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return pred.new_tensor(0.0, requires_grad = True)

        pos_count = pos_scores.numel()
        neg_count = neg_scores.numel()
        if neg_count % pos_count == 0:
            neg_ratio = neg_count // pos_count
            neg_scores = neg_scores.view(neg_ratio, pos_count)
            if self.local_loss.use_max_negative:
                neg_per_pos = neg_scores.max(dim = 0).values
            else:
                neg_per_pos = neg_scores.mean(dim = 0)
        else:
            neg_value = neg_scores.max() if self.local_loss.use_max_negative else neg_scores.mean()
            neg_per_pos = neg_value.expand_as(pos_scores)

        if self.local_loss.score_is_distance:
            e_pos = pos_scores
            e_neg = neg_per_pos
        else:
            e_pos = -pos_scores
            e_neg = -neg_per_pos

        delta = self.local_loss.margin + e_pos - e_neg
        hinge = F.relu(delta)

        pos_triples = x_batch[pos_mask].detach().cpu().tolist()
        conf_values = [self.local_loss._confidence.get(tuple(triple), 1.0) for triple in pos_triples]
        local_conf = pred.new_tensor(conf_values)
        prior_conf = self._get_prior_confidence(pos_triples, device = pred.device, dtype = pred.dtype)
        combined_conf = self.lambda_1_lt * local_conf + self.lambda_2_pp * prior_conf
        loss = (combined_conf * hinge).mean()

        q_values = (-delta).detach()
        for triple, q_val, conf_val in zip(pos_triples, q_values.tolist(), conf_values):
            if q_val <= 0:
                new_conf = max(conf_val * self.local_loss.alpha, self.local_loss.min_conf)
            else:
                new_conf = min(conf_val + self.local_loss.beta, self.local_loss.max_conf)
            self.local_loss._confidence[tuple(triple)] = new_conf

        return loss


class LocalTripleWithPriorAndAdaptivePathLoss(nn.Module):
    """
    Combine local triple confidence (LT), prior path confidence (PP),
    and adaptive path confidence (AP).
    """

    requires_x_batch = True
    requires_model = True

    def __init__(
        self,
        margin = 1.0,
        alpha = 0.9,
        beta = 1e-4,
        min_conf = 0.0,
        max_conf = 1.0,
        positive_threshold = 0.5,
        use_max_negative = True,
        score_is_distance = False,
        lambda_1_lt = 1.5,
        lambda_2_pp = 0.1,
        lambda_3_ap = 0.4,
        adaptive_use_l1 = True,
        prior_confidence_map = None,
        max_paths_per_triple = 30,
    ):
        super().__init__()
        self.local_loss = LocalTripleLoss(
            margin = margin,
            alpha = alpha,
            beta = beta,
            min_conf = min_conf,
            max_conf = max_conf,
            positive_threshold = positive_threshold,
            use_max_negative = use_max_negative,
            score_is_distance = score_is_distance,
        )
        self.lambda_1_lt = lambda_1_lt
        self.lambda_2_pp = lambda_2_pp
        self.lambda_3_ap = lambda_3_ap
        self.adaptive_use_l1 = adaptive_use_l1
        self._prior_confidence = prior_confidence_map or {}
        self._path_data = {}
        # cap on paths kept per triple (top-K by reliability)
        self.max_paths_per_triple = int(max_paths_per_triple) if max_paths_per_triple else None
        # Pre-built (N, K, L) buffers — populated lazily on first forward call
        # (num_relations is only available once model embeddings exist).
        self._path_rel_ids_buf  = None   # (N, K, L) long
        self._path_signs_buf    = None   # (N, K, L) float32
        self._path_weights_buf  = None   # (N, K)    float32
        self._path_mask_buf     = None   # (N, K)    bool
        self._path_triple_to_idx: dict = {}

    def set_prior_confidence_map(self, confidence_map):
        self._prior_confidence = confidence_map or {}

    def set_path_data(self, path_data):
        """Store path data, truncate to top-K, and mark buffers as needing rebuild."""
        if not path_data:
            self._path_data = {}
            self._path_rel_ids_buf = None
            return
        cap = self.max_paths_per_triple
        if cap and cap > 0:
            self._path_data = {
                k: sorted(v, key=lambda x: -float(x[1]))[:cap]
                for k, v in path_data.items()
            }
        else:
            self._path_data = path_data
        # Buffers will be built on first _get_adaptive_confidence call
        # (we need num_relations from model.relation_embeddings at that point).
        self._path_rel_ids_buf = None

    def _build_path_buffers(self, num_relations: int):
        """Pre-build (N, K, L) arrays from self._path_data.

        Symbols
        -------
        N : number of triples with at least one path
        K : max paths per triple
        L : max path length in hops
        """
        import numpy as np
        triples_list = list(self._path_data.keys())
        N = len(triples_list)
        K = max(len(v) for v in self._path_data.values())
        L = max(
            (len(p) for v in self._path_data.values() for p, _ in v),
            default=1,
        )
        L = max(L, 1)

        rel_ids = np.zeros((N, K, L), dtype=np.int64)
        # signs=0 for padding so padding hops contribute 0 to path vector
        signs   = np.zeros((N, K, L), dtype=np.float32)
        weights = np.zeros((N, K),    dtype=np.float32)
        mask    = np.zeros((N, K),    dtype=np.bool_)

        self._path_triple_to_idx = {}
        for i, triple in enumerate(triples_list):
            self._path_triple_to_idx[triple] = i
            path_list = self._path_data[triple]
            # Guard: skip if all reliabilities are zero (degenerate)
            if all(float(r) <= 0 for _, r in path_list):
                continue
            for j, (rel_path, reliability) in enumerate(path_list):
                if float(reliability) <= 0:
                    continue  # skip zero-prob individual paths
                for l, pid in enumerate(rel_path):
                    pid = int(pid)
                    if pid < num_relations:
                        rel_ids[i, j, l] = pid
                        signs[i, j, l]   = 1.0    # forward
                    else:
                        rel_ids[i, j, l] = pid - num_relations
                        signs[i, j, l]   = -1.0   # inverse
                # Store raw reliability (NOT normalised by z) to match the original
                # scalar loop:  all_path_conf += float(pr) / dist  (no /z).
                weights[i, j] = float(reliability)
                mask[i, j]    = True

        self._path_rel_ids_buf  = torch.from_numpy(rel_ids)
        self._path_signs_buf    = torch.from_numpy(signs)
        self._path_weights_buf  = torch.from_numpy(weights)
        self._path_mask_buf     = torch.from_numpy(mask)

    def _gather_path_tensors(self, batch_triples_list, rel_emb, device, dtype):
        """Gather pre-built buffers for the batch and compute path vectors.

        Returns
        -------
        path_vecs : (B, K, D)
        weights   : (B, K)
        mask      : (B, K) bool
        """
        raw_idx   = [self._path_triple_to_idx.get(tuple(t), -1)
                     for t in batch_triples_list]
        has_paths = torch.tensor([i >= 0 for i in raw_idx], device=device)
        safe_idx  = torch.tensor([i if i >= 0 else 0 for i in raw_idx],
                                 dtype=torch.long)

        rel_ids_b = self._path_rel_ids_buf[safe_idx].to(device)
        signs_b   = self._path_signs_buf[safe_idx].to(device, dtype)
        weights_b = self._path_weights_buf[safe_idx].to(device, dtype)
        mask_b    = self._path_mask_buf[safe_idx].to(device)

        if not has_paths.all():
            no_path   = ~has_paths
            weights_b = weights_b.clone(); weights_b[no_path] = 0.0
            mask_b    = mask_b.clone();    mask_b[no_path]    = False

        # (B, K, L, D) → sum over L → (B, K, D)
        emb_seq   = rel_emb[rel_ids_b]
        path_vecs = (signs_b.unsqueeze(-1) * emb_seq).sum(dim=2)
        return path_vecs, weights_b, mask_b

    def _get_prior_confidence(self, pos_triples, device, dtype):
        if not self._prior_confidence:
            return torch.zeros(len(pos_triples), device = device, dtype = dtype)
        values = [self._prior_confidence.get(tuple(triple), 0.0) for triple in pos_triples]
        return torch.tensor(values, device = device, dtype = dtype)

    def _calc_path_distance(self, rel_id, rel_path, rel_emb, num_relations):
        vec = rel_emb[rel_id].clone()
        for pid in rel_path:
            pid = int(pid)
            if pid < num_relations:
                vec = vec - rel_emb[pid]
            else:
                vec = vec + rel_emb[pid - num_relations]
        if self.adaptive_use_l1:
            return vec.abs().sum()
        return (vec * vec).sum()

    def _get_adaptive_confidence(self, pos_triples, model, device, dtype):
        """Compute AP confidence for each positive triple.

        Fully vectorised: builds (N,K,L) buffers on first call, then uses a
        single tensor gather + distance op per forward call — zero Python loops
        in steady state.
        """
        if not self._path_data or not hasattr(model, "relation_embeddings"):
            return torch.zeros(len(pos_triples), device=device, dtype=dtype)

        rel_emb = model.relation_embeddings.weight.detach()  # (R, D)
        num_relations = rel_emb.shape[0]

        # Build buffers lazily on first call (num_relations known only here).
        if self._path_rel_ids_buf is None:
            self._build_path_buffers(num_relations)

        path_vecs, weights, mask = self._gather_path_tensors(
            pos_triples, rel_emb, device, dtype
        )  # (B,K,D), (B,K), (B,K)

        rel_ids = torch.tensor([int(t[1]) for t in pos_triples],
                               device=device, dtype=torch.long)   # (B,)
        r_embs  = rel_emb[rel_ids]                                # (B, D)

        # d(r, p) = ||r_emb - path_vec|| matching _calc_path_distance exactly.
        diffs = r_embs.unsqueeze(1) - path_vecs   # (B, K, D)
        if self.adaptive_use_l1:
            dists = diffs.abs().sum(dim=-1)        # (B, K)  L1
        else:
            dists = (diffs * diffs).sum(dim=-1)    # (B, K)  L2-squared (no sqrt)
        dists = dists.clamp(min=1e-12)

        accum = (weights / dists * mask.float()).sum(dim=1)   # (B,)
        return torch.sigmoid(accum)

    def forward(self, pred, target, current_epoch = None, x_batch = None, model = None):
        if pred.dim() > 1:
            pred = pred.reshape(-1)
            target = target.reshape(-1)

        pos_mask = target > self.local_loss.positive_threshold
        pos_scores = pred[pos_mask]
        neg_scores = pred[~pos_mask]

        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return pred.new_tensor(0.0, requires_grad = True)

        pos_count = pos_scores.numel()
        neg_count = neg_scores.numel()
        if neg_count % pos_count == 0:
            neg_ratio = neg_count // pos_count
            neg_scores = neg_scores.view(neg_ratio, pos_count)
            if self.local_loss.use_max_negative:
                neg_per_pos = neg_scores.max(dim = 0).values
            else:
                neg_per_pos = neg_scores.mean(dim = 0)
        else:
            neg_value = neg_scores.max() if self.local_loss.use_max_negative else neg_scores.mean()
            neg_per_pos = neg_value.expand_as(pos_scores)

        if self.local_loss.score_is_distance:
            e_pos = pos_scores
            e_neg = neg_per_pos
        else:
            e_pos = -pos_scores
            e_neg = -neg_per_pos

        delta = self.local_loss.margin + e_pos - e_neg
        hinge = F.relu(delta)

        pos_triples = x_batch[pos_mask].detach().cpu().tolist()
        conf_values = [self.local_loss._confidence.get(tuple(triple), 1.0) for triple in pos_triples]
        local_conf = pred.new_tensor(conf_values)
        prior_conf = self._get_prior_confidence(pos_triples, device = pred.device, dtype = pred.dtype)
        adaptive_conf = self._get_adaptive_confidence(pos_triples, model = model, device = pred.device, dtype = pred.dtype)

        combined_conf = (
            self.lambda_1_lt * local_conf +
            self.lambda_2_pp * prior_conf +
            self.lambda_3_ap * adaptive_conf
        )
        loss = (combined_conf * hinge).mean()

        q_values = (-delta).detach()
        for triple, q_val, conf_val in zip(pos_triples, q_values.tolist(), conf_values):
            if q_val <= 0:
                new_conf = max(conf_val * self.local_loss.alpha, self.local_loss.min_conf)
            else:
                new_conf = min(conf_val + self.local_loss.beta, self.local_loss.max_conf)
            self.local_loss._confidence[tuple(triple)] = new_conf

        return loss


class general_robust_loss(nn.Module):

    def __init__(self, alpha_grl, scale_grl, eps_grl = 1e-6):
        super().__init__()

        self.alpha_grl = alpha_grl
        self.scale_grl = scale_grl
        self.eps_grl = eps_grl 
    
    @staticmethod
    def log1p_safe(x):
        """The same as torch.log1p(x), but clamps the input to prevent NaNs."""
        x = torch.as_tensor(x)
        return torch.log1p(torch.min(x, torch.tensor(33e37).to(x)))

    @staticmethod
    def expm1_safe(x):
        """The same as tf.math.expm1(x), but clamps the input to prevent NaNs."""
        x = torch.as_tensor(x)
        return torch.expm1(torch.min(x, torch.tensor(87.5).to(x)))   

    def forward(self, pred, target, current_epoch = None):

        pred = torch.sigmoid(pred) 

        x = pred - target.float()
        alpha_grl = torch.as_tensor(self.alpha_grl, device=x.device, dtype=x.dtype)

        square_scaled_x = (x / self.scale_grl) ** 2 

        loss_two = 0.5 * square_scaled_x 

        loss_zero = self.log1p_safe(0.5 * square_scaled_x) 

        loss_neginf = -torch.expm1(-0.5 * square_scaled_x) 

        loss_posinf = self.expm1_safe(0.5 * square_scaled_x) 

        machine_epsilon = torch.tensor(np.finfo(np.float32).eps).to(x) 

        beta_safe = torch.max(machine_epsilon, torch.abs(alpha_grl - 2.0)) 

        alpha_safe = torch.where(alpha_grl >= 0, torch.ones_like(alpha_grl), -torch.ones_like(alpha_grl)) * torch.max(machine_epsilon, torch.abs(alpha_grl)) 

        loss_otherwise = (beta_safe / alpha_safe)  * (torch.pow(square_scaled_x / beta_safe + 1.0, 0.5 * self.alpha_grl) - 1.0)

        loss = torch.where(alpha_grl == -float('inf'), loss_neginf, 
            torch.where(alpha_grl == 0, loss_zero, 
                torch.where(alpha_grl == 2, loss_two, 
                    torch.where(alpha_grl == float('inf'), loss_posinf, loss_otherwise))))
        
        loss = loss.mean() 

        return loss 

"""
Losses from the paper: Mitigating Label Noise through Data Ambiguation
"""

eps = 1e-7 

class GCELoss(nn.Module):
    def __init__(self, q=0.2):
        super().__init__()
        self.q = q
        self.eps = eps

    def forward(self, pred, target, current_epoch = None):
        p = torch.sigmoid(pred)
        p_t = target * p + (1.0 - target) * (1.0 - p) 
        p_t = torch.clamp(p_t, min = eps, max = 1.0)
        loss = (1.0 - torch.pow(p_t, self.q)) / self.q 
        return loss.mean()

# class NCELoss(nn.Module):
#     def __init__(self, scale = 1.0, eps = 1e-7):
#         super().__init__()
#         self.scale = scale 
#         self.eps = eps

#     def forward(self, pred, target, current_epoch = None):
#         target = target.float()
#         p = torch.sigmoid(pred)
#         p = torch.clamp(p, min=self.eps, max=1.0 - self.eps)

#         p_t = target * p + (1.0 - target) * (1.0 - p)
#         p_t = torch.clamp(p_t, min=self.eps, max=1.0 - self.eps)

#         numerator = -torch.log(p_t) 

#         denom = -torch.log(p) - torch.log(1.0 - p) 
#         denom = torch.clamp(denom, min=self.eps)

#         loss = numerator / denom

#         return self.scale * loss.mean() 

class NCELoss(nn.Module):
    def __init__(self, scale = 1.0):
        super().__init__()
        self.scale = float(scale) 

    def forward(self, pred, target, current_epoch = None):

        target = target.float()
        log_p = F.log_softmax(pred, dim = -1)
        num_pos = target.sum(dim = -1)
        numerator = -(target * log_p).sum(dim = -1) / num_pos
        denom = -log_p.sum(dim = -1) + 1e-7
        loss = (numerator / denom)
        return self.scale * loss.mean()

class NCEandAGCELoss(nn.Module):
    def __init__(self, nce_scale=1.0, agce_a=0.1, agce_q=1.0, agce_eps=1e-8, agce_scale=1.0):
        super().__init__()

        self.nce = NCELoss(scale=nce_scale)
        self.agce = AGCELoss(agce_a=agce_a, agce_q=agce_q, eps=agce_eps, scale=agce_scale)

    def forward(self, pred, target, current_epoch = None):
        return self.nce(pred, target, current_epoch=current_epoch) + self.agce(pred, target, current_epoch=current_epoch)

class NCEandAULoss(nn.Module):
    def __init__(self, nce_scale=1.0, aul_a=2.0, aul_p=1.8, aul_eps=1e-7, aul_scale=1.0):
        super().__init__()

        self.nce = NCELoss(scale=nce_scale)
        self.aul = AULoss(aul_a=aul_a, aul_p=aul_p, eps=aul_eps, scale=aul_scale)

    def forward(self, pred, target, current_epoch = None):
        return self.nce(pred, target, current_epoch=current_epoch) + self.aul(pred, target, current_epoch=current_epoch)

class RDALoss(nn.Module):
    def __init__(
        self, alpha_rda = 0.1, beta_rda = 0.2, adaptive_beta = True, epochs = None, warmup = False, adaptive_start_beta = None,
        adaptive_end_beta = None, adaptive_type = "cosine", eps = 1e-8, warmup_epochs = 0
    ):

        super().__init__()
        self.alpha_rda = max(alpha_rda,1e-6) #relaxation parameter
        self.beta_rda = beta_rda #confidence threshold,60 atleast
        self.adaptive_beta = adaptive_beta
        self.epochs = epochs
        self.warmup = warmup
        self.start_beta = adaptive_start_beta
        self.end_beta = adaptive_end_beta
        self.adaptive_type = adaptive_type
        self.eps = eps
        self.warmup_epochs = warmup_epochs

        if self.adaptive_beta: 
            assert self.epochs is not None 
            assert self.start_beta is not None 
            assert self.end_beta is not None 

    def _get_beta(self, epoch):
        if not self.adaptive_beta:
            return self.beta_rda 
        if epoch is None:
            return None 
        if self.adaptive_type == "linear":
            return (1.0 - epoch / self.epochs) * self.start_beta + (epoch / self.epochs) * self.end_beta
        elif self.adaptive_type == "cosine":
            return self.end_beta + 0.5 * (self.start_beta - self.end_beta) * (1.0 + torch.cos(torch.tensor(torch.pi * epoch / self.epochs))).item() 
        else:
            raise ValueError(f"Unknown adaptive beta type: {self.adaptive_type}")
    
    @staticmethod
    def _kl_div(q, p, eps):
        q = torch.clamp(q, min = eps, max = 1.0 - eps)
        p = torch.clamp(p, min = eps, max = 1.0 - eps)
        return q * (torch.log(q) - torch.log(p)) + (1.0 - q) * (torch.log(1.0 - q) - torch.log(1.0 - p))

    def forward(self, pred, target, current_epoch = None):

        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, min = self.eps, max = 1.0 - self.eps) 

        # if self.adaptive_beta:
        #     beta = self._get_beta(current_epoch)
        #     if beta is None:
        #         suspicious_neg = torch.zeros_like(target, dtype = torch.bool)
        #     else:
        #         suspicious_neg = (target == 0) & (pred > beta) #Flags negatives that look suspiciously high-confidence
        # else:
        #     if self.warmup:
        #         suspicious_neg = torch.zeros_like(target, dtype=torch.bool)
        #     else:
        #         suspicious_neg = (target == 0) & (pred > self.beta_rda)

        # inside_pos = (target == 1) & (pred >= 1.0 - self.alpha_rda)  #clean positives
        # inside_neg = (target == 0) & (~suspicious_neg) & (pred <= self.alpha_rda) #clean and confidently negative entries
        # inside_amb = suspicious_neg #ambigious ones

        # # Loss is zero if sample is:
        # # a confident enough positive
        # # a confident enough negative
        # # or an ambiguous negative
        # inside = inside_pos | inside_neg | inside_amb 

        # q_r = torch.where(
        #     target == 1,
        #     torch.full_like(pred, 1.0 - self.alpha_rda),   #replaces hard labels {1, 0} with softened targets {1-alpha, alpha}
        #     torch.full_like(pred, self.alpha_rda)
        # )

        # q_r = torch.clamp(q_r, min=self.eps, max=1.0 - self.eps)

        # #KL(Bernoulli(q_r) || Bernoulli(pred))
        # kl = (q_r * (torch.log(q_r) - torch.log(pred)) + (1.0 - q_r) * (torch.log(1.0 - q_r) - torch.log(1.0 - pred)))

        # loss = torch.where(inside, torch.zeros_like(kl), kl)
        # return loss.mean()

        beta = self._get_beta(current_epoch)

        warmup_active = self.warmup and (current_epoch is not None) and (current_epoch < self.warmup_epochs)

        if warmup_active:
            suspicious_neg = torch.zeros_like(target, dtype = torch.bool)
        else:
            suspicious_neg = (target == 0) & (pred > beta)
        
        pos_mask = (target == 1)
        neg_mask = (target == 0)
        clean_neg_mask = neg_mask & (~suspicious_neg)

        q_r = torch.empty_like(pred)

        # positives: must be at least 1-alpha
        q_r[pos_mask] = torch.clamp(pred[pos_mask], min=1.0 - self.alpha_rda, max=1.0)

        # clean negatives: should stay near 0, up to alpha
        q_r[clean_neg_mask] = torch.clamp(pred[clean_neg_mask], min=0.0, max=self.alpha_rda)

        # suspicious negatives: relaxed negatives, allowed up to beta
        q_r[suspicious_neg] = torch.clamp(pred[suspicious_neg], min=0.0, max=beta)

        # KL(Bern(q_r) || Bern(prob))
        kl = self._kl_div(q_r, pred, self.eps)

        return kl.mean()

#End of losses from Mitigating Label Noise through Data Ambiguation

#RDA + Waveloss, RDA + RoBoss  

class RDARoBossLoss(nn.Module):

    def __init__(
            self, a = 1.5, lambda_r = 1.0, beta_start = 0.75, beta_end = 0.60, total_epochs = None, beta_fixed = None, eps = 1e-8 
    ):
        super().__init__()
        self.a = a
        self.lambda_r = lambda_r
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_epochs = total_epochs
        self.beta_fixed = beta_fixed
        self.eps = eps
    
    def _get_beta(self, current_epoch):
        if self.beta_fixed is not None:
            return self.beta_fixed
        if current_epoch is None or self.total_epochs is None or self.total_epochs <= 0:
            return self.beta_start
        t = min(max(int(current_epoch), 0), int(self.total_epochs))
        cos = math.cos(math.pi * t / float(self.total_epochs))
        return self.beta_end + 0.5 * (self.beta_start - self.beta_end) * (1 + cos)
    
    def forward(self, pred, target, current_epoch = None):
        pred_dtype = pred.dtype
        target = target.float()
        prob = torch.sigmoid(pred.float())
        prob = torch.clamp(prob, min = self.eps, max = 1.0 - self.eps)

        p_true = target * prob + (1.0 - target) * (1.0 - prob)

        u = 1.0 - p_true #model's probability on the correct label — the probabilistic analogue of target·pred

        beta = max(self._get_beta(current_epoch), self.eps)

# TARGET SEPARATION LOGIC
        # If target == 0, we gate based on how high the raw probability is.
        # If target == 1, we typically retain full structural loss (s=1.0) to enforce learning true facts.

        s_neg = torch.clamp(1.0 - (prob.detach() / beta), min = 0.0)
        s = torch.where(target == 0.0, s_neg, torch.ones_like(prob)) #if tar == 0 use gating, else tar == 1, s = 1.0

        lambda_dyn = self.lambda_r * s 
        exp = -self.a * u 
        loss = lambda_dyn * (1.0 - (self.a * u + 1.0) * torch.exp(exp))

# Explicitly zero out fully ambiguated targets to prevent floating point residual gradients
        loss = torch.where(s > 0, loss, torch.zeros_like(loss))

        return loss.mean().to(pred_dtype)

class RDAWaveLoss(nn.Module):
    
    def __init__(
            self, a = 1.5, lambda_w = 1.0, beta_start = 0.75, beta_end = 0.60, total_epochs = None, beta_fixed = None, eps = 1e-8 
    ):
        super().__init__()
        self.a = a
        self.lambda_w = lambda_w
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_epochs = total_epochs
        self.beta_fixed = beta_fixed
        self.eps = eps
    
    def _get_beta(self, current_epoch):
        if self.beta_fixed is not None:
            return self.beta_fixed
        if current_epoch is None or self.total_epochs is None or self.total_epochs <= 0:
            return self.beta_start
        t = min(max(int(current_epoch), 0), int(self.total_epochs))
        cos = math.cos(math.pi * t / float(self.total_epochs))
        return self.beta_end + 0.5 * (self.beta_start - self.beta_end) * (1 + cos)
    
    def forward(self, pred, target, current_epoch = None):
        pred_dtype = pred.dtype
        target = target.float()
        prob = torch.sigmoid(pred.float())
        prob = torch.clamp(prob, min = self.eps, max = 1.0 - self.eps)

        p_true = target * prob + (1.0 - target) * (1.0 - prob)

        u = 1.0 - p_true

        beta = max(self._get_beta(current_epoch), self.eps)

        s_neg = torch.clamp(1.0 - (prob.detach() / beta), min = 0.0)
        s = torch.where(target == 0.0, s_neg, torch.ones_like(prob)) #if tar == 0 use gating, else tar == 1, s = 1.0

        lambda_dyn = self.lambda_w / (s + self.eps) 
        exp = -self.a * u 
        denom = 1.0 + lambda_dyn * (u ** 2) * torch.exp(exp)
        loss = (1.0 / lambda_dyn) * (1.0 - 1.0 / denom)

        loss = torch.where(s > 0, loss, torch.zeros_like(loss))

        return loss.mean().to(pred_dtype)

class CORESLoss(nn.Module):
    def __init__(self, beta_max = 2.0, warmup_epochs = 30, eps = 1e-8):
        super().__init__()
        self.beta_max = beta_max
        self.warmup_epochs = warmup_epochs
        self.eps = eps 

    def _get_beta(self, epoch):
        if epoch is None:
            return float(self.beta_max)

        epoch = int(epoch)
        if epoch < 10:
            return 0.0 
        elif epoch < 40:
            return float(self.beta_max) * ((epoch - 10) / 29.0)
        return float(self.beta_max)

    def forward(self, pred, target, current_epoch = None):
        beta = self._get_beta(current_epoch)
        ce_loss = F.cross_entropy(pred, target, reduction="none") #one loss value per sample so we can later decide which samples to keep/drop
        neg_log_probs = -F.log_softmax(pred, dim=1)
        regularizer = neg_log_probs.mean(dim=1)

        loss_per_sample = ce_loss - beta * regularizer
        selection_score = ce_loss - regularizer 

        if current_epoch is None or current_epoch <= self.warmup_epochs: #For initial num of epochs, mask all val to 1
            mask = torch.ones_like(loss_per_sample)
        else:
            mask = (selection_score <= 0).float().detach() #mask 1 only for sel <= 0 

        loss = (mask * loss_per_sample).sum() / mask.sum().clamp_min(1.0)
        return loss 

class DSKRLLoss(nn.Module):
    """
    DSKRL (Shao et al., 2021) loss:
      Loss(h,r,t) = (L(h,r,t) + (1 / Z) * sum_p R(p|h,t) * L(p,r)) * S(h,r,t)

    Paper-faithful defaults:
      L(h,r,t) = max(0, margin + PT(pos) - PT(neg))                    (Eq. 13/14)
      PT(h,r,t) = EHT(h,r,t) + RP(h,P,t)                               (Eq. 6)
      EHT(h,r,t) = || T_h + r - T_t ||_2                               (Eq. 4)
        with T_e = Σ_i α_i (M_type[t_i] · M_domain[d_i]) · e_emb       (Eq. 2 + Eq. 3)
        where (t_i, d_i, α_i) are entity e's own types (loaded from
        entityTypes.txt). Falls back to a relation-keyed encoder when
        entity-typed data is not available.
      S(h,r,t)  = k1 * LS(h,r,t) + k2 * DPS(h,r,t)                     (Eq. 12)
      L(p,r)    = max(0, path_margin + ||p - r|| - ||p - r'||)         (Eq. 15)
      LS update: LS <- gamma * LS  iff Q(h,r,t) <= 0                   (Eq. 8-9)
      DPS = sigmoid( sum_p R(p|h,t) / ||r - p||_2 )                    (Eq. 11)

    `use_native_score=True` opts back into using the model's native score
    inside L(h,r,t) (faster, lets training/eval geometries match) -- not
    paper-faithful.
    """

    requires_x_batch = True
    requires_model = True
    _VALID_ABLATION_MODES = {"full", "ls", "pt", "eht"}
    _VALID_TYPE_COMBINE_MODES = {"chain", "weighted_sum"}

    def __init__(
        self,
        margin=1.0,
        local_decay_gamma=0.9,
        path_margin=1.0,
        positive_threshold=0.5,
        use_max_negative=True,
        score_is_distance=False,
        support_k1=0.6,
        support_k2=0.4,
        dps_use_l1=False,
        num_entities=None,
        num_relations=None,
        embedding_dim=None,
        eps=1e-12,
        ablation_mode="full",
        use_native_score=False,
        use_native_score_for_ls=False,
        max_paths_per_triple=30,
        type_combine_mode="weighted_sum",
    ):
        super().__init__()
        self.margin = float(margin)
        self.local_decay_gamma = float(local_decay_gamma)
        self.path_margin = float(path_margin)
        self.positive_threshold = float(positive_threshold)
        self.use_max_negative = bool(use_max_negative)
        self.score_is_distance = bool(score_is_distance)
        self.support_k1 = float(support_k1)
        self.support_k2 = float(support_k2)
        self.dps_use_l1 = bool(dps_use_l1)
        self.eps = float(eps)
        self.num_relations = int(num_relations) if num_relations is not None else None
        self.embedding_dim = int(embedding_dim) if embedding_dim is not None else None
        self.ablation_mode = str(ablation_mode).lower()
        self.use_native_score = bool(use_native_score)
        self.use_native_score_for_ls = (
            self.use_native_score if use_native_score_for_ls is None else bool(use_native_score_for_ls)
        )
        self.type_combine_mode = str(type_combine_mode).lower()
        if self.type_combine_mode not in self._VALID_TYPE_COMBINE_MODES:
            raise RuntimeError(
                f"Unsupported DSKRL type_combine_mode={self.type_combine_mode!r}. "
                f"Choose one of {sorted(self._VALID_TYPE_COMBINE_MODES)}."
            )
        # max_paths_per_triple: keep only the top-K most reliable paths per triple
        # to bound per-forward-pass cost; None / 0 = keep all paths.
        self.max_paths_per_triple = int(max_paths_per_triple) if max_paths_per_triple else None
        self._local_support: Dict[tuple, float] = {}
        self._path_data: Dict[tuple, List] = {}
        # Pre-built (N, K, L) buffers for zero-Python-loop forward pass.
        # Populated by _build_path_buffers() when set_path_data() is called.
        self._path_rel_ids_buf      = None   # (N, K, L) long
        self._path_signs_buf        = None   # (N, K, L) float32
        self._path_weights_buf      = None   # (N, K)    float32 -- normalised R/Z (RP, L(p,r))
        self._path_weights_raw_buf  = None   # (N, K)    float32 -- raw R(p|h,t) (DPS)
        self._path_mask_buf         = None   # (N, K)    bool
        self._path_triple_to_idx: dict = {}
        self._aux_ready = False
        self._num_types = 0
        self._num_domains = 0
        # Per-entity type buffers (_entity_*_buf) are registered below so they
        # are moved by model.to(device). They are zero-sized until
        # set_aux_data populates them with per-entity types (Eq. 2).

        if self.num_relations is None or self.embedding_dim is None:
            raise RuntimeError("DSKRLLoss requires num_relations and embedding_dim.")
        if self.ablation_mode not in self._VALID_ABLATION_MODES:
            raise RuntimeError(
                f"Unsupported DSKRL ablation_mode={self.ablation_mode!r}. "
                f"Choose one of {sorted(self._VALID_ABLATION_MODES)}."
            )

        self.domain_mats = None
        self.type_mats = None
        self.register_buffer("_head_type_ids", torch.full((self.num_relations,), -1, dtype=torch.long), persistent=False)
        self.register_buffer("_tail_type_ids", torch.full((self.num_relations,), -1, dtype=torch.long), persistent=False)
        self.register_buffer("_head_domain_ids", torch.full((self.num_relations,), -1, dtype=torch.long), persistent=False)
        self.register_buffer("_tail_domain_ids", torch.full((self.num_relations,), -1, dtype=torch.long), persistent=False)
        # Per-entity type buffers — registered so model.to(device) migrates them.
        # Empty (E=0, K=0) until set_aux_data is called with entity_types.
        self.register_buffer("_entity_type_ids_buf",     torch.zeros(0, 0, dtype=torch.long),    persistent=False)
        self.register_buffer("_entity_domain_ids_buf",   torch.zeros(0, 0, dtype=torch.long),    persistent=False)
        self.register_buffer("_entity_type_weights_buf", torch.zeros(0, 0, dtype=torch.float32), persistent=False)
        self.register_buffer("_entity_type_mask_buf",    torch.zeros(0, 0, dtype=torch.bool),    persistent=False)

    def set_path_data(self, path_data):
        """Store path data and pre-build fixed-size tensors for a zero-Python-loop forward pass.

        The tensors built here allow the three path-energy functions to replace
        their outer Python loop (one iteration per batch triple) with a single
        tensor gather + einsum — O(1) Python per forward call.
        """
        if not path_data:
            self._path_data = {}
            self._path_rel_ids_buf = None   # signals "no path data" to callers
            return
        cap = self.max_paths_per_triple
        if cap and cap > 0:
            self._path_data = {
                k: sorted(v, key=lambda x: -float(x[1]))[:cap]
                for k, v in path_data.items()
            }
        else:
            self._path_data = path_data
        self._build_path_buffers()

    def _build_path_buffers(self):
        """Pre-build (N, K, L) numpy arrays from ``self._path_data``.

        Symbols
        -------
        N : number of triples that have at least one path
        K : max paths per triple  (== self.max_paths_per_triple when capped)
        L : max path length in hops (usually 1 or 2 for PCRA)
        """
        import numpy as np
        triples_list = list(self._path_data.keys())
        N = len(triples_list)
        K = max(len(v) for v in self._path_data.values())
        L = max(
            (len(p) for v in self._path_data.values() for p, _ in v),
            default=1,
        )
        L = max(L, 1)
        num_rel = self.num_relations

        rel_ids     = np.zeros((N, K, L), dtype=np.int64)   # relation indices (base, always < R)
        signs       = np.zeros((N, K, L), dtype=np.float32) # 0 for padding; set to ±1 per hop below
        weights     = np.zeros((N, K),    dtype=np.float32) # normalised reliability R/Z (RP, L(p,r))
        weights_raw = np.zeros((N, K),    dtype=np.float32) # raw R(p|h,t) (DPS, Eq. 11)
        mask        = np.zeros((N, K),    dtype=np.bool_)   # True for valid paths

        self._path_triple_to_idx: dict = {}
        for i, triple in enumerate(triples_list):
            self._path_triple_to_idx[triple] = i
            path_list = self._path_data[triple]
            z = sum(float(r) for _, r in path_list)
            if z <= 0:
                continue
            for j, (rel_path, reliability) in enumerate(path_list):
                for l, pid in enumerate(rel_path):
                    pid = int(pid)
                    if pid < num_rel:
                        rel_ids[i, j, l] = pid
                        signs[i, j, l]   = 1.0
                    else:
                        rel_ids[i, j, l] = pid - num_rel
                        signs[i, j, l]   = -1.0
                weights[i, j]     = float(reliability) / z
                weights_raw[i, j] = float(reliability)
                mask[i, j]        = True

        # Store as CPU tensors; moved to the correct device on first use.
        self._path_rel_ids_buf     = torch.from_numpy(rel_ids)      # (N, K, L) long
        self._path_signs_buf       = torch.from_numpy(signs)        # (N, K, L) float32
        self._path_weights_buf     = torch.from_numpy(weights)      # (N, K)    float32 -- R/Z
        self._path_weights_raw_buf = torch.from_numpy(weights_raw)  # (N, K)    float32 -- raw R
        self._path_mask_buf        = torch.from_numpy(mask)         # (N, K)    bool

    def _gather_path_tensors(self, batch_triples_list, rel_emb, device, dtype):
        """Look up pre-built buffers for a batch and compute path vectors.

        The only Python iteration is O(B) dict lookups to build the index
        array; everything else is pure tensor ops (gather + einsum).

        Returns
        -------
        path_vecs   : (B, K, D)  -- path embedding vectors  (sum of relation embeddings, sign-aware)
        weights     : (B, K)     -- normalised reliability  R(p|h,t) / Z   (RP, L(p,r))
        weights_raw : (B, K)     -- raw reliability         R(p|h,t)       (DPS, Eq. 11)
        mask        : (B, K)     -- True for valid (non-padding) paths
        """
        # O(B) dict lookups — cheap
        raw_idx = [self._path_triple_to_idx.get(tuple(t), -1)
                   for t in batch_triples_list]
        has_paths  = torch.tensor([i >= 0 for i in raw_idx], device=device)   # (B,)
        safe_idx   = torch.tensor([i if i >= 0 else 0 for i in raw_idx],
                                  dtype=torch.long)                            # (B,)

        # Gather from CPU buffers → device in one op each
        rel_ids_b     = self._path_rel_ids_buf[safe_idx].to(device)            # (B, K, L)
        signs_b       = self._path_signs_buf[safe_idx].to(device, dtype)       # (B, K, L)
        weights_b     = self._path_weights_buf[safe_idx].to(device, dtype)     # (B, K)
        weights_raw_b = self._path_weights_raw_buf[safe_idx].to(device, dtype) # (B, K)
        mask_b        = self._path_mask_buf[safe_idx].to(device)               # (B, K)

        # Zero out rows for triples with no path entry
        if not has_paths.all():
            no_path       = ~has_paths          # (B,)
            weights_b     = weights_b.clone()
            weights_raw_b = weights_raw_b.clone()
            mask_b        = mask_b.clone()
            weights_b[no_path]     = 0.0
            weights_raw_b[no_path] = 0.0
            mask_b[no_path]        = False

        # Compute path vectors via a single gather + einsum:
        #   rel_emb[rel_ids_b] : (B, K, L, D)
        #   signs_b            : (B, K, L)  → unsqueeze → (B, K, L, 1)
        #   product.sum(L)     : (B, K, D)
        emb_seq   = rel_emb[rel_ids_b]                           # (B, K, L, D)
        path_vecs = (signs_b.unsqueeze(-1) * emb_seq).sum(dim=2) # (B, K, D)

        return path_vecs, weights_b, weights_raw_b, mask_b

    def set_aux_data(self, aux_data):
        # Store type/domain metadata used to project entities before
        # computing EHT(h,r,t) and LS(h,r,t).
        #
        # Two formats are accepted:
        #
        # (a) relation-keyed (TKRL-style, fallback): one (head_type, head_domain,
        #     tail_type, tail_domain) tuple per relation. Eq. 2 collapses to n=1.
        #
        # (b) entity-keyed (paper-faithful, preferred when available): a list of
        #     (type_id, domain_id, weight) tuples per entity. The encoder
        #     computes T_e = Σ_i α_i (M_type · M_domain) (Eq. 2 + Eq. 3) and
        #     uses the same T_e regardless of head/tail role.
        if not aux_data:
            self._num_types = 0
            self._num_domains = 0
            self.domain_mats = None
            self.type_mats = None
            self._aux_ready = False
            self._clear_entity_type_buffers()
            return

        self._num_types = int(aux_data.get("num_types", 0))
        self._num_domains = int(aux_data.get("num_domains", 0))
        self._head_type_ids = torch.as_tensor(aux_data.get("head_type_ids", [-1] * self.num_relations), dtype=torch.long)
        self._tail_type_ids = torch.as_tensor(aux_data.get("tail_type_ids", [-1] * self.num_relations), dtype=torch.long)
        self._head_domain_ids = torch.as_tensor(aux_data.get("head_domain_ids", [-1] * self.num_relations), dtype=torch.long)
        self._tail_domain_ids = torch.as_tensor(aux_data.get("tail_domain_ids", [-1] * self.num_relations), dtype=torch.long)

        # Build per-entity type buffers (format b) when provided.
        entity_types = aux_data.get("entity_types")
        if entity_types:
            self._build_entity_type_buffers(entity_types)
        else:
            self._clear_entity_type_buffers()

        self._init_aux_parameters()

    def _clear_entity_type_buffers(self):
        # Replace the registered buffers with empty (0, 0) tensors so
        # ``has_entity_types()`` (via .numel() == 0) flips back to False
        # while keeping them registered for module.to(device).
        device = self._entity_type_ids_buf.device
        self._entity_type_ids_buf     = torch.zeros(0, 0, dtype=torch.long,    device=device)
        self._entity_domain_ids_buf   = torch.zeros(0, 0, dtype=torch.long,    device=device)
        self._entity_type_weights_buf = torch.zeros(0, 0, dtype=torch.float32, device=device)
        self._entity_type_mask_buf    = torch.zeros(0, 0, dtype=torch.bool,    device=device)

    def _build_entity_type_buffers(self, entity_types):
        """Build padded (E, K) tensors of per-entity type chains for Eq. 2.

        ``entity_types`` is a list indexed by entity id; each element is a list
        of ``(type_id, domain_id, weight)`` triples (any of which may be -1
        when missing).
        """
        import numpy as np
        E = len(entity_types)
        K = max((len(ts) for ts in entity_types), default=0)
        if E == 0 or K == 0:
            self._clear_entity_type_buffers()
            return

        type_ids   = np.full((E, K), -1, dtype=np.int64)
        domain_ids = np.full((E, K), -1, dtype=np.int64)
        weights    = np.zeros((E, K),     dtype=np.float32)
        mask       = np.zeros((E, K),     dtype=np.bool_)

        for e, triples in enumerate(entity_types):
            total = sum(float(w) for _, _, w in triples) or 0.0
            if total <= 0:
                continue
            for k, (t_id, d_id, w) in enumerate(triples):
                type_ids[e, k]   = int(t_id)
                domain_ids[e, k] = int(d_id)
                weights[e, k]    = float(w) / total
                mask[e, k]       = True

        device = self._entity_type_ids_buf.device
        self._entity_type_ids_buf     = torch.from_numpy(type_ids).to(device)
        self._entity_domain_ids_buf   = torch.from_numpy(domain_ids).to(device)
        self._entity_type_weights_buf = torch.from_numpy(weights).to(device)
        self._entity_type_mask_buf    = torch.from_numpy(mask).to(device)

    @staticmethod
    def _resolve_target(pred, target):
        # Helper only: recover the binary labels y used to split positive and
        # negative triples in a NegSample batch.
        if torch.is_tensor(target):
            return target
        if isinstance(target, (list, tuple)):
            tensors = [item for item in target if torch.is_tensor(item)]
            if tensors:
                if pred is not None:
                    for item in tensors:
                        if item.numel() == pred.numel():
                            return item
                return tensors[-1]
        raise RuntimeError("DSKRLLoss expects target to be a torch.Tensor.")

    def _scores_to_energy(self, scores):
        # DSKRL is written in energy form, where smaller is better.
        # This helper converts model scores into energies when needed.
        if self.score_is_distance:
            return scores
        return -scores

    def _pair_negative_scores(self, pos_scores, neg_scores):
        # Higher native scores are assumed to mean more plausible triples unless
        # score_is_distance=True. The hardest negative therefore has the largest
        # score for score models, and the smallest value for distance models.
        pos_count = pos_scores.numel()
        neg_count = neg_scores.numel()
        if neg_count % pos_count == 0:
            neg_ratio = neg_count // pos_count
            neg_scores = neg_scores.view(neg_ratio, pos_count)
            if self.use_max_negative:
                if self.score_is_distance:
                    return neg_scores.min(dim=0).values
                return neg_scores.max(dim=0).values
            return neg_scores.mean(dim=0)
        if self.use_max_negative:
            neg_value = neg_scores.min() if self.score_is_distance else neg_scores.max()
        else:
            neg_value = neg_scores.mean()
        return neg_value.expand_as(pos_scores)

    def _pair_negative_energies(self, pos_energies, neg_energies):
        # DSKRL energy terms are lower-is-better, so the hardest negative is the
        # one with the smallest energy.
        pos_count = pos_energies.numel()
        neg_count = neg_energies.numel()
        if neg_count % pos_count == 0:
            neg_ratio = neg_count // pos_count
            neg_energies = neg_energies.view(neg_ratio, pos_count)
            if self.use_max_negative:
                return neg_energies.min(dim=0).values
            return neg_energies.mean(dim=0)
        neg_value = neg_energies.min() if self.use_max_negative else neg_energies.mean()
        return neg_value.expand_as(pos_energies)

    def _relation_path_vector(self, rel_path, rel_emb, num_relations):
        # Kept for backward-compatibility; not called in the normal forward path.
        pids  = rel_emb.new_tensor(rel_path, dtype=torch.long)
        fwd   = pids < num_relations
        base  = torch.where(fwd, pids, pids - num_relations)
        signs = torch.where(fwd, rel_emb.new_ones(len(pids)), -rel_emb.new_ones(len(pids)))
        return (signs.unsqueeze(1) * rel_emb[base]).sum(0)

    def _distance(self, x):
        # Distance used by all DSKRL energy terms:
        # d(x) = ||x||_2 by default, or ||x||_1 when configured.
        if self.dps_use_l1:
            return x.abs().sum(dim=-1)
        return torch.sqrt(torch.clamp((x * x).sum(dim=-1), min=self.eps))

    def _init_aux_parameters(self):
        # Trainable type / domain projection matrices used to build T_e (Eq. 2-3).
        if self._aux_ready:
            return
        if self._num_domains > 0:
            domain_eye = torch.eye(self.embedding_dim).unsqueeze(0).repeat(self._num_domains, 1, 1)
            self.domain_mats = nn.Parameter(domain_eye)
        if self._num_types > 0:
            type_eye = torch.eye(self.embedding_dim).unsqueeze(0).repeat(self._num_types, 1, 1)
            self.type_mats = nn.Parameter(type_eye)
        # Learnable α weights for the two-component weighted-sum encoder (Eq. 2).
        # The softmax over this 2-vector gives (α_type, α_domain) summing to 1.
        # Initialised to (1, 1) -> equal weight after softmax.
        self._type_mix_logits = nn.Parameter(torch.zeros(2))
        self._aux_ready = True

    def _project_entities(self, entity_emb, domain_ids, type_ids, entity_ids=None):
        # Build T_e per entity and project: e_proj = T_e @ e_emb.
        #
        # When per-entity type data is loaded (Eq. 2 + Eq. 3, paper-faithful):
        #     T_e = Σ_i α_i (M_type[t_i] @ M_domain[d_i])
        # uses the entity's own list of (type, domain, weight) triples and
        # ignores ``domain_ids`` / ``type_ids`` (which are relation-keyed).
        #
        # Otherwise we fall back to the relation-keyed encoder:
        #   type_combine_mode="weighted_sum"  (Eq. 2, n=2, m=1):
        #     T_e = α_type * M_type[t_id] + α_domain * M_domain[d_id]
        #   type_combine_mode="chain"         (Eq. 3, n=1, m=2):
        #     T_e = M_type[t_id] @ M_domain[d_id]
        if self.domain_mats is None and self.type_mats is None:
            return entity_emb
        if entity_ids is not None and self._entity_type_ids_buf.numel() > 0:
            return self._project_entities_by_entity(entity_emb, entity_ids)
        if self.type_combine_mode == "weighted_sum":
            return self._project_weighted_sum(entity_emb, domain_ids, type_ids)
        return self._project_chain(entity_emb, domain_ids, type_ids)

    def _project_entities_by_entity(self, entity_emb, entity_ids):
        # Paper-faithful encoder (Eq. 2 + Eq. 3):
        #   for each entity e (with K' valid (type_i, domain_i, α_i) triples):
        #     v_i   = M_type[t_i] · M_domain[d_i] · e_emb        (Eq. 3, m=2)
        #     e_proj = Σ_i α_i · v_i                              (Eq. 2)
        # The (type, domain) chain follows TKRL's TKRL/WHE projection: apply
        # domain first (coarse) then type (fine), per the paper's "first
        # mapped to the more general sub-type space ... then sequentially
        # mapped to the more precise sub-type space".
        device = entity_emb.device
        dtype  = entity_emb.dtype
        B, D = entity_emb.shape

        # Buffers are registered, so they already live on `device`.
        type_ids   = self._entity_type_ids_buf[entity_ids]                  # (B, K)
        domain_ids = self._entity_domain_ids_buf[entity_ids]                # (B, K)
        weights    = self._entity_type_weights_buf[entity_ids].to(dtype)    # (B, K)
        mask       = self._entity_type_mask_buf[entity_ids]                 # (B, K)
        K = type_ids.shape[1]

        # Flatten over (B*K) for batched matmul.
        type_flat   = type_ids.reshape(-1).clamp(min=0)                       # (B*K,)
        domain_flat = domain_ids.reshape(-1).clamp(min=0)                     # (B*K,)
        has_type    = (type_ids   >= 0).to(dtype)                             # (B, K)
        has_domain  = (domain_ids >= 0).to(dtype)                             # (B, K)

        I = torch.eye(D, device=device, dtype=dtype)                          # (D, D)

        # Per-slot matrices (B*K, D, D). When type / domain is -1 use identity
        # so that branch becomes a no-op in the chain.
        if self.type_mats is not None:
            type_mats_flat = self.type_mats[type_flat]                        # (B*K, D, D)
        else:
            type_mats_flat = I.expand(B * K, D, D)
        if self.domain_mats is not None:
            domain_mats_flat = self.domain_mats[domain_flat]                  # (B*K, D, D)
        else:
            domain_mats_flat = I.expand(B * K, D, D)

        has_type_flat   = has_type.reshape(-1, 1, 1)                          # (B*K, 1, 1)
        has_domain_flat = has_domain.reshape(-1, 1, 1)                        # (B*K, 1, 1)
        I_exp = I.expand(B * K, D, D)
        type_mats_flat   = has_type_flat   * type_mats_flat   + (1.0 - has_type_flat)   * I_exp
        domain_mats_flat = has_domain_flat * domain_mats_flat + (1.0 - has_domain_flat) * I_exp

        # Chain: project domain first, then type (Eq. 3 with m=2).
        # .contiguous() ensures the expand-view is safe to reshape on all torch versions.
        e_rep = entity_emb.unsqueeze(1).expand(B, K, D).contiguous().view(B * K, D, 1)  # (B*K, D, 1)
        domain_proj = torch.bmm(domain_mats_flat, e_rep)                      # (B*K, D, 1)
        type_proj   = torch.bmm(type_mats_flat, domain_proj)                  # (B*K, D, 1)
        v = type_proj.squeeze(-1).reshape(B, K, D)                            # (B, K, D)

        # Weighted sum over the entity's types (Eq. 2). `weights` already sums
        # to 1 over valid slots per entity.
        masked_weights = weights * mask.to(dtype)                             # (B, K)
        e_proj = (masked_weights.unsqueeze(-1) * v).sum(dim=1)                # (B, D)

        # Entities with no type info fall back to the raw embedding.
        any_valid = mask.any(dim=1).to(dtype).unsqueeze(-1)                   # (B, 1)
        return any_valid * e_proj + (1.0 - any_valid) * entity_emb

    def _project_chain(self, entity_emb, domain_ids, type_ids):
        # Eq. 3 form: e -> M_domain · e -> M_type · (M_domain · e)
        projected = entity_emb
        if self.domain_mats is not None:
            valid_domain = domain_ids >= 0
            if valid_domain.any():
                projected = projected.clone()
                projected[valid_domain] = torch.bmm(
                    self.domain_mats[domain_ids[valid_domain]],
                    projected[valid_domain].unsqueeze(-1),
                ).squeeze(-1)
        if self.type_mats is not None:
            valid_type = type_ids >= 0
            if valid_type.any():
                projected = projected.clone()
                projected[valid_type] = torch.bmm(
                    self.type_mats[type_ids[valid_type]],
                    projected[valid_type].unsqueeze(-1),
                ).squeeze(-1)
        return projected

    def _project_weighted_sum(self, entity_emb, domain_ids, type_ids):
        # Eq. 2 form with n=2, m=1:
        #   T_e = α_type * M_type[t_id] + α_domain * M_domain[d_id]
        #   e_proj = T_e @ e_emb
        # = α_type * (M_type · e) + α_domain * (M_domain · e)
        weights = F.softmax(self._type_mix_logits, dim=0)        # (2,) -> (α_type, α_domain)
        alpha_type   = weights[0]
        alpha_domain = weights[1]

        type_branch   = torch.zeros_like(entity_emb)
        domain_branch = torch.zeros_like(entity_emb)
        has_type_mask   = torch.zeros(entity_emb.shape[0], device=entity_emb.device, dtype=entity_emb.dtype)
        has_domain_mask = torch.zeros(entity_emb.shape[0], device=entity_emb.device, dtype=entity_emb.dtype)

        if self.type_mats is not None:
            valid_type = type_ids >= 0
            if valid_type.any():
                type_branch[valid_type] = torch.bmm(
                    self.type_mats[type_ids[valid_type]],
                    entity_emb[valid_type].unsqueeze(-1),
                ).squeeze(-1)
                has_type_mask[valid_type] = 1.0
        if self.domain_mats is not None:
            valid_domain = domain_ids >= 0
            if valid_domain.any():
                domain_branch[valid_domain] = torch.bmm(
                    self.domain_mats[domain_ids[valid_domain]],
                    entity_emb[valid_domain].unsqueeze(-1),
                ).squeeze(-1)
                has_domain_mask[valid_domain] = 1.0

        # If only one branch is available for an entity, renormalise so that
        # branch carries the full weight; if neither is available, fall back
        # to the raw embedding.
        denom = (alpha_type * has_type_mask + alpha_domain * has_domain_mask).clamp(min=self.eps)
        weighted = (alpha_type * type_branch + alpha_domain * domain_branch) / denom.unsqueeze(-1)
        any_valid = (has_type_mask + has_domain_mask) > 0
        return torch.where(any_valid.unsqueeze(-1), weighted, entity_emb)

    def _get_entity_hierarchical_type_energy(self, triples, model):
        # EHT(T_h, r, T_t) = || T_h + r - T_t ||_2                        (Eq. 4)
        # where T_h = T_{e_h} @ h_emb, T_t = T_{e_t} @ t_emb are the
        # projected head/tail entity embeddings (T_e built per Eq. 2-3).
        # We use the norm-symmetric form ||t - h - r|| = ||h + r - t||.
        triples = triples.long()
        head_ids = triples[:, 0]
        rel_ids = triples[:, 1]
        tail_ids = triples[:, 2]

        if not hasattr(model, "entity_embeddings") or model.entity_embeddings is None:
            raise RuntimeError("DSKRLLoss requires model.entity_embeddings for EHT computation.")
        if not hasattr(model, "relation_embeddings") or model.relation_embeddings is None:
            raise RuntimeError("DSKRLLoss requires model.relation_embeddings for EHT computation.")

        head_emb = model.entity_embeddings(head_ids)
        tail_emb = model.entity_embeddings(tail_ids)
        rel_emb = model.relation_embeddings(rel_ids)

        head_domains = self._head_domain_ids.to(triples.device)[rel_ids]
        tail_domains = self._tail_domain_ids.to(triples.device)[rel_ids]
        head_types = self._head_type_ids.to(triples.device)[rel_ids]
        tail_types = self._tail_type_ids.to(triples.device)[rel_ids]

        head_final = self._project_entities(head_emb, head_domains, head_types, entity_ids=head_ids)
        tail_final = self._project_entities(tail_emb, tail_domains, tail_types, entity_ids=tail_ids)
        return self._distance(tail_final - head_final - rel_emb)

    def _get_relation_path_energy(self, triples, model): #RP(h, P, t)
        # RP(h,P,t) = (1/Z) * Σ_p R(p|h,t) * d(h + p_vec - t)   (Eq. 5)
        if self._path_rel_ids_buf is None:
            return torch.zeros(triples.shape[0], device=triples.device,
                               dtype=model.relation_embeddings.weight.dtype)
        head_emb  = model.entity_embeddings(triples[:, 0])   # (B, D)
        tail_emb  = model.entity_embeddings(triples[:, 2])   # (B, D)
        rel_emb   = model.relation_embeddings.weight          # (R, D)
        dtype     = rel_emb.dtype

        path_vecs, weights, _weights_raw, mask = self._gather_path_tensors(
            triples.detach().cpu().tolist(), rel_emb, triples.device, dtype
        )  # (B,K,D), (B,K), (B,K), (B,K)

        # (B,1,D) + (B,K,D) - (B,1,D) = (B,K,D)  →  distances (B,K)
        diffs     = head_emb.unsqueeze(1) + path_vecs - tail_emb.unsqueeze(1)
        distances = self._distance(diffs)                              # (B, K)
        # weights here are already R/Z, so summing yields (1/Z) Σ R · d.
        return (weights * distances * mask.float()).sum(dim=1)         # (B,)

    def _get_triple_dissimilarity(self, triples, model):
        # Paper notation:
        #   PT(h,r,t) = EHT(h,r,t) + RP(h,P,t)
        eht = self._get_entity_hierarchical_type_energy(triples, model)
        rp = self._get_relation_path_energy(triples, model)
        return eht + rp

    def _get_selected_dissimilarity(self, triples, model):
        # Ablation selector:
        #   DSKRL(EHT) -> EHT
        #   DSKRL(PT), DSKRL(LS), DSKRL -> PT = EHT + RP
        if self.ablation_mode == "eht":
            return self._get_entity_hierarchical_type_energy(triples, model)
        return self._get_triple_dissimilarity(triples, model)

    def _get_local_quality_energy(self, triples, model):
        # Local quality term used to update LS:
        #   Q(h,r,t) = d(h_PT + r - t_PT)
        triples = triples.long()
        if not hasattr(model, "entity_embeddings") or model.entity_embeddings is None:
            raise RuntimeError("DSKRLLoss requires model.entity_embeddings for local quality computation.")
        if not hasattr(model, "relation_embeddings") or model.relation_embeddings is None:
            raise RuntimeError("DSKRLLoss requires model.relation_embeddings for local quality computation.")

        head_ids = triples[:, 0]
        rel_ids = triples[:, 1]
        tail_ids = triples[:, 2]

        head_emb = model.entity_embeddings(head_ids)
        tail_emb = model.entity_embeddings(tail_ids)
        rel_emb = model.relation_embeddings(rel_ids)

        head_domains = self._head_domain_ids.to(triples.device)[rel_ids]
        tail_domains = self._tail_domain_ids.to(triples.device)[rel_ids]
        head_types = self._head_type_ids.to(triples.device)[rel_ids]
        tail_types = self._tail_type_ids.to(triples.device)[rel_ids]

        head_pt = self._project_entities(head_emb, head_domains, head_types, entity_ids=head_ids)
        tail_pt = self._project_entities(tail_emb, tail_domains, tail_types, entity_ids=tail_ids)
        return self._distance(head_pt + rel_emb - tail_pt)

    def _sample_negative_relation_ids(self, rel_ids, num_relations, device):
        # Helper for L(p,r): sample a corrupted relation r' != r.
        if num_relations <= 1:
            return rel_ids
        sampled = torch.randint(0, num_relations - 1, size=rel_ids.shape, device=device)
        return sampled + (sampled >= rel_ids).long()

    def _get_local_support(self, pos_triples, device, dtype):
        # Paper notation:
        #   LS(h,r,t)
        # is the cached local support value for each positive triple.
        values = [self._local_support.get(tuple(triple), 1.0) for triple in pos_triples]
        return torch.tensor(values, device=device, dtype=dtype)

    def _get_support_term(self, pos_triples, model, device, dtype):
        # Ablation selector:
        #   DSKRL(EHT), DSKRL(PT) -> support = 1
        #   DSKRL(LS)             -> support = LS
        #   DSKRL                 -> support = k1 * LS + k2 * DPS
        if self.ablation_mode in {"eht", "pt"}:
            return torch.ones(len(pos_triples), device=device, dtype=dtype)

        local_support = self._get_local_support(pos_triples, device=device, dtype=dtype)
        if self.ablation_mode == "ls":
            return local_support

        dynamic_path_support = self._get_dynamic_path_support(
            pos_triples,
            model=model,
            device=device,
            dtype=dtype,
        )
        return self.support_k1 * local_support + self.support_k2 * dynamic_path_support

    def _get_dynamic_path_support(self, pos_triples, model, device, dtype):
        # DPS(h,r,t) = sigmoid( Σ_p R(p|h,t) / ||r - p||_2 )         (Eq. 11)
        # Note: per the paper, the sum is over raw R(p|h,t), not R/Z.
        if self._path_rel_ids_buf is None:
            return torch.zeros(len(pos_triples), device=device, dtype=dtype)
        rel_emb = model.relation_embeddings.weight

        path_vecs, _weights_norm, weights_raw, mask = self._gather_path_tensors(
            pos_triples, rel_emb, device, dtype
        )  # (B,K,D), (B,K), (B,K), (B,K)

        rel_ids = torch.tensor([int(t[1]) for t in pos_triples],
                               device=device, dtype=torch.long)     # (B,)
        r_embs  = rel_emb[rel_ids]                                  # (B, D)

        # (B,1,D) - (B,K,D) = (B,K,D)  →  (B,K)
        dists = self._distance(r_embs.unsqueeze(1) - path_vecs).clamp(min=self.eps)
        accum = (weights_raw / dists * mask.float()).sum(dim=1)      # (B,)
        return torch.sigmoid(accum)

    def _get_path_relation_loss(self, pos_triples, model, device, dtype):
        # (1/Z) * Σ_p R(p|h,t) * L(p,r)                              (inner term of Eq. 13)
        # L(p,r) = max(0, path_margin + ||p - r|| - ||p - r'||)      (Eq. 15)
        if self._path_rel_ids_buf is None:
            return torch.zeros(len(pos_triples), device=device, dtype=dtype)
        rel_emb     = model.relation_embeddings.weight
        num_relations = rel_emb.shape[0]
        rel_ids     = torch.tensor([int(t[1]) for t in pos_triples],
                                   device=device, dtype=torch.long)     # (B,)
        neg_rel_ids = self._sample_negative_relation_ids(rel_ids, num_relations, device=device)

        path_vecs, weights, _weights_raw, mask = self._gather_path_tensors(
            pos_triples, rel_emb, device, dtype
        )  # (B,K,D), (B,K), (B,K), (B,K)

        r_embs  = rel_emb[rel_ids]      # (B, D)
        nr_embs = rel_emb[neg_rel_ids]  # (B, D)

        # (B,K,D) - (B,1,D) = (B,K,D)  →  (B,K)
        pos_e = self._distance(path_vecs - r_embs.unsqueeze(1))
        neg_e = self._distance(path_vecs - nr_embs.unsqueeze(1))
        # `weights` is R/Z, so summing already yields (1/Z) Σ R · L(p,r).
        margin_loss = weights * F.relu(self.path_margin + pos_e - neg_e) * mask.float()
        return margin_loss.sum(dim=1)

    def _get_selected_path_relation_loss(self, pos_triples, model, device, dtype):
        # Ablation selector:
        #   Only the full DSKRL objective includes the auxiliary path-relation loss.
        if self.ablation_mode != "full":
            return torch.zeros(len(pos_triples), device=device, dtype=dtype)
        return self._get_path_relation_loss(pos_triples, model, device, dtype)

    def _should_update_local_support(self):
        # LS is only part of the ablations that explicitly keep the local support term.
        return self.ablation_mode in {"ls", "full"}

    def forward(self, pred, target, current_epoch=None, x_batch=None, model=None):
        # Paper objective (Eq. 13) for each positive triple:
        #   Loss(h,r,t) = ( L(h,r,t) + (1/Z) Σ_p R(p|h,t) L(p,r) ) * S(h,r,t)
        # with the paper-faithful defaults:
        #   L(h,r,t) = max(0, margin + PT(pos) - PT(neg))           (Eq. 14)
        #   S(h,r,t) = k1 * LS(h,r,t) + k2 * DPS(h,r,t)             (Eq. 12)
        # use_native_score=True opts back into a native-score margin (faster,
        # makes train/eval geometries match, but not paper-faithful).
        # The second half of this function performs the online LS update (Eq. 8-9).
        target = self._resolve_target(pred, target)

        if pred.dim() > 1:
            pred = pred.reshape(-1)
            target = target.reshape(-1)

        pos_mask = target > self.positive_threshold
        pos_scores = pred[pos_mask]
        neg_scores = pred[~pos_mask]

        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return pred.new_tensor(0.0, requires_grad=True)

        pos_triples_tensor = x_batch[pos_mask]
        neg_triples_tensor = x_batch[~pos_mask]
        pos_triples = pos_triples_tensor.detach().cpu().tolist()

        if self.use_native_score:
            paired_neg_scores = self._pair_negative_scores(pos_scores, neg_scores)
            if self.score_is_distance:
                triple_loss = F.relu(self.margin + pos_scores - paired_neg_scores)
            else:
                triple_loss = F.relu(self.margin - pos_scores + paired_neg_scores)
        else:
            pos_dissimilarity = self._get_selected_dissimilarity(pos_triples_tensor, model)
            neg_dissimilarity = self._pair_negative_energies(
                pos_dissimilarity,
                self._get_selected_dissimilarity(neg_triples_tensor, model),
            )
            triple_loss = F.relu(self.margin + pos_dissimilarity - neg_dissimilarity)

        support = self._get_support_term(
            pos_triples,
            model=model,
            device=pred.device,
            dtype=pred.dtype,
        )
        local_support = self._get_local_support(pos_triples, device=pred.device, dtype=pred.dtype)

        path_relation_loss = self._get_selected_path_relation_loss(
            pos_triples,
            model=model,
            device=pred.device,
            dtype=pred.dtype,
        )
        # Final loss: mean( S(h,r,t) * (L(h,r,t) + L(p,r)) ).
        loss = (support * (triple_loss + path_relation_loss)).mean()

        # Online LS update: decay LS by gamma when the positive triple is not
        # locally better than its matched negative.
        if self._should_update_local_support():
            if self.use_native_score_for_ls:
                paired_neg_scores = self._pair_negative_scores(pos_scores, neg_scores)
                if self.score_is_distance:
                    q_values_tensor = paired_neg_scores - pos_scores - self.margin
                else:
                    q_values_tensor = pos_scores - paired_neg_scores - self.margin
            else:
                pos_quality_energy = self._get_local_quality_energy(pos_triples_tensor, model)
                neg_quality_energy = self._pair_negative_energies(
                    pos_quality_energy,
                    self._get_local_quality_energy(neg_triples_tensor, model),
                )
                q_values_tensor = -(self.margin + pos_quality_energy - neg_quality_energy)
            q_values = q_values_tensor.detach().cpu().tolist()
            local_support_values = local_support.detach().cpu().tolist()
            for triple, q_val, support_val in zip(pos_triples, q_values, local_support_values):
                if q_val <= 0.0:
                    self._local_support[tuple(triple)] = self.local_decay_gamma * support_val
                else:
                    self._local_support[tuple(triple)] = support_val

        return loss


class DSKRLEHTLoss(DSKRLLoss):
    """Ablation: only the EHT triple-ranking term."""

    def __init__(self, *args, **kwargs):
        kwargs["ablation_mode"] = "eht"
        super().__init__(*args, **kwargs)


class DSKRLPTLoss(DSKRLLoss):
    """Ablation: PT = EHT + RP, without LS/DPS/L(p,r)."""

    def __init__(self, *args, **kwargs):
        kwargs["ablation_mode"] = "pt"
        super().__init__(*args, **kwargs)


class DSKRLLSLoss(DSKRLLoss):
    """Ablation: PT weighted only by LS, without DPS/L(p,r)."""

    def __init__(self, *args, **kwargs):
        kwargs["ablation_mode"] = "ls"
        super().__init__(*args, **kwargs)

class PTrustELoss(nn.Module):
    #Used with Negative Sampling
    def __init__(self, margin=1.0, positive_threshold=0.5):
        super().__init__()
        self.margin = margin
        self.positive_threshold = positive_threshold
    
    def forward(self, pred, target, current_epoch = None):

        if torch.all((target == 0) | (target == 1)):
            pos_mask = target == 1
        else:
            pos_mask = target > self.positive_threshold #Use thresholding when labels are softened/continuous in case of label smoothing
        pos = pred[pos_mask] #Direct model scores instaed of path score matirx
        neg = pred[~pos_mask]

        if pos.numel() == 0 or neg.numel() == 0:
            return pred.new_tensor(0.0, requires_grad=True)

        pos_count = pos.numel()
        neg_count = neg.numel()

        log_margin = pred.new_tensor(self.margin).log() #log(gamma)

        if neg_count % pos_count == 0:
            #Assumes negatives are pre-grouped so each positive has exactly neg_ratio dedicated negatives
            neg_ratio = neg_count // pos_count
            neg = neg.view(neg_ratio, pos_count)
            loss = torch.logaddexp(log_margin, neg - pos.unsqueeze(0)) #logaddexp(a, b) = log(exp(a) + exp(b))
        else:
            #When the ratio isn't uniform, it broadcasts all negatives against all positives
            loss = torch.logaddexp(log_margin, neg.unsqueeze(1) - pos.unsqueeze(0))
            #unsqueeze inserts a size-1 dimension to control how PyTorch broadcasts the subtraction. The indices refer to which axis gets the new size-1 dimension.

        return loss.sum()


class BertKGEContrastiveCCALoss(nn.Module):
    """
    Normal KGE BCE loss plus contrastive alignment to frozen N-BERT triple representations.

    The N-BERT export file must contain:
      - keys: Dice integer triples with shape [N, 3]
      - bert_repr: frozen N-BERT triple representations with shape [N, D]
    """

    requires_x_batch = True
    requires_model = True

    def __init__(
        self,
        nbert_repr_path,
        num_entities,
        num_relations,
        embedding_dim,
        projection_dim=128,
        temperature=0.1,
        lambda_cca=0.1,
        positive_threshold=0.5,
    ):
        super().__init__()
        if not nbert_repr_path:
            raise ValueError("BertKGEContrastiveCCALoss requires --nbert_repr_path.")

        payload = torch.load(nbert_repr_path, map_location="cpu")
        if "keys" not in payload:
            raise ValueError("N-BERT file must contain `keys`. Re-export with --dice_mapping_dir.")
        if "bert_repr" not in payload:
            raise ValueError("N-BERT file must contain `bert_repr`.")

        keys = payload["keys"].long()
        bert_repr = payload["bert_repr"].float()

        if keys.ndim != 2 or keys.size(1) != 3:
            raise ValueError(f"Expected keys shape [N, 3], got {tuple(keys.shape)}.")
        if bert_repr.ndim != 2:
            raise ValueError(f"Expected bert_repr shape [N, D], got {tuple(bert_repr.shape)}.")

        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.temperature = temperature
        self.lambda_cca = lambda_cca
        self.positive_threshold = positive_threshold

        self.register_buffer("bert_repr_table", bert_repr)

        encoded_keys = self._encode_keys(keys)
        self.key_to_row = {int(key): row for row, key in enumerate(encoded_keys.tolist())}

        bert_dim = bert_repr.size(1)
        self.bert_projection = nn.Linear(bert_dim, projection_dim)
        self.kge_projection = nn.LazyLinear(projection_dim)
        self.base_loss = nn.BCEWithLogitsLoss()

    def _encode_keys(self, triples):
        triples = triples.long()
        return (
            triples[:, 0] * (self.num_relations * self.num_entities)
            + triples[:, 1] * self.num_entities
            + triples[:, 2]
        )

    def _lookup_rows(self, triples):
        encoded = self._encode_keys(triples.detach().cpu())
        rows = [self.key_to_row.get(int(key), -1) for key in encoded.tolist()]
        return torch.tensor(rows, device=triples.device, dtype=torch.long)

    def _flatten_inputs(self, x_batch, target):
        if x_batch.dim() == 3:
            x_batch = x_batch.reshape(-1, x_batch.size(-1))
        if x_batch.dim() != 2 or x_batch.size(-1) != 3:
            raise ValueError(
                "BertKGEContrastiveCCALoss requires triple batches shaped [N, 3]. "
                "Use --scoring_technique NegSample first."
            )
        if target.dim() > 1:
            target = target.reshape(-1)
        return x_batch, target

    def _kge_repr(self, triples, model):
        head_repr, rel_repr, tail_repr = model.get_triple_representation(triples)
        return torch.cat(
            [
                head_repr.reshape(head_repr.size(0), -1),
                rel_repr.reshape(rel_repr.size(0), -1),
                tail_repr.reshape(tail_repr.size(0), -1),
            ],
            dim=-1,
        )

    def _contrastive_loss(self, kge_repr, bert_repr):
        kge_z = F.normalize(self.kge_projection(kge_repr), dim=-1)
        bert_z = F.normalize(self.bert_projection(bert_repr), dim=-1)
        logits = torch.matmul(kge_z, bert_z.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def forward(self, pred, target, current_epoch=None, x_batch=None, model=None):
        if x_batch is None or model is None:
            raise ValueError("BertKGEContrastiveCCALoss requires x_batch and model.")

        base_loss = self.base_loss(pred.reshape_as(target).float(), target.float())
        triples, flat_target = self._flatten_inputs(x_batch, target)
        positive_mask = flat_target > self.positive_threshold
        positive_triples = triples[positive_mask]

        if positive_triples.numel() == 0:
            return base_loss

        rows = self._lookup_rows(positive_triples)
        available = rows >= 0
        if not torch.any(available):
            return base_loss

        positive_triples = positive_triples[available].to(pred.device)
        rows = rows[available]
        bert_repr = self.bert_repr_table[rows].to(device=pred.device, dtype=pred.dtype)
        kge_repr = self._kge_repr(positive_triples, model)
        cca_loss = self._contrastive_loss(kge_repr, bert_repr)
        return base_loss + self.lambda_cca * cca_loss


class cca(nn.Module):
    #confidence-weighted KvsAll cross-entropy
    def __init__(self, temp = 0.1, use_confidence = True, warmup_epochs = 5, min_weight = 0.5, reweight_strength = 1.0):
        super().__init__()
        self.T = temp 
        self.use_conf = use_confidence 
        self.warmup_epochs = warmup_epochs 
        self.min_weight = min_weight 
        self.reweight_strength = reweight_strength 
        self.eps = 1e-12 

    def forward(self, pred, target, current_epoch):
        log_prob = F.log_softmax(pred / self.T, dim = 1)

        pos_mass = target.sum(dim=1)
        valid = pos_mass > 0 
        target_norm = target / pos_mass.clamp(min=self.eps).unsqueeze(1) 
        per_sample = -(target_norm * log_prob).sum(dim=1) 

        if not self.use_conf or current_epoch < self.warmup_epochs:
            weights = torch.ones_like(per_sample) #weights all samples 1.0
        else:
            detached = per_sample.detach()
            low, high = detached.min(), detached.max() 
            norm = (detached - low) / (high - low).clamp(min = self.eps) 
            weights = (1.0 - self.reweight_strength * norm).clamp(self.min_weight, 1.0)
        
        weights = weights * valid.float() 
        return (weights * per_sample).sum() / weights.sum().clamp(min = self.eps)











    
