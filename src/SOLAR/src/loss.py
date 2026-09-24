from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning with Additive Margin (CosFace style).

    Optional class_weights reweights each anchor contribution by inverse class
    frequency (imbalance-mode weighted).
    """
    def __init__(
        self,
        temperature=0.07,
        contrast_mode='all',
        base_temperature=0.07,
        margin=0.0,
        class_weights: Optional[Mapping[str, float]] = None,
    ):
        """
        Args:
            margin: Additive margin m. 推荐 0.1 ~ 0.3.
                    如果 margin > 0，会强制正样本对的相似度更高。
            class_weights: Optional mapping from string label -> weight.
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.margin = margin
        self.class_weights = (
            {str(k): float(v) for k, v in class_weights.items()}
            if class_weights is not None
            else None
        )

    def forward(self, features, labels=None, mask=None, class_weights=None):
        device = features.device

        # 归一化特征
        features = F.normalize(features, dim=2)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        weight_lookup = class_weights if class_weights is not None else self.class_weights
        raw_labels = labels

        # 构建 Mask
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            if isinstance(labels[0], str):
                unique_labels = list(dict.fromkeys(labels))
                label_map = {l: i for i, l in enumerate(unique_labels)}
                labels = torch.tensor([label_map[l] for l in labels], device=device)
            else:
                labels = labels.to(device)

            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # 1. 计算 Cosine Similarity (Logits)
        # anchor_dot_contrast: [2N, 2N]
        # 这里先不要除以 temperature，先处理 margin
        logits = torch.matmul(anchor_feature, contrast_feature.T)

        # 2. 扩展 Mask
        mask = mask.repeat(anchor_count, contrast_count)

        # 3. [核心修改] 应用 Additive Margin
        # 逻辑：只对正样本对 (mask==1) 减去 margin
        if self.margin > 0:
            # 这里的逻辑是：logits_pos = cos(theta) - m
            # 我们构建一个 margin_mask，只有在 mask=1 的地方有值
            # 注意：对角线（自己对自己）通常不减 margin，或者减了也没事
            logits = logits - (mask * self.margin)

        # 4. 温度缩放
        logits = torch.div(logits, self.temperature)

        # 5. 屏蔽自对比 (Numerical Stability)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Inactive rows retain zero weight in the historical all-row mean.
        # Define their denominator before log, rather than multiplying log(0)
        # by zero (which produces NaN values and gradients for singleton pools).
        mask_sum = mask.sum(1)
        active = mask_sum > 0
        exp_logits = torch.exp(logits) * logits_mask
        denominator = exp_logits.sum(1, keepdim=True)
        denominator = torch.where(active[:, None], denominator, torch.ones_like(denominator))
        log_prob = logits - torch.log(denominator)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum.clamp_min(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return self._reduce_loss(
            loss,
            raw_labels=raw_labels,
            batch_size=batch_size,
            anchor_count=anchor_count,
            weight_lookup=weight_lookup,
            device=device,
        )

    @staticmethod
    def _reduce_loss(
        loss: torch.Tensor,
        raw_labels,
        batch_size: int,
        anchor_count: int,
        weight_lookup: Optional[Mapping[str, float]],
        device: torch.device,
    ) -> torch.Tensor:
        if weight_lookup is None or raw_labels is None:
            return loss.mean()
        if not isinstance(raw_labels[0], str):
            raise ValueError(
                "class_weights require string labels; got non-string labels."
            )
        per_sample = [
            float(weight_lookup.get(str(label), 1.0)) for label in raw_labels
        ]
        weights = torch.tensor(per_sample, dtype=loss.dtype, device=device)
        weights = weights.repeat(anchor_count)
        if weights.shape[0] != loss.shape[0]:
            raise ValueError(
                f"class weight length {weights.shape[0]} does not match "
                f"loss length {loss.shape[0]} "
                f"(batch_size={batch_size}, anchor_count={anchor_count})."
            )
        weight_sum = weights.sum()
        if weight_sum <= 0:
            return loss.mean()
        return (loss * weights).sum() / weight_sum


def inverse_frequency_weights(labels: Sequence[str]) -> dict[str, float]:
    """Build class-frequency inverse weights (mean-normalized to 1.0)."""
    if len(labels) == 0:
        return {}
    counts: dict[str, int] = {}
    for label in labels:
        key = str(label)
        counts[key] = counts.get(key, 0) + 1
    raw = {key: 1.0 / float(count) for key, count in counts.items()}
    mean_weight = sum(raw.values()) / len(raw)
    if mean_weight <= 0:
        return {key: 1.0 for key in raw}
    return {key: value / mean_weight for key, value in raw.items()}

