"""Implementation of the adjusted Rand index.
   Adapted for PyTorch from
   https://github.com/deepmind/multi_object_datasets/blob/master/segmentation_metrics.py
"""

import torch
import torch.nn.functional as F


def adjusted_rand_index(true_mask, pred_mask):
    """ Computes the adjusted Rand index (ARI), a clustering similarity score.

    Parameters
    ----------
    true_mask: `torch.Tensor` of shape [batch_size, n_groups, ...]
    pred_mask: `torch.Tensor` of shape [batch_size, n_groups, ...]

    Returns
    -------
    out: `torch.Tensor` of shape [batch_size]

    """
    batch_size, n_true_groups, *_ = true_mask.shape

    true_mask = torch.squeeze(true_mask).reshape(batch_size, n_true_groups, -1).transpose(2, 1)
    pred_mask = torch.squeeze(pred_mask).reshape(batch_size, n_true_groups, -1).transpose(2, 1)

    n_points = true_mask.shape[1]
    n_pred_groups = pred_mask.shape[-1]

    assert not (n_points <= n_true_groups and n_points <= n_pred_groups), (
        "adjusted_rand_index requires n_groups < n_points. We don't handle the special cases that can occur when you have one cluster per datapoint.")

    true_group_ids = torch.argmax(true_mask, -1)
    pred_group_ids = torch.argmax(pred_mask, -1)
    true_mask_oh = true_mask.to(torch.float32)
    pred_mask_oh = F.one_hot(pred_group_ids, n_pred_groups).to(torch.float32)

    n_points = torch.sum(true_mask_oh, dim=[1, 2]).to(torch.float32)

    bki = torch.einsum('bji,bjk->bki', pred_mask_oh, true_mask_oh)
    a = torch.sum(bki, dim=1)
    b = torch.sum(bki, dim=2)

    rindex = torch.sum(bki * (bki - 1), dim=[1, 2])
    aindex = torch.sum(a * (a - 1), dim=1)
    bindex = torch.sum(b * (b - 1), dim=1)
    expected_rindex = aindex * bindex / (n_points * (n_points - 1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

    both_single_cluster = _all_equal(true_group_ids) & _all_equal(pred_group_ids)
    return torch.where(both_single_cluster, torch.ones_like(ari), ari)


def _all_equal(values):
    eq = values == values[..., :1]
    return torch.all(eq, dim=-1)

if __name__ == '__main__':
    torch.manual_seed(42)
    true_mask = torch.randint(0, 2, (10, 6, 1, 128, 128)).float()
    pred_mask = torch.rand((10, 6, 1, 128, 128)).float()

    ari = adjusted_rand_index(true_mask, pred_mask)
    print(ari)



