#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt



# 1. data_split

def data_split(gt, train_fraction=0.7, rem_classes=None, split_method='same_hist'):
    if rem_classes is None:
        rem_classes = []

    if not isinstance(gt, torch.Tensor):
        gt = torch.tensor(gt)

    rem = torch.tensor(rem_classes, dtype=gt.dtype)

    all_catgs = torch.unique(gt)
    catgs     = all_catgs[~torch.isin(all_catgs, rem)]

    num_pixels      = (~torch.isin(gt, rem)).sum().item()
    counts          = torch.tensor([(gt == c).sum().item() for c in catgs])
    catg_ratios     = counts.float() / counts.sum().float()
    num_sample_catgs = (catg_ratios * num_pixels).floor().long()

    train_rows, train_cols, test_rows, test_cols = [], [], [], []

    for n_samples, catg in zip(num_sample_catgs, catgs):
        r, c = torch.where(gt == catg)
        n = n_samples.item()

        if split_method == 'same_hist':
            n_train = math.floor(n * train_fraction)
            perm        = torch.randperm(n)
            train_idx   = perm[:n_train]
            test_idx    = perm[n_train:]

        elif isinstance(split_method, dict):
            n_train   = split_method.get(catg.item(), 0)
            perm      = torch.randperm(n)
            train_idx = perm[:n_train]
            test_idx  = perm[n_train:]

        else:
            raise ValueError('Please select a valid option')

        train_rows.append(r[train_idx])
        train_cols.append(c[train_idx])
        test_rows.append(r[test_idx])
        test_cols.append(c[test_idx])

    return (
        (torch.cat(train_rows), torch.cat(train_cols)),
        (torch.cat(test_rows),  torch.cat(test_cols))
    )


# 2. reduce_dim 

def reduce_dim(img_data, n_components=0.95):
    """img_data: (H, W, C) numpy array. Returns (H, W, n_components) array."""
    H, W, C = img_data.shape
    img_unravel = img_data.reshape(-1, C)           # (H*W, C)

    pca                  = PCA(n_components=n_components)
    unravel_transformed  = pca.fit_transform(img_unravel)   # (H*W, k)

    return unravel_transformed.reshape(H, W, -1)    # (H, W, k)



# 3. create_patch

def create_patch(data_set, gt, pixel_indices, patch_size=5, label_vect_dict=None):
    """
    data_set      : (H, W, C) numpy array  
    gt            : (H, W)    numpy array of int labels  
    pixel_indices : (row_array, col_array)  
    Returns       : input_tensor (N, C, patch, patch), target_tensor (N, num_classes)
                    — channels-first for PyTorch
    """
    rows, cols = pixel_indices
    if len(rows) != len(cols):
        raise ValueError(
            f'Unmatched rows/cols: {len(rows)} rows vs {len(cols)} cols.'
        )

    max_row, max_col = data_set.shape[0] - 1, data_set.shape[1] - 1
    N, C             = len(rows), data_set.shape[2]

    # (N, C, patch_size, patch_size)  ← channels-first
    input_tensor = np.zeros((N, C, patch_size, patch_size), dtype=np.float32)
    catg_labels  = []

    for idx in range(N):
        patch        = np.zeros((patch_size, patch_size, C), dtype=np.float32)
        pr, pc       = rows[idx], cols[idx]
        top_row      = pr - patch_size // 2
        left_col     = pc - patch_size // 2
        catg_labels.append(gt[pr, pc])

        for i in range(patch_size):
            for j in range(patch_size):
                ri, ci = top_row + i, left_col + j
                if 0 <= ri <= max_row and 0 <= ci <= max_col:
                    patch[i, j, :] = data_set[ri, ci, :]

        input_tensor[idx] = patch.transpose(2, 0, 1)   # HWC → CHW

    if label_vect_dict is None:
        label_vect_dict = label_2_one_hot(np.unique(gt))

    target_tensor = np.array([
        label_vect_dict[label] for label in catg_labels
        if label in label_vect_dict
    ], dtype=np.float32)

    return input_tensor, target_tensor



# 4. label_2_one_hot

def label_2_one_hot(label_list):
    catgs     = np.unique(label_list)
    num_catgs = len(catgs)
    return {
        cls: np.eye(1, num_catgs, i, dtype=np.float32).ravel()
        for i, cls in enumerate(catgs)
    }



# 5. one_hot_2_label

def one_hot_2_label(int_to_vector_dict):
    return {tuple(v): k for k, v in int_to_vector_dict.items()}



# 6. val_split

def val_split(rows, cols, gt, val_fraction=0.1, rem_classes=None,
              split_method='same_hist'):
    if rem_classes is None:
        rem_classes = [-1]

    gt_no_test = np.full(gt.shape, -1, dtype=int)

    # convert tensors → numpy if needed for indexing
    r = rows.numpy() if isinstance(rows, torch.Tensor) else rows
    c = cols.numpy() if isinstance(cols, torch.Tensor) else cols
    gt_no_test[r, c] = gt[r, c]

    (train_rows, train_cols), (val_rows, val_cols) = data_split(
        gt_no_test, 1 - val_fraction, rem_classes, split_method
    )
    return (train_rows, train_cols), (val_rows, val_cols)


# ─────────────────────────────────────────────────────────────────────────────
# 7. rescale_data
# ─────────────────────────────────────────────────────────────────────────────
def rescale_data(data_set, method='standard'):
    """data_set: (H, W, C) numpy array. Returns rescaled numpy array."""
    if not isinstance(data_set, np.ndarray) or data_set.ndim != 3:
        raise ValueError('data_set must be a 3-D numpy array!')

    out = np.zeros_like(data_set, dtype=np.float32)

    for i in range(data_set.shape[-1]):
        ch = data_set[:, :, i].astype(np.float32)
        if method == 'standard':
            out[:, :, i] = (ch - ch.mean()) / ch.std()
        elif method == 'zero_mean':
            out[:, :, i] = ch - ch.mean()
        elif method == 'min_max_norm':
            out[:, :, i] = (ch - ch.min()) / (ch.max() - ch.min())
        elif method == 'mean_norm':
            out[:, :, i] = (ch - ch.mean()) / (ch.max() - ch.min())
        else:
            raise ValueError(f'{method} is not a valid method.')

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 8. calc_metrics  (replaces keras model.evaluate)
# ─────────────────────────────────────────────────────────────────────────────
def calc_metrics(nn_model, test_inputs, y_test, int_to_vector_dict,
                 device='cpu', verbose=True):
    """
    nn_model    : a PyTorch nn.Module (eval mode expected)
    test_inputs : numpy (N, C, H, W)  or torch tensor
    y_test      : numpy (N, num_classes) one-hot
    """
    vector_2_label = one_hot_2_label(int_to_vector_dict)
    label_list     = [vector_2_label.get(tuple(v)) for v in y_test]
    test_catgs, test_catg_counts = np.unique(label_list, return_counts=True)

    nn_model.eval()
    criterion = nn.CrossEntropyLoss()

    from_to_list  = []
    res_container = [(cls, []) for cls in test_catgs]
    i = 0
    for cnt in test_catg_counts:
        from_to_list.append((i, i + cnt))
        i += cnt

    with torch.no_grad():
        for (cls, metrics_list), (fr, to) in zip(res_container, from_to_list):
            x = torch.tensor(test_inputs[fr:to], dtype=torch.float32).to(device)
            y = torch.tensor(y_test[fr:to],      dtype=torch.float32).to(device)

            logits  = nn_model(x)
            loss    = criterion(logits, y.argmax(dim=1)).item()
            preds   = logits.argmax(dim=1)
            targets = y.argmax(dim=1)
            acc     = (preds == targets).float().mean().item()

            metrics_list.append({'loss': loss, 'accuracy': acc})

    model_metrics = {cls: m for cls, m in res_container}
    if verbose:
        for key, val in model_metrics.items():
            print(key, val)
    return model_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 9. plot_partial_map
# ─────────────────────────────────────────────────────────────────────────────
def plot_partial_map(nn_model, gt, pixel_indices, input_tensor, targ_tensor,
                     int_to_vector_dict, device='cpu', plo=True):
    rows, cols      = pixel_indices
    vect_2_label    = one_hot_2_label(int_to_vector_dict)

    nn_model.eval()
    x = torch.tensor(input_tensor, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = nn_model(x)                        # (N, num_classes)

    pred_indices = logits.argmax(dim=1).cpu().numpy()
    num_classes  = logits.shape[1]

    y_pred = np.array([
        vect_2_label.get(tuple(np.eye(1, num_classes, k=idx, dtype=int).ravel()))
        for idx in pred_indices
    ], dtype=int)

    gt_pred_map = np.zeros(gt.shape, dtype=int)
    for i, (r, c) in enumerate(zip(rows, cols)):
        gt_pred_map[r, c] = y_pred[i]

    if plo:
        plt.imshow(gt_pred_map)
    return gt_pred_map


# ─────────────────────────────────────────────────────────────────────────────
# 10. plot_full_map
# ─────────────────────────────────────────────────────────────────────────────
def plot_full_map(nn_model, data_set, gt, int_to_vector_dict,
                  patch_size, device='cpu', plo=True):

    rr, cc            = np.meshgrid(np.arange(gt.shape[0]), np.arange(gt.shape[1]))
    all_pixel_indices = (rr.ravel(), cc.ravel())
    vector_2_label    = one_hot_2_label(int_to_vector_dict)

    all_inputs, _ = create_patch(
        data_set, gt, all_pixel_indices, patch_size, int_to_vector_dict
    )

    # ── detect Conv3D vs Conv2D automatically ────────────────────────────
    def _needs_depth_dim(model):
        for module in model.modules():
            if isinstance(module, torch.nn.Conv3d):
                return True
            if isinstance(module, torch.nn.Conv2d):
                return False
        return False

    if _needs_depth_dim(nn_model):
        all_inputs = np.expand_dims(all_inputs, axis=1)   # (N,C,H,W) → (N,1,C,H,W)

    nn_model.eval()
    x = torch.tensor(all_inputs, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = nn_model(x)

    pred_indices = logits.argmax(dim=1).cpu().numpy()
    num_classes  = logits.shape[1]

    all_y_pred = np.array([
        vector_2_label.get(tuple(np.eye(1, num_classes, k=idx, dtype=int).ravel()))
        for idx in pred_indices
    ], dtype=int)

    gt_pred_all_map = np.zeros(gt.shape, dtype=int)
    for i, (r, c) in enumerate(zip(rr.ravel(), cc.ravel())):
        gt_pred_all_map[r, c] = all_y_pred[i]

    if plo:
        plt.imshow(gt_pred_all_map)
    return gt_pred_all_map

# ─────────────────────────────────────────────────────────────────────────────
# 11. ZeroPad3DIfNeeded  (replaces zero_pad_3D)
# ─────────────────────────────────────────────────────────────────────────────
class ZeroPad3DIfNeeded(nn.Module):
    """
    Appends zero-padding on the last spatial dim (W) so it is
    divisible by pool_size — mirrors zero_pad_3D() logic exactly.
    """
    def __init__(self, pool_layer: nn.Module):
        super().__init__()

        if not isinstance(pool_layer, (nn.MaxPool3d, nn.AvgPool3d)):
            raise TypeError(
                'ZeroPad3DIfNeeded must follow a Pool3D layer.'
            )

        ks = pool_layer.kernel_size
        st = pool_layer.stride
        if isinstance(ks, int): ks = (ks, ks, ks)
        if isinstance(st, int): st = (st, st, st)

        if ks != st:
            raise ValueError('stride must equal kernel_size in the pooling layer.')

        self.pool_size = ks

    def forward(self, x):
        # x: (B, C, D, H, W)
        stride    = self.pool_size[-1]
        remainder = x.shape[-1] % stride
        if remainder:
            x = F.pad(x, (0, stride - remainder))
        return x