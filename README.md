# Hyperspectral Image Classification — PyTorch

A PyTorch implementation of **2D CNN**, **3D CNN**, and **Transformer (D2BERT)** models for hyperspectral image classification on the [Indian Pines](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) benchmark dataset.

---

## Overview

Hyperspectral images contain hundreds of spectral bands per pixel, making them rich sources of information for land-cover classification. This project implements and compares three deep learning architectures — all in pure PyTorch — along with a shared utility library (`img_util_pyt.py`) that handles data splitting, patch extraction, preprocessing, and evaluation.

| Notebook | Architecture | Input | Key idea |
|---|---|---|---|
| `IndianPines2DCNN_PyTorch.ipynb` | 2D CNN | `(N, C, 5, 5)` | Spatial convolution over spectral-PCA patches |
| `IndianPines3DCNN_PyTorch.ipynb` | 3D CNN | `(N, 1, C, 9, 9)` | Joint spectral-spatial convolution |
| `hyperspectral-imaging-attention.ipynb` | D2BERT (Transformer) | `(N, seq, bands)` | Self-attention over flattened patch tokens |

---

## Dataset

**Indian Pines** — AVIRIS hyperspectral sensor, 145×145 pixels, 200 spectral bands (after removing water-absorption bands: 200 → 200 corrected), 16 land-cover classes.

| File | Description |
|---|---|
| `Indian_pines_corrected.mat` | Image data — shape `(145, 145, 200)` |
| `Indian_pines_gt.mat` | Ground truth labels — shape `(145, 145)`, class `0` = background |

Download from the [UPV/EHU Hyperspectral Repository](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) and place both `.mat` files in the same directory as the notebooks.

---

## Project Structure

```
.
├── img_util_pyt.py                        # Shared utility library (PyTorch)
├── IndianPines2DCNN_PyTorch.ipynb         # 2D CNN experiment
├── IndianPines3DCNN_PyTorch.ipynb         # 3D CNN experiment
├── hyperspectral-imaging-attention.ipynb  # D2BERT Transformer experiment
└── README.md
```

---

## Installation

```bash
pip install torch torchvision numpy scipy matplotlib scikit-learn
```

Python 3.9+ recommended. GPU optional but speeds up the 3D CNN and Transformer significantly.

---

## `img_util_pyt.py` — Utility Library

All shared preprocessing and evaluation logic lives here. No Keras/TensorFlow dependency.

### Functions

#### `data_split(gt, train_fraction, rem_classes, split_method)`
Splits a 2D ground-truth label map into train/test pixel index sets while preserving class proportions.

- Returns `(train_rows, train_cols), (test_rows, test_cols)` as **torch tensors**
- `rem_classes=[0]` excludes background pixels
- `split_method='same_hist'` preserves per-class ratios; pass a `dict` for fixed per-class counts

```python
(train_rows, train_cols), (test_rows, test_cols) = util.data_split(
    gt, train_fraction=0.75, rem_classes=[0]
)
```

#### `val_split(rows, cols, gt, val_fraction)`
Carves a validation set out of an existing train split. Operates on the same 2D label map — background pixels are masked out automatically.

```python
(train_rows_sub, train_cols_sub), (val_rows, val_cols) = util.val_split(
    train_rows, train_cols, gt, val_fraction=0.05
)
```

#### `reduce_dim(img_data, n_components)`
PCA dimensionality reduction on `(H, W, C)` hyperspectral cubes. Flattens spatially, fits PCA, reshapes back.

```python
data_set = util.reduce_dim(data_set, n_components=0.999)  # keep 99.9% variance
```

#### `rescale_data(data_set, method)`
Per-band rescaling of a `(H, W, C)` array. Supported methods:

| `method` | Formula |
|---|---|
| `'standard'` (default) | `(x − μ) / σ` |
| `'zero_mean'` | `x − μ` |
| `'min_max_norm'` | `(x − min) / (max − min)` |
| `'mean_norm'` | `(x − μ) / (max − min)` |

#### `create_patch(data_set, gt, pixel_indices, patch_size, label_vect_dict)`
Extracts spatial patches centred on each labelled pixel. Border pixels are zero-padded.

- Input: `(H, W, C)` numpy array
- Output: `(N, C, patch_size, patch_size)` numpy array — **channels-first** for PyTorch

```python
train_input, y_train = util.create_patch(
    data_set, gt, (train_rows, train_cols),
    patch_size=5, label_vect_dict=int_to_vector_dict
)
```

#### `label_2_one_hot(label_list)` / `one_hot_2_label(dict)`
Bidirectional conversion between integer class labels and one-hot numpy vectors.

#### `calc_metrics(nn_model, test_inputs, y_test, int_to_vector_dict, device)`
Per-class loss and accuracy evaluation. Replacement for Keras `model.evaluate()`.

```python
metrics = util.calc_metrics(model, test_input, y_test, int_to_vector_dict, device=device)
# returns: {class_int: [{'loss': float, 'accuracy': float}], ...}
```

#### `plot_partial_map(...)` / `plot_full_map(...)`
Generates classification maps from model predictions. `plot_full_map` automatically detects whether the model uses `Conv2d` or `Conv3d` and reshapes inputs accordingly — no manual flag needed.

### `ZeroPad3DIfNeeded` (class)
`nn.Module` that dynamically zero-pads the last spatial dimension to be divisible by the pool stride. Used inside the 3D CNN to replace Keras `ZeroPadding3D` after each pooling layer.

```python
ZeroPad3DIfNeeded(pool_size=(1, 1, 3))  # pads spectral dim only
```

---

## Architectures

### 2D CNN (`IndianPines2DCNN_PyTorch.ipynb`)

Processes `(N, C, 5, 5)` patches — `C` PCA-reduced spectral bands treated as independent channels.

```
Conv2d(C → 32, k=3, pad=same) → ReLU → MaxPool2d(2)
Conv2d(32 → 64, k=3, pad=same) → ReLU → MaxPool2d(2)
Flatten → Dropout(0.35)
Linear(→ 512) → ReLU
Linear(→ 256) → ReLU
Linear(→ num_classes)
```

**Key settings:** `patch_size=5`, `lr=1e-4`, `batch_size=32`, `epochs=20`, L2 via `weight_decay=1e-3`

---

### 3D CNN (`IndianPines3DCNN_PyTorch.ipynb`)

Processes `(N, 1, C, 9, 9)` volumes — single-channel 3D cube where the spectral axis is the depth dimension. Allows the model to learn joint spectral-spatial features.

```
Conv3d(1 → 64, k=(3,3,5)) → ReLU → MaxPool3d((1,1,3)) → ZeroPad3D
Conv3d(64 → 32, k=(3,3,5)) → ReLU → MaxPool3d((1,1,3)) → ZeroPad3D
Conv3d(32 → 32, k=(3,3,5)) → ReLU
Flatten → Dropout(0.35)
Linear(→ 128) → ReLU → Linear(→ 128) → ReLU
Linear(→ num_classes)
```

**Key settings:** `patch_size=9`, `kernel=(3,3,5)`, `pool=(1,1,3)` (spectral only), `lr=1e-4`, `epochs=20`

> The `ZeroPad3DIfNeeded` layers replace the Keras-specific `util.zero_pad_3D()` mutation — padding is computed dynamically inside `forward()` based on the runtime tensor shape.

---

### D2BERT Transformer (`hyperspectral-imaging-attention.ipynb`)

Treats each patch as a sequence of spatial tokens, each with a spectral feature vector. Self-attention learns which spatial positions are most discriminative.

```
Input: (N, patch²=25, pca_bands=30)  ← flattened patch pixels as sequence tokens
Linear embedding → d_model=64
Learnable positional embedding
TransformerEncoder (4 layers, 4 heads)
Mean pooling over sequence
Linear(→ num_classes)
```

**Key settings:** `patch_size=5`, `pca_bands=30`, `d_model=64`, `lr=1e-3`, `batch_size=128`, `epochs=20`, Adam optimiser

---

## Workflow (all notebooks)

```
Load .mat files
      ↓
data_split()       — stratified pixel-level train/test split
      ↓
val_split()        — carve validation set from training pixels
      ↓
reduce_dim()       — PCA to reduce spectral bands
      ↓
rescale_data()     — z-score normalisation per band
      ↓
create_patch()     — extract (N, C, H, W) patch tensors
      ↓
DataLoader         — batch iterator for training
      ↓
Train loop         — CrossEntropyLoss + RMSprop/Adam
      ↓
calc_metrics()     — per-class accuracy
      ↓
plot_full_map()    — full scene classification map
```

---

## Results (indicative)

Performance depends on the random seed, number of PCA components, and training epochs. Typical ranges on Indian Pines:

| Model | OA (Overall Accuracy) | Notes |
|---|---|---|
| 2D CNN | ~99.34% | Fast to train, strong spatial baseline |
| 3D CNN | ~85–92% | Better spectral-spatial modelling |
| D2BERT | ~97.78% | Best on classes with fine spectral differences |

---


## Citation / Dataset Reference

```
M. Graña, M. Veganzones, S. Maldonado, "Hyperspectral Remote Sensing Scenes",
GIC, University of the Basque Country. http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
```
