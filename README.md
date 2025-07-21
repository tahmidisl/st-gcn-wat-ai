
## Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Test your setup:**
```bash
python test_setup.py
```

### 2. Prepare BOLD Dataset

1. Download the BOLD dataset and place it in the project root as `BOLD_public/`
2. Convert BOLD data to ST-GCN format:

```bash
cd code
python prepare_bold_for_stgcn.py --bold_path ../BOLD_public --output_path ../st-gcn/data/BOLD
```

This creates:
- `train_data_categorical.npy` - Training skeleton data
- `train_label_categorical.pkl` - Training emotion labels  
- `val_data_categorical.npy` - Validation skeleton data
- `val_label_categorical.pkl` - Validation emotion labels
- `train_categorical.yaml` - Training configuration

### 3. Train the Model

**Simple training script (recommended):**
```bash
cd st-gcn

# CPU training
python train.py --epochs 200 --batch_size 16

# GPU training  
python train.py --gpu --epochs 200 --batch_size 64 --lr 0.01
```

**Advanced training (direct):**
```bash
python main.py recognition \
    -c data/BOLD/train_categorical.yaml \
    --device 0 \
    --batch_size 64 \
    --num_epoch 200 \
    --base_lr 0.01
```

## Key Parameters

**Simple script (`train.py`):**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--epochs` | Number of training epochs | 200 |
| `--batch_size` | Training batch size | 16 |
| `--gpu` | Use GPU for training | False |
| `--lr` | Learning rate | 0.01 |
| `--debug` | Use only 100 samples | False |

**Advanced (`main.py`):**
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_epoch` | Number of training epochs | 200 |
| `--batch_size` | Training batch size | 16 |
| `--device` | GPU device (0) or CPU (-1) | [-1] |
| `--base_lr` | Learning rate | 0.01 |
| `--save_interval` | Save model every N epochs | 10 |
| `--eval_interval` | Evaluate every N epochs | 5 |

## Configuration Files

Edit `st-gcn/data/BOLD/train_categorical.yaml` to customize:

```yaml
# Training settings
device: [0]        # GPU (0) or CPU ([-1])
batch_size: 64     # Batch size
num_epoch: 200     # Number of epochs
base_lr: 0.01      # Learning rate
step: [50, 100, 150]  # LR decay epochs

# Model settings  
model_args:
  num_class: 26    # Number of emotion classes
  dropout: 0.5     # Dropout rate
```

## Output

- **Models**: Saved in `work_dir/emotion/bold/ST_GCN_categorical/`
- **Logs**: Training progress and metrics
- **Best model**: `epoch{N}_model.pt` files

## Testing Trained Model

```bash
python main.py recognition \
    -c data/BOLD/train_categorical.yaml \
    --phase test \
    --weights work_dir/emotion/bold/ST_GCN_categorical/epoch200_model.pt
```

## Quick Examples

**Fast debug training (100 samples):**
```bash
python train.py --debug --epochs 5
```

**Full GPU training:**
```bash
python train.py --gpu --epochs 200 --batch_size 64
```

**Custom learning rate:**
```bash
python train.py --gpu --lr 0.005 --epochs 100
```

## Data Format

- **Input**: Skeleton sequences (N, C, T, V, M)
  - N: Number of samples
  - C: Coordinate channels (x, y)  
  - T: Time frames
  - V: Number of joints (18)
  - M: Number of persons
  
- **Output**: 26 emotion categories



---

Based on [ST-GCN](https://github.com/yysijie/st-gcn) for skeleton-based action recognition. 
