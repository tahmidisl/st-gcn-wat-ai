# ST-GCN Emotion Recognition

Train ST-GCN (Spatial Temporal Graph Convolutional Networks) for emotion recognition using the BOLD dataset.

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test setup
python test_setup.py
```

### 2. Prepare Data
```bash
# Convert BOLD dataset to ST-GCN format
cd code
python prepare_bold_for_stgcn.py --bold_path ../BOLD_public --output_path ../st-gcn/data/BOLD
```

### 3. Train Model

**CPU Training:**
```bash
python train.py --epochs 200 --batch_size 16
```

**GPU Training:**
```bash
python train.py --gpu --epochs 200 --batch_size 64
```

## Training Commands

### Basic Training
```bash
# CPU training (recommended for testing)
python train.py --epochs 200 --batch_size 8

# GPU training (faster)
python train.py --gpu --epochs 200 --batch_size 64

# Debug mode (100 samples only)
python train.py --debug --epochs 5
```

### Resume Training
```bash
# Resume from checkpoint
python main.py recognition \
    -c data/BOLD/train_categorical.yaml \
    --weights work_dir/emotion/bold/ST_GCN_categorical/epoch30_model.pt \
    --start_epoch 30 --num_epoch 200 --batch_size 8 \
    --use_gpu False --device -1
```

### Test Model
```bash
python main.py recognition \
    -c data/BOLD/train_categorical.yaml \
    --phase test \
    --weights work_dir/emotion/bold/ST_GCN_categorical/epoch200_model.pt
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--epochs` | Training epochs | 200 |
| `--batch_size` | Batch size | 16 |
| `--gpu` | Use GPU | False |
| `--lr` | Learning rate | 0.01 |
| `--debug` | Debug mode (100 samples) | False |

## Output

- **Models**: `work_dir/emotion/bold/ST_GCN_categorical/`
- **Logs**: `work_dir/emotion/bold/ST_GCN_categorical/log.txt`
- **Checkpoints**: `epoch{N}_model.pt` files

## Data Format

- **Input**: Skeleton sequences (N, C, T, V, M)
  - N: Batch size
  - C: 2 (x, y coordinates)
  - T: 64 time frames
  - V: 18 joints
  - M: 1 person
- **Output**: 26 emotion categories

## Stop Training

Press `Ctrl + C` in the terminal to stop training safely.

---

Based on [ST-GCN](https://github.com/yysijie/st-gcn) for skeleton-based action recognition. 
