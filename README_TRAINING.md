# ST-GCN Training Guide

This guide explains how to train the Spatial Temporal Graph Convolutional Network (ST-GCN) for skeleton-based action recognition.

## Table of Contents
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pre-trained Models](#pre-trained-models)
- [Troubleshooting](#troubleshooting)

## Requirements

### Dependencies
Install the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- PyTorch
- torchvision
- numpy
- pyyaml
- h5py
- opencv-python
- imageio
- scikit-video

### Hardware Requirements
- GPU recommended (training can use multiple GPUs)
- Sufficient RAM for data loading
- Storage space for datasets and models

## Data Preparation

### Data Format
The ST-GCN model expects skeleton data in the following format:

- **Training data**: `.npy` file with shape `(N, C, T, V, M)`
  - `N`: Number of samples
  - `C`: Number of channels (typically 3 for x, y, z coordinates)
  - `T`: Number of frames (temporal dimension)
  - `V`: Number of joints/vertices
  - `M`: Number of persons

- **Labels**: `.pkl` file containing `(sample_names, labels)` tuple

### Supported Datasets

#### 1. NTU RGB+D Dataset

Generate NTU dataset using the provided tool:

```bash
cd tools
python ntu_gendata.py \
    --data_path path/to/nturgb+d_skeletons \
    --out_folder ./data/NTU-RGB-D
```

This creates:
- `./data/NTU-RGB-D/xsub/train_data.npy`
- `./data/NTU-RGB-D/xsub/train_label.pkl`
- `./data/NTU-RGB-D/xsub/val_data.npy`
- `./data/NTU-RGB-D/xsub/val_label.pkl`
- Similar files for `xview` benchmark

#### 2. Kinetics Skeleton Dataset

Generate Kinetics dataset:

```bash
cd tools
python kinetics_gendata.py \
    --data_path ./data/Kinetics/kinetics-skeleton \
    --out_folder ./data/Kinetics/kinetics-skeleton
```

#### 3. Custom Dataset

For custom datasets, ensure your data follows the expected format:

```python
# Example data preparation
import numpy as np
import pickle

# Create training data: (N_samples, 3, T_frames, V_joints, M_persons)
train_data = np.random.rand(1000, 3, 300, 25, 2)
train_labels = list(range(1000))
train_names = [f'sample_{i}' for i in range(1000)]

# Save data
np.save('train_data.npy', train_data)
with open('train_label.pkl', 'wb') as f:
    pickle.dump((train_names, train_labels), f)
```

## Configuration

### Configuration Files

Training configurations are stored in YAML files under `config/`. Key configuration sections:

#### Model Configuration
```yaml
# model
model: net.st_gcn.Model
model_args:
  in_channels: 3
  num_class: 60  # Number of action classes
  dropout: 0.5
  edge_importance_weighting: True
  graph_args:
    layout: 'ntu-rgb+d'  # or 'openpose'
    strategy: 'spatial'
```

#### Data Configuration
```yaml
# feeder
feeder: feeder.feeder.Feeder
train_feeder_args:
  data_path: ./data/NTU-RGB-D/xsub/train_data.npy
  label_path: ./data/NTU-RGB-D/xsub/train_label.pkl
  random_choose: True
  random_move: True
  window_size: 300
test_feeder_args:
  data_path: ./data/NTU-RGB-D/xsub/val_data.npy
  label_path: ./data/NTU-RGB-D/xsub/val_label.pkl
```

#### Training Configuration
```yaml
# training parameters
device: [0,1,2,3]  # GPU devices
batch_size: 64
test_batch_size: 64
num_epoch: 80

# optimization
base_lr: 0.1
step: [10, 50]  # Learning rate decay epochs
weight_decay: 0.0001
optimizer: SGD
```

#### Work Directory
```yaml
work_dir: ./work_dir/recognition/my_experiment
```

### Available Configurations

Pre-configured training setups:

- `config/st_gcn/ntu-xsub/train.yaml` - NTU RGB+D Cross-Subject
- `config/st_gcn/ntu-xview/train.yaml` - NTU RGB+D Cross-View  
- `config/st_gcn/kinetics-skeleton/train.yaml` - Kinetics Skeleton
- `config/st_gcn.twostream/` - Two-stream variants

## Training

### Basic Training Command

```bash
python main.py recognition -c config/st_gcn/ntu-xsub/train.yaml
```

### Training Arguments

Common training arguments:

```bash
python main.py recognition \
    -c config/st_gcn/ntu-xsub/train.yaml \
    --work_dir ./work_dir/my_experiment \
    --device 0 1 2 3 \
    --batch_size 32 \
    --num_epoch 100 \
    --base_lr 0.05
```

### Key Parameters

- `-c, --config`: Path to configuration file
- `--work_dir`: Directory to save logs and models
- `--device`: GPU device IDs
- `--batch_size`: Training batch size
- `--num_epoch`: Number of training epochs
- `--base_lr`: Initial learning rate
- `--save_interval`: Model saving interval (epochs)
- `--eval_interval`: Evaluation interval (epochs)

### Multi-GPU Training

For multiple GPUs, specify device IDs:

```bash
python main.py recognition \
    -c config/st_gcn/ntu-xsub/train.yaml \
    --device 0 1 2 3
```

### Resume Training

To resume from a checkpoint:

```bash
python main.py recognition \
    -c config/st_gcn/ntu-xsub/train.yaml \
    --weights ./work_dir/recognition/ntu-xsub/ST_GCN/epoch50_model.pt \
    --start_epoch 50
```

## Evaluation

### Test Trained Model

```bash
python main.py recognition \
    -c config/st_gcn/ntu-xsub/test.yaml \
    --weights ./work_dir/recognition/ntu-xsub/ST_GCN/epoch80_model.pt
```

### Evaluation Configuration

Test configurations are similar to training configs but with:
```yaml
phase: test
weights: path/to/model.pt
```

## Pre-trained Models

### Download Pre-trained Models

```bash
cd tools
bash get_models.sh
```

This downloads pre-trained models for:
- Kinetics skeleton dataset
- NTU RGB+D dataset variants

### Using Pre-trained Models

Load pre-trained weights:

```bash
python main.py recognition \
    -c config/st_gcn/ntu-xsub/train.yaml \
    --weights ./models/st_gcn.ntu-xsub.pt
```

## Training Outputs

During training, the following files are generated in `work_dir`:

- `epochX_model.pt` - Model checkpoints
- `log.txt` - Training logs
- `config.yaml` - Configuration backup

### Monitoring Training

Training progress includes:
- Loss values
- Learning rate
- Top-k accuracy (during evaluation)
- Training time per epoch

Example training log:
```
Training epoch: 0
Mean training loss: 3.892
Eval epoch: 0
Mean test loss: 3.245
Top1: 12.34%
Top5: 35.67%
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size
   - Use fewer GPUs or smaller input resolution

2. **Data loading errors**
   - Check data paths in configuration
   - Verify data format (N, C, T, V, M)
   - Ensure label file contains (names, labels) tuple

3. **Model convergence issues**
   - Adjust learning rate
   - Check data preprocessing
   - Verify class labels are 0-indexed

4. **Configuration errors**
   - Ensure all paths exist
   - Check YAML syntax
   - Verify model arguments match data

### Performance Tips

1. **Data Loading Optimization**
   - Use multiple workers: `--num_worker 8`
   - Enable memory mapping for large datasets

2. **Training Acceleration**
   - Use mixed precision training if available
   - Optimize batch size for your GPU memory
   - Use data augmentation appropriately

3. **Memory Management**
   - Monitor GPU memory usage
   - Clear cache between experiments
   - Use gradient checkpointing for very large models

## Custom Training

### Creating Custom Configuration

1. Copy existing configuration file
2. Modify data paths, model parameters
3. Adjust training hyperparameters
4. Set appropriate work directory

Example custom configuration:
```yaml
work_dir: ./work_dir/recognition/custom_dataset

# feeder  
feeder: feeder.feeder.Feeder
train_feeder_args:
  data_path: ./data/custom/train_data.npy
  label_path: ./data/custom/train_label.pkl
  window_size: 150

# model
model: net.st_gcn.Model  
model_args:
  in_channels: 3
  num_class: 10  # Your number of classes
  edge_importance_weighting: True
  graph_args:
    layout: 'openpose'
    strategy: 'spatial'

# training
device: [0]
batch_size: 32
num_epoch: 50
base_lr: 0.01
```

### Training Custom Models

```bash
python main.py recognition -c config/custom/train.yaml
```

For questions or issues, refer to the original ST-GCN paper and repository documentation. 

### Training on My Local System

```
cd st-gcn
source venv/bin/activate
pip install
python main.py recognition -c data/BOLD/train_categorical.yaml --use_gpu False
python main.py recognition -c data/BOLD/train_categorical.yaml \
    --use_gpu False \
    --phase test \
    --weights work_dir/emotion/bold/ST_GCN_categorical/epoch30_model.pt
```
