#!/usr/bin/env python
"""
Simple training script for ST-GCN emotion recognition
"""
import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description='Train ST-GCN for emotion recognition')
    
    # Common training arguments
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training (default: CPU)')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID (if using GPU)')
    
    # Advanced options
    parser.add_argument('--config', default='data/BOLD/train_categorical.yaml', help='Config file path')
    parser.add_argument('--work_dir', default=None, help='Work directory for outputs')
    parser.add_argument('--save_interval', type=int, default=10, help='Save model every N epochs')
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluate every N epochs')
    parser.add_argument('--debug', action='store_true', help='Use only 100 samples for debugging')
    
    args = parser.parse_args()
    
    # Build command for main.py
    cmd = [
        sys.executable, 'main.py', 'recognition',
        '-c', args.config,
        '--num_epoch', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--test_batch_size', str(args.batch_size),
        '--base_lr', str(args.lr),
        '--save_interval', str(args.save_interval),
        '--eval_interval', str(args.eval_interval)
    ]
    
    # GPU/CPU device selection
    if args.gpu:
        cmd.extend(['--device', str(args.device)])
        print(f"Training on GPU {args.device}")
    else:
        cmd.extend(['--device', '-1'])
        print("Training on CPU")
    
    # Work directory
    if args.work_dir:
        cmd.extend(['--work_dir', args.work_dir])
    
    # Debug mode
    if args.debug:
        cmd.append('--debug')
        print("Debug mode: using only 100 samples")
    
    # Display training info
    print(f"\nStarting ST-GCN training:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Config: {args.config}")
    print()
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)

if __name__ == '__main__':
    main() 
