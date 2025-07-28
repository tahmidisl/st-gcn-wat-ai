#!/usr/bin/env python3
"""
BOLD Dataset to ST-GCN Format Converter
Converts BOLD dataset pose data to ST-GCN compatible format for emotion recognition.
"""

import os
import numpy as np
import pandas as pd
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import torch

class BOLDToSTGCNConverter:
    def __init__(self, bold_path, output_path):
        self.bold_path = Path(bold_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        self.coco_layout = {
            0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
            5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
            10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
            15: "LEye", 16: "REar", 17: "LEar"
        }
        
        # Emotion categories (26 total)
        self.emotion_categories = [
            'Peace', 'Affection', 'Esteem', 'Anticipation', 'Engagement', 
            'Confidence', 'Happiness', 'Pleasure', 'Excitement', 'Surprise',
            'Sympathy', 'Doubt/Confusion', 'Disconnection', 'Fatigue', 'Embarrassment',
            'Yearning', 'Disapproval', 'Aversion', 'Annoyance', 'Anger',
            'Sensitivity', 'Sadness', 'Disquietment', 'Fear', 'Pain', 'Suffering'
        ]
        
    def load_annotations(self, split):
        """Load train/val/test annotations"""
        ann_file = self.bold_path / "annotations" / f"{split}.csv"
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        column_names = ['file_name', 'person_id', 'start_frame', 'end_frame', 'categorical_emotion'] + \
                      [f'emotion_{i}' for i in range(26)] + ['valence', 'arousal', 'dominance', 'gender', 'age', 'ethnicity']
        
        df = pd.read_csv(ann_file, header=None)
        # The categorical emotion is actually columns 4-29 (26 emotions)
        # Convert float values to binary (threshold at 0.5)
        emotion_cols = df.iloc[:, 4:30]
        df['categorical_emotion'] = emotion_cols.apply(
            lambda row: ''.join(['1' if val > 0.5 else '0' for val in row]), axis=1
        )
        df['file_name'] = df.iloc[:, 0]
        df['person_id'] = df.iloc[:, 1] 
        df['start_frame'] = df.iloc[:, 2]
        df['end_frame'] = df.iloc[:, 3]
        df['valence'] = df.iloc[:, 30]
        df['arousal'] = df.iloc[:, 31]
        df['dominance'] = df.iloc[:, 32]
        
        print(f"Loaded {len(df)} samples for {split} split")
        return df
    
    def parse_emotion_vector(self, emotion_str):
        """Parse emotion binary string to multi-hot vector"""
        if pd.isna(emotion_str) or len(emotion_str) != 26:
            return np.zeros(26)
        return np.array([int(c) for c in emotion_str])
    
    def load_joint_data(self, video_name, person_id, start_frame, end_frame):
        """Load joint data for specific video, person, and frame range"""
        parts = video_name.split('/')
        if len(parts) >= 3:
            base_video = parts[1]  # "videoname.mp4"
            segment_num = parts[2].replace('.mp4', '')  # e.g., "0959"
            joint_file = self.bold_path / "joints" / "003" / base_video / f"{segment_num}.npy"
        else:
            joint_file = self.bold_path / "joints" / video_name / f"{person_id:04d}.npy"
        if not joint_file.exists():
            print(f"Warning: Joint file not found: {joint_file}")
            return None
        try:
            joints = np.load(joint_file)  # Shape: (N, 56)
            person_mask = joints[:, 1] == person_id
            person_joints = joints[person_mask]
            
            if len(person_joints) == 0:
                print(f"Warning: No data for person {person_id} in {joint_file}")
                return None
            available_frames = np.unique(person_joints[:, 0])
            total_frames = len(available_frames)
            if start_frame >= total_frames:
                print(f"Warning: Start frame {start_frame} exceeds available frames ({total_frames}) for {joint_file}")
                return None
            actual_end = min(end_frame, total_frames - 1)
            if start_frame > actual_end:
                print(f"Warning: Invalid frame range [{start_frame}, {actual_end}] for {joint_file}")
                return None
            selected_frame_nums = available_frames[start_frame:actual_end + 1]
            frame_mask = np.isin(person_joints[:, 0], selected_frame_nums)
            filtered_joints = person_joints[frame_mask]
            
            if len(filtered_joints) == 0:
                print(f"Warning: No frames found for person {person_id} in range [{start_frame}, {actual_end}] for {joint_file}")
                return None
            filtered_joints = filtered_joints[np.argsort(filtered_joints[:, 0])]
            pose_data = filtered_joints[:, 2:]  # (T, 54)
            pose_data = pose_data.reshape(-1, 18, 3)  # (T, 18, 3) (x, y, confidence)     
            return pose_data
        except Exception as e:
            print(f"Error loading {joint_file}: {e}")
            return None
    
    def normalize_pose_data(self, pose_data):
        """Normalize pose data to ST-GCN format"""
        if pose_data is None:
            return None   
        T, V, C = pose_data.shape  # T=frames, V=joints(18), C=coords(3)
        pose_xy = pose_data[:, :, :2]  # (T, 18, 2)
        neck_pos = pose_xy[:, 1:2, :]  # (T, 1, 2)
        normalized_pose = pose_xy - neck_pos
        # Transpose to ST-GCN format: (C, T, V, M) where M=1 (single person)
        # C=2 (x,y), T=frames, V=18 joints, M=1 person
        stgcn_pose = normalized_pose.transpose(2, 0, 1)  # (2, T, 18)
        stgcn_pose = np.expand_dims(stgcn_pose, axis=-1)  # (2, T, 18, 1)
        return stgcn_pose
    
    def pad_or_crop_sequence(self, pose_data, target_length=64):
        """Pad or crop sequence to target length"""
        if pose_data is None:
            return None
        C, T, V, M = pose_data.shape
        if T == target_length:
            return pose_data
        elif T < target_length:
            padding = np.tile(pose_data[:, -1:, :, :], (1, target_length - T, 1, 1))
            return np.concatenate([pose_data, padding], axis=1)
        else:
            indices = np.linspace(0, T-1, target_length, dtype=int)
            return pose_data[:, indices, :, :]
    def create_emotion_labels(self, df, task='categorical'):
        """Create labels for different emotion recognition tasks"""
        labels = []
        for idx, row in df.iterrows():
            if task == 'categorical':
                emotion_vector = self.parse_emotion_vector(row['categorical_emotion'])
                primary_idx = np.argmax(emotion_vector) if np.sum(emotion_vector) > 0 else 0
                labels.append(primary_idx)
            elif task == 'valence':
                labels.append(float(row['valence']))
            elif task == 'arousal': 
                labels.append(float(row['arousal']))
            elif task == 'dominance':
                labels.append(float(row['dominance']))
            elif task == 'primary_emotion':
                emotion_vector = self.parse_emotion_vector(row['categorical_emotion'])
                primary_idx = np.argmax(emotion_vector) if np.sum(emotion_vector) > 0 else 0
                labels.append(primary_idx)
        return np.array(labels)
    def process_split(self, split, task='categorical', sequence_length=64):
        """Process a single split (train/val/test)"""
        print(f"\nProcessing {split} split for {task} task...")
        df = self.load_annotations(split)
        all_data = []
        all_labels = []
        valid_samples = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
            video_name = row['file_name']
            person_id = int(row['person_id'])
            start_frame = int(row['start_frame'])
            end_frame = int(row['end_frame'])
            if end_frame <= start_frame:
                continue
            pose_data = self.load_joint_data(video_name, person_id, start_frame, end_frame)
            if pose_data is None:
                continue
            normalized_pose = self.normalize_pose_data(pose_data)
            if normalized_pose is None:
                continue
            final_pose = self.pad_or_crop_sequence(normalized_pose, sequence_length)
            if final_pose is None:
                continue
            all_data.append(final_pose)
            valid_samples += 1
        if valid_samples == 0:
            print(f"No valid samples found for {split} split!")
            return
        data_array = np.array(all_data)  # (N, C, T, V, M)
        print(f"Data shape for {split}: {data_array.shape}")
        valid_df = df.iloc[:valid_samples].reset_index(drop=True)
        labels = self.create_emotion_labels(valid_df, task)
        data_file = self.output_path / f"{split}_data_{task}.npy"
        label_file = self.output_path / f"{split}_label_{task}.pkl" 
        np.save(data_file, data_array)
        sample_names = [f"{split}_{i:06d}" for i in range(len(labels))]
        with open(label_file, 'wb') as f:
            pickle.dump((sample_names, labels), f)
        print(f"Saved {valid_samples} samples to {data_file}")
        print(f"Saved labels to {label_file}")
        return data_array, labels
    
    def create_config_file(self, task='categorical', num_classes=26):
        """Create ST-GCN config file for BOLD dataset"""
        use_gpu = torch.cuda.is_available()
        device_val = "[0]" if use_gpu else "[-1]"
        batch_size_val = 32 if use_gpu else 16
        config_template = f"""work_dir: ./work_dir/emotion/bold/ST_GCN_{task}
# feeder
feeder: feeder.feeder.Feeder
train_feeder_args:
  data_path: {self.output_path}/train_data_{task}.npy
  label_path: {self.output_path}/train_label_{task}.pkl
test_feeder_args:
  data_path: {self.output_path}/val_data_{task}.npy
  label_path: {self.output_path}/val_label_{task}.pkl

# model
model: net.st_gcn.Model
model_args:
  in_channels: 2
  num_class: {num_classes}
  dropout: 0.5
  edge_importance_weighting: True
  graph_args:
    layout: 'openpose'
    strategy: 'spatial'

#optim
weight_decay: 0.0001
base_lr: 0.01
step: [20, 40]

# training
device = {device_val}
batch_size = {batch_size_val}
test_batch_size: {batch_size_val}
num_epoch: 60
"""
        config_file = self.output_path / f"train_{task}.yaml"
        with open(config_file, 'w') as f:
            f.write(config_template)
            
        print(f"Created config file: {config_file}")
def main():
    parser = argparse.ArgumentParser(description='Convert BOLD dataset to ST-GCN format')
    parser.add_argument('--bold_path', type=str, default='./BOLD_public', 
                       help='Path to BOLD dataset')
    parser.add_argument('--output_path', type=str, default='./st-gcn/data/BOLD',
                       help='Output path for processed data')
    parser.add_argument('--task', type=str, default='categorical',
                       choices=['categorical', 'valence', 'arousal', 'dominance', 'primary_emotion'],
                       help='Emotion recognition task')
    parser.add_argument('--sequence_length', type=int, default=64,
                       help='Target sequence length')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                       help='Dataset splits to process')
    args = parser.parse_args()
    converter = BOLDToSTGCNConverter(args.bold_path, args.output_path)
    for split in args.splits:
        try:
            converter.process_split(split, args.task, args.sequence_length)
        except Exception as e:
            print(f"Error processing {split} split: {e}")
            continue
    if args.task == 'categorical':
        num_classes = 26
    elif args.task == 'primary_emotion':
        num_classes = 26
    else:
        num_classes = 1 
    converter.create_config_file(args.task, num_classes)
    print(f"\nConversion complete! Files saved to: {args.output_path}")
    print(f"1. Install ST-GCN dependencies: cd st-gcn && pip install -r requirements.txt")
    print(f"2. Install torchlight: cd torchlight && python setup.py install")
    print(f"3. Train model: python main.py recognition -c data/BOLD/train_{args.task}.y

if __name__ == "__main__":
    main() 
