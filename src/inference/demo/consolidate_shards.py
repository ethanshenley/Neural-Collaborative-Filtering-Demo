#!/usr/bin/env python3
"""
Consolidate a sharded checkpoint directory into a single-file state_dict (my_model.pt).

Usage:
  python consolidate_shards.py
"""

import torch
import os
import sys
import logging
import json
from collections import OrderedDict

# Import your model code
from src.model.architecture import AdvancedNCF

def analyze_checkpoint(model, sharded_dir):
    """Analyze the checkpoint shards and model parameters"""
    # Get model parameter and buffer info
    param_info = []
    total_params = 0
    
    # Add parameters
    for name, param in model.named_parameters():
        param_info.append({
            'name': name,
            'shape': list(param.shape),
            'numel': param.numel()
        })
        print(f"Model param: {name} - Shape: {param.shape}")
        total_params += param.numel()
    
    # Add buffers
    for name, buffer in model.named_buffers():
        param_info.append({
            'name': name,
            'shape': list(buffer.shape),
            'numel': buffer.numel()
        })
        print(f"Model buffer: {name} - Shape: {buffer.shape}")
        total_params += buffer.numel()
    
    print(f"\nModel has {len(param_info)} parameters/buffers with {total_params} total elements")
    
    # Rest of the function remains the same
    data_dir = os.path.join(sharded_dir, "data")
    shard_info = []
    
    for file_name in sorted(os.listdir(data_dir), key=lambda x: int(x)):
        file_path = os.path.join(data_dir, file_name)
        shard_idx = int(file_name)
        
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            tensor = torch.frombuffer(raw_data, dtype=torch.float32)
            shard_info.append({
                'index': shard_idx,
                'size': tensor.shape[0],
                'file': file_name
            })
            print(f"Shard {shard_idx}: {tensor.shape[0]} elements")
            
            if shard_idx < len(param_info):
                expected_shape = param_info[shard_idx]['shape']
                expected_size = param_info[shard_idx]['numel']
                print(f"  Expected shape for {param_info[shard_idx]['name']}: {expected_shape} ({expected_size} elements)")
    
    return param_info, shard_info

def consolidate_checkpoint(model, sharded_dir, param_info):
    """Consolidate shards into a single state dict"""
    state_dict = OrderedDict()
    data_dir = os.path.join(sharded_dir, "data")
    
    # Create a mapping of tensor sizes to parameter names
    size_to_param = {}
    for info in param_info:
        size = info['numel']
        if size not in size_to_param:
            size_to_param[size] = []
        size_to_param[size].append(info)
    
    # Special case for temporal_encoding.pe
    temporal_pe_size = 11680  # 365 days * 32 dim
    
    # Load each shard and try to match by size
    for file_name in sorted(os.listdir(data_dir), key=lambda x: int(x)):
        file_path = os.path.join(data_dir, file_name)
        
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            tensor = torch.frombuffer(raw_data, dtype=torch.float32)
            size = tensor.numel()
            
            # Special handling for temporal_encoding.pe
            if size == temporal_pe_size:
                tensor = tensor.view(365, 32)  # Reshape to expected dimensions
                state_dict['temporal_encoding.pe'] = tensor
                print(f"Mapped shard {file_name} ({size} elements) to temporal_encoding.pe")
                continue
            
            # Find matching parameter by size
            if size in size_to_param and size_to_param[size]:
                param_meta = size_to_param[size].pop(0)
                try:
                    tensor = tensor.view(param_meta['shape'])
                    state_dict[param_meta['name']] = tensor
                    print(f"Mapped shard {file_name} ({size} elements) to {param_meta['name']}")
                except RuntimeError as e:
                    print(f"Error reshaping shard {file_name}: {e}")
            else:
                print(f"No matching parameter found for shard {file_name} with {size} elements")
    
    return state_dict

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Path to sharded checkpoint
    sharded_dir = "train_20241225_002713_model"
    if not os.path.isdir(sharded_dir):
        print(f"ERROR: Directory not found: {sharded_dir}")
        sys.exit(1)
    
    print(f"Loading sharded checkpoint from: {sharded_dir}")
    
    # Initialize model
    model = AdvancedNCF(
        num_users=8031,
        num_products=366,
        num_departments=5,
        num_categories=24,
        mf_embedding_dim=64,
        mlp_embedding_dim=64,
        temporal_dim=32,
        mlp_hidden_dims=[256, 128, 64],
        num_heads=4,
        dropout=0.2,
        negative_samples=4
    ).cpu()
    
    # Analyze checkpoint
    param_info, shard_info = analyze_checkpoint(model, sharded_dir)
    
    # Consolidate checkpoint
    print("\nConsolidating checkpoint...")
    state_dict = consolidate_checkpoint(model, sharded_dir, param_info)
    
    # Try loading into model
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: Missing keys: {missing}")
    if unexpected:
        print(f"WARNING: Unexpected keys: {unexpected}")
    
    # Save consolidated checkpoint
    output_file = "my_model.pt"
    torch.save(state_dict, output_file)
    print(f"\nSaved single-file checkpoint to: {output_file}")
    print("\nYou can now load 'my_model.pt' in local_inference.py with:")
    print("  state_dict = torch.load('my_model.pt')")
    print("  model.load_state_dict(state_dict)")
    print("\nDone!")

if __name__ == "__main__":
    main()
