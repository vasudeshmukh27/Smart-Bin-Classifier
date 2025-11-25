"""
Data Loader Module for Smart Bin Classifier
Handles image loading, preprocessing, and dataset creation
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class BinImageDataset(Dataset):
    """
    Custom PyTorch Dataset for Amazon Bin Images
    Supports both classification and metadata features
    """
    
    def __init__(self, image_dir, metadata_dir, image_list=None, transform=None, 
                 target_size=(416, 416)):
        """
        Args:
            image_dir: Path to bin-images directory
            metadata_dir: Path to metadata directory
            image_list: List of image IDs to use (if None, use all)
            transform: Torchvision transforms
            target_size: Target image size (H, W)
        """
        self.image_dir = Path(image_dir)
        self.metadata_dir = Path(metadata_dir)
        self.target_size = target_size
        self.transform = transform
        
        # Load all metadata
        self.metadata = {}
        self.asin_to_idx = {}
        self.idx_to_asin = {}
        self.all_asins = set()
        
        print("Loading metadata files...")
        for json_file in sorted(self.metadata_dir.glob("*.json")):
            image_id = json_file.stem
            with open(json_file, 'r') as f:
                self.metadata[image_id] = json.load(f)
            
            # Collect all unique ASINs
            bin_data = self.metadata[image_id].get('BIN_FCSKU_DATA', {})
            for asin in bin_data.keys():
                self.all_asins.add(asin)
        
        # Create ASIN indexing
        self.all_asins = sorted(list(self.all_asins))
        for idx, asin in enumerate(self.all_asins):
            self.asin_to_idx[asin] = idx
            self.idx_to_asin[idx] = asin
        
        print(f"Found {len(self.all_asins)} unique ASINs")
        
        # Get list of valid images
        if image_list is None:
            self.image_ids = sorted([f.stem for f in self.image_dir.glob("*.jpg")])
        else:
            self.image_ids = image_list
        
        print(f"Dataset contains {len(self.image_ids)} images")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load image
        img_path = self.image_dir / f"{image_id}.jpg"
        image = Image.open(img_path).convert('RGB')
        
        # Get original size for reference
        orig_size = image.size  # (W, H)
        
        # Resize to target size
        image = image.resize(self.target_size, Image.Resampling.BILINEAR)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1)
            image = image / 255.0  # Normalize to [0, 1]
        
        # Get metadata
        meta = self.metadata[image_id]
        expected_qty = meta.get('EXPECTED_QUANTITY', 0)
        bin_data = meta.get('BIN_FCSKU_DATA', {})
        
        # Create multi-label target (which ASINs are present)
        multi_label = torch.zeros(len(self.all_asins), dtype=torch.float32)
        
        for asin, info in bin_data.items():
            asin_idx = self.asin_to_idx[asin]
            multi_label[asin_idx] = 1.0  # Item is present
        
        return {
            'image': image,
            'multi_label': multi_label,
            'expected_qty': torch.tensor(expected_qty, dtype=torch.float32),
            'num_products': torch.tensor(len(bin_data), dtype=torch.float32),
            'image_id': image_id  # Store as string, not dict
        }
    
    def get_asin_name(self, asin_idx):
        """Get ASIN code from index"""
        return self.idx_to_asin.get(asin_idx, "Unknown")
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced data"""
        class_counts = torch.zeros(len(self.all_asins))
        
        for image_id in self.image_ids:
            bin_data = self.metadata[image_id].get('BIN_FCSKU_DATA', {})
            for asin in bin_data.keys():
                asin_idx = self.asin_to_idx[asin]
                class_counts[asin_idx] += 1
        
        # Inverse frequency weighting
        total = class_counts.sum()
        class_weights = total / (class_counts + 1e-6)  # Avoid division by zero
        class_weights = class_weights / class_weights.max()  # Normalize
        
        return class_weights


def custom_collate_fn(batch):
    """
    Custom collate function to handle batching properly
    """
    # Stack tensors
    images = torch.stack([item['image'] for item in batch])
    multi_labels = torch.stack([item['multi_label'] for item in batch])
    expected_qtys = torch.stack([item['expected_qty'] for item in batch])
    num_products = torch.stack([item['num_products'] for item in batch])
    
    # Keep image_ids as list of strings
    image_ids = [item['image_id'] for item in batch]
    
    return {
        'image': images,
        'multi_label': multi_labels,
        'expected_qty': expected_qtys,
        'num_products': num_products,
        'image_id': image_ids
    }


def create_dataloaders(data_path, batch_size=32, val_split=0.15, test_split=0.15, 
                       target_size=(416, 416), num_workers=4, seed=42, pin_memory=False):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_path: Path to data/raw directory
        batch_size: Batch size for training
        val_split: Fraction for validation (0-1)
        test_split: Fraction for test (0-1)
        target_size: Target image size
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility
        pin_memory: Whether to pin memory (set True only when using GPU)
    
    Returns:
        train_loader, val_loader, test_loader, dataset
    """
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    data_path = Path(data_path)
    img_dir = data_path / "bin-images"
    meta_dir = data_path / "metadata"
    
    print("="*60)
    print("CREATING DATA LOADERS")
    print("="*60)
    
    # Create full dataset
    full_dataset = BinImageDataset(
        image_dir=img_dir,
        metadata_dir=meta_dir,
        target_size=target_size
    )
    
    # Split data
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size
    
    print(f"\nDataset split:")
    print(f"  Train: {train_size} ({100*train_size/total_size:.1f}%)")
    print(f"  Val:   {val_size} ({100*val_size/total_size:.1f}%)")
    print(f"  Test:  {test_size} ({100*test_size/total_size:.1f}%)")
    
    indices = np.random.permutation(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )
    
    print(f"\nDataLoader created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {target_size}")
    print(f"  Num unique ASINs: {len(full_dataset.all_asins)}")
    
    return train_loader, val_loader, test_loader, full_dataset


if __name__ == "__main__":
    # Test the data loader
    print("\n" + "="*60)
    print("TESTING DATA LOADER")
    print("="*60 + "\n")
    
    data_path = "../../data/raw"
    
    # Use num_workers=0 and pin_memory=False for testing without GPU access
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        data_path=data_path,
        batch_size=16,
        target_size=(416, 416),
        num_workers=0,  # Single-threaded for testing
        pin_memory=False  # Don't use CUDA
    )
    
    # Get a sample batch
    print("\nGetting sample batch from train loader...")
    sample_batch = next(iter(train_loader))
    
    print(f"\nBatch contents:")
    print(f"  Images shape: {sample_batch['image'].shape}")
    print(f"  Multi-label shape: {sample_batch['multi_label'].shape}")
    print(f"  Expected qty shape: {sample_batch['expected_qty'].shape}")
    print(f"  Num products shape: {sample_batch['num_products'].shape}")
    print(f"  Image IDs (first 3): {sample_batch['image_id'][:3]}")
    
    print(f"\nSample values:")
    print(f"  Expected quantities (first 5): {sample_batch['expected_qty'][:5].tolist()}")
    print(f"  Number of products (first 5): {sample_batch['num_products'][:5].tolist()}")
    
    # Check multi-label targets
    print(f"\nMulti-label analysis (first sample):")
    first_labels = sample_batch['multi_label'][0]
    positive_indices = torch.where(first_labels > 0)[0]
    print(f"  Number of ASINs present: {len(positive_indices)}")
    print(f"  ASIN indices: {positive_indices[:5].tolist()}...")
    
    # Test a few more batches
    print("\nTesting iteration over 3 batches...")
    for i, batch in enumerate(train_loader):
        if i >= 2:
            break
        print(f"  Batch {i+1}: Images {batch['image'].shape}, Labels {batch['multi_label'].shape}")
    
    print("\n" + "="*60)
    print("DATA LOADER TEST SUCCESSFUL!")
    print("="*60)