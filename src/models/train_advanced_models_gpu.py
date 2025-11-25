"""
Advanced Model Training with GPU Support
Includes: Data Augmentation, Better Architecture, Mixed Precision
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'data_processing'))
from data_loader import create_dataloaders

# ============================================================
# DATA AUGMENTATION
# ============================================================

class AugmentedDataLoader:
    """DataLoader with data augmentation"""
    def __init__(self, base_loader, augment=True):
        self.base_loader = base_loader
        self.augment = augment
        
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
            ])
    
    def __iter__(self):
        for batch in self.base_loader:
            if self.augment:
                # Apply augmentation to images
                images = batch['image']
                aug_images = []
                for img in images:
                    # Convert back from tensor for augmentation
                    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    aug_img = self.transform(img_np)
                    aug_images.append(aug_img)
                batch['image'] = torch.stack(aug_images)
            yield batch
    
    def __len__(self):
        return len(self.base_loader)

# ============================================================
# IMPROVED RESNET-STYLE ARCHITECTURE
# ============================================================

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class ImprovedQuantityPredictor(nn.Module):
    """ResNet-inspired architecture for quantity prediction"""
    def __init__(self):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x

# ============================================================
# ADVANCED TRAINING WITH MIXED PRECISION
# ============================================================

def train_advanced_model(model, train_loader, val_loader, num_epochs=25):
    """Train with GPU, mixed precision, and advanced techniques"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"ADVANCED TRAINING WITH GPU")
    print(f"{'='*70}")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"{'='*70}\n")
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Mixed precision scaler
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # Data augmentation
    train_loader_aug = AugmentedDataLoader(train_loader, augment=True)
    
    best_val_mae = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [], 'lr': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_mae = 0
        num_batches = 0
        
        pbar = tqdm(train_loader_aug, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        for batch in pbar:
            images = batch['image'].to(device)
            targets = batch['expected_qty'].unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if scaler:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            mae = torch.abs(outputs - targets).mean().item()
            train_loss += loss.item()
            train_mae += mae
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mae': f'{mae:.2f}'})
        
        train_loss /= num_batches
        train_mae /= num_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_mae = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]  ', leave=False)
            for batch in pbar:
                images = batch['image'].to(device)
                targets = batch['expected_qty'].unsqueeze(1).to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                mae = torch.abs(outputs - targets).mean().item()
                
                val_loss += loss.item()
                val_mae += mae
                num_batches += 1
        
        val_loss /= num_batches
        val_mae /= num_batches
        
        # Update scheduler
        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'history': history
            }, '../../models/best_quantity_model_gpu.pth')
            print(f"  ✓ Saved best model (MAE: {val_mae:.4f})")
        
        print()
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"Best Validation MAE: {best_val_mae:.4f}")
    print(f"Model saved to: models/best_quantity_model_gpu.pth")
    print(f"{'='*70}")
    
    return history, best_val_mae

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*70)
    print("ADVANCED MODEL TRAINING ON GPU")
    print("="*70)
    
    # Load data
    print("\nLoading 10K dataset...")
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        data_path='../../data/raw',
        batch_size=64,  # Larger batch size for GPU
        val_split=0.1,
        test_split=0.1,
        target_size=(416, 416),
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} images")
    
    # Create model
    print("\nCreating improved ResNet-style model...")
    model = ImprovedQuantityPredictor()
    
    # Train
    history, best_mae = train_advanced_model(
        model, train_loader, val_loader, num_epochs=25
    )
    
    # Save training history
    with open('../../models/training_history_gpu.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n✓ Training history saved to: models/training_history_gpu.json")

if __name__ == "__main__":
    main()
