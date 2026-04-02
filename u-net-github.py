import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from my_functions import calculate_model_complexity
from eof_k import generate_background


# ==================== Data Utilities ====================
def split_indices_by_year(n_samples: int = 18876, years: int = 13, months: int = 12, points: int = 121):
    """Split training/validation/test set indices by year"""
    assert n_samples == years * months * points, f"Number of samples {n_samples} does not match {years}*{months}*{points}"
    
    year_ids = np.repeat(np.arange(1, years + 1), months * points)
    train_idx = np.where(np.isin(year_ids, list(range(1, 8)) + [9] + list(range(11, 14))))[0]
    val_idx = np.where(year_ids == 8)[0]
    test_idx = np.where(year_ids == 10)[0]
    return train_idx, val_idx, test_idx


# ==================== Dataset ====================
class SoundFieldDataset(Dataset):
    def __init__(self, input1, input2, target, indices):
        self.input1 = input1[indices].astype(np.float32)
        self.input2 = input2[:, :, :].astype(np.float32)
        self.target = target[indices].astype(np.float32)
        
    def __len__(self):
        return len(self.input1)
    
    def __getitem__(self, idx):
        # Dimension adjustment: from [36, 4, 250] -> [4, 36, 250]
        x2 = np.transpose(self.input2[idx], (1, 0, 2))  # [4, 36, 250]
        y = np.transpose(self.target[idx], (1, 0, 2))   # [4, 36, 250]
        return torch.from_numpy(self.input1[idx]), torch.from_numpy(x2), torch.from_numpy(y)


# ==================== U-Net Baseline Model ====================
class ConditionEncoder(nn.Module):
    """Simple condition encoder"""
    def __init__(self, input_dim=52, cond_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, cond_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)


class UNetDownBlock(nn.Module):
    """U-Net downsampling block"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class UNetUpBlock(nn.Module):
    """U-Net upsampling block"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = UNetDownBlock(in_channels, out_channels, dropout)
        
    def forward(self, x, skip):
        x = self.up(x)
        # Adjust skip connection dimensions to match
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SFUNet(nn.Module):
    """Simplified U-Net baseline model"""
    def __init__(self, x1_dim=52, in_ch=4, base_ch=64, cond_dim=256, dropout=0.0):
        super().__init__()
        
        # Condition encoder
        self.cond_encoder = ConditionEncoder(x1_dim, cond_dim)
        
        # Initial convolutional layer
        self.init_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)
        
        # Downsampling path
        self.down1 = UNetDownBlock(base_ch, base_ch, dropout)
        self.down2 = UNetDownBlock(base_ch, base_ch * 2, dropout)
        self.down3 = UNetDownBlock(base_ch * 2, base_ch * 4, dropout)
        self.down4 = UNetDownBlock(base_ch * 4, base_ch * 8, dropout)
        
        # Downsampling pooling
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 16, base_ch * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 16, base_ch * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch * 16),
            nn.ReLU(inplace=True)
        )
        
        # Condition injection layer
        self.cond_proj = nn.Linear(cond_dim, base_ch * 16)
        
        # Upsampling path
        self.up4 = UNetUpBlock(base_ch * 16, base_ch * 8, dropout)
        self.up3 = UNetUpBlock(base_ch * 8, base_ch * 4, dropout)
        self.up2 = UNetUpBlock(base_ch * 4, base_ch * 2, dropout)
        self.up1 = UNetUpBlock(base_ch * 2, base_ch, dropout)
        
        # Output layer
        self.output_conv = nn.Conv2d(base_ch, in_ch, kernel_size=1)
        
    def forward(self, x1, x2):
        """
        Forward pass
        
        Args:
            x1: Condition input [batch, 52]
            x2: Background field input [batch, 4, 36, 250]
            
        Returns:
            Predicted sound field [batch, 4, 36, 250]
        """
        # 1. Condition encoding
        cond = self.cond_encoder(x1)  # [batch, cond_dim]
        
        # 2. Initial convolution
        x = self.init_conv(x2)  # [batch, base_ch, 36, 250]
        
        # 3. Downsampling path
        d1 = self.down1(x)  # [batch, base_ch, 36, 250]
        p1 = self.pool(d1)  # [batch, base_ch, 18, 125]
        
        d2 = self.down2(p1)  # [batch, base_ch*2, 18, 125]
        p2 = self.pool(d2)   # [batch, base_ch*2, 9, 62]
        
        d3 = self.down3(p2)  # [batch, base_ch*4, 9, 62]
        p3 = self.pool(d3)   # [batch, base_ch*4, 4, 31]
        
        d4 = self.down4(p3)  # [batch, base_ch*8, 4, 31]
        p4 = self.pool(d4)   # [batch, base_ch*8, 2, 15]
        
        # 4. Bottleneck
        bottleneck = self.bottleneck(p4)  # [batch, base_ch*16, 2, 15]
        
        # 5. Inject condition information
        cond_proj = self.cond_proj(cond)  # [batch, base_ch*16]
        cond_proj = cond_proj.view(cond_proj.size(0), cond_proj.size(1), 1, 1)  # [batch, base_ch*16, 1, 1]
        cond_proj = cond_proj.expand(-1, -1, 2, 15)  # [batch, base_ch*16, 2, 15]
        bottleneck = bottleneck + cond_proj  # Condition injection
        
        # 6. Upsampling path
        u4 = self.up4(bottleneck, d4)  # [batch, base_ch*8, 4, 31]
        u3 = self.up3(u4, d3)          # [batch, base_ch*4, 9, 62]
        u2 = self.up2(u3, d2)          # [batch, base_ch*2, 18, 125]
        u1 = self.up1(u2, d1)          # [batch, base_ch, 36, 250]
        
        # 7. Output
        output = self.output_conv(u1)  # [batch, 4, 36, 250]
        
        return output


# ==================== Loss Function ====================
class MixedLoss(nn.Module):
    """Mixed loss function: L1 loss + gradient loss"""
    def __init__(self, alpha=1.0, beta=0.5, split_idx=125):
        super().__init__()
        self.alpha = alpha  # L1 loss weight
        self.beta = beta    # Gradient loss weight
        self.split_idx = split_idx  # Distance split point
        
        # Create weight matrix [1, 1, 36, 250]
        weights = torch.ones(1, 1, 36, 250)
        weights[:, :, :, :split_idx] = 1  # Near field part
        weights[:, :, :, split_idx:] = 2   # Far field part
        self.weight_matrix = weights  # Not registered as parameter, just constant
        
        self.l1_loss = nn.L1Loss(reduction='none')  # Need element-wise computation
    
    def gradient_loss(self, pred, target, weights):
        """Compute gradient loss"""
        # Compute gradients of prediction and target
        pred_grad_h = pred[:, :, 1:, :] - pred[:, :, :-1, :]  # Vertical gradient
        pred_grad_w = pred[:, :, :, 1:] - pred[:, :, :, :-1]  # Horizontal gradient
        
        target_grad_h = target[:, :, 1:, :] - target[:, :, :-1, :]
        target_grad_w = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        # Vertical and horizontal weights
        weights_h = weights[:, :, 1:, :]  # [1, 1, 35, 250]
        weights_w = weights[:, :, :, 1:]  # [1, 1, 36, 249]
        
        # Gradient L1 loss
        grad_loss_h = (self.l1_loss(pred_grad_h, target_grad_h) * weights_h).mean()
        grad_loss_w = (self.l1_loss(pred_grad_w, target_grad_w) * weights_w).mean()
        
        return (grad_loss_h + grad_loss_w) / 2
    
    def forward(self, pred, target):
        weights = self.weight_matrix.to(pred.device)
        # Compute weighted L1 loss
        l1_loss_elem = self.l1_loss(pred, target)  # [batch, 4, 36, 250]
        l1 = (l1_loss_elem * weights).mean()
        
        grad = self.gradient_loss(pred, target, weights)
        
        return self.alpha * l1 + self.beta * grad


# ==================== Training Utilities ====================
@dataclass
class Metrics:
    loss: float
    l1: float
    grad: float
    rmse: float


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================== Main Training Function ====================
def main():
    print("U-Net Baseline Model Training")
    # Configuration parameters
    out_dir = "./outputs_unet_baseline"
    epochs, batch_size = 150, 32
    lr, weight_decay = 5e-4, 1e-4
    patience, seed = 20, 42
    base_channels = 32  # Base number of channels, can be adjusted
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"U-Net base channels: {base_channels}")
    
    # Load data
    input1 = np.load("shareddata2/sf_input2.npy", mmap_mode="r").astype(np.float32)
    
    target = np.load("shareddata2/sf_res2.npy", mmap_mode="r").astype(np.float32)
    assert len(input1) == len(target), "Data length mismatch"
    
    # Generate background field and normalize
    input2 = generate_background(input1, target)
    x1_mean, x1_std = input1.mean(0), input1.std(0) + 1e-6
    t_mean, t_std = target.mean(), target.std() + 1e-6
    
    input1 = (input1 - x1_mean) / x1_std
    input2 = (input2 - t_mean) / t_std
    target = (target - t_mean) / t_std
    
    # Data split
    train_idx, val_idx, test_idx = split_indices_by_year(len(input1))
    print(f"Data split: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")
    
    # Data loaders
    train_loader = DataLoader(
        SoundFieldDataset(input1, input2, target, train_idx),
        batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        SoundFieldDataset(input1, input2, target, val_idx),
        batch_size=batch_size, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        SoundFieldDataset(input1, input2, target, test_idx),
        batch_size=batch_size, shuffle=False, pin_memory=True
    )
    
    # Initialize U-Net model
    model = SFUNet(
        x1_dim=52,  # Using trimmed dimension
        in_ch=4,
        base_ch=base_channels,  # Adjustable base channels
        cond_dim=256,
        dropout=0.1
    ).to(device)
    
    # Calculate model complexity
    total_params, trainable_params = calculate_model_complexity(model)
    print(f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Loss function
    criterion = MixedLoss(alpha=1.0, beta=0.5, split_idx=125)
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode="min",          # Monitoring metric: lower is better
        factor=0.5,         # Learning rate decay factor
        patience=5,         # Number of epochs with no improvement after which lr will be reduced
        min_lr=1e-6         # Minimum learning rate
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    
    # Training loop
    os.makedirs(out_dir, exist_ok=True)
    best_val, patience_counter = float("inf"), 0
    history = {'train_loss': [], 'train_loss_l1': [], 'train_loss_grad': [], 
               'val_loss': [], 'val_loss_l1': [], 'val_loss_grad': []}
    default_weights = torch.ones(1, 1, 36, 250, device=device)
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_metrics = {'loss': 0, 'l1': 0, 'grad': 0, 'rmse': 0}
        
        for batch_idx, (x1, x2, y) in enumerate(train_loader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred = model(x1, x2)
                loss = criterion(pred, y)
                
                # Decompose losses for logging
                l1_loss = l1_criterion(pred, y)
                grad_loss = criterion.gradient_loss(pred, y, default_weights)
                mse_loss = mse_criterion(pred, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Record training metrics
            with torch.no_grad():
                train_metrics['loss'] += loss.item()
                train_metrics['l1'] += l1_loss.item()
                train_metrics['grad'] += grad_loss.item()
                train_metrics['rmse'] += mse_loss.item()
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx:3d}/{len(train_loader):3d}: "
                      f"loss={loss.item():.5f}, l1={l1_loss.item():.5f}")
        
        # Validation
        model.eval()
        val_metrics = {'loss': 0, 'l1': 0, 'grad': 0, 'rmse': 0}
        val_metrics_real = {'loss': 0, 'l1': 0, 'grad': 0, 'rmse': 0}
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    pred = model(x1, x2)    
                    # Transmission loss error in real scale
                    pred_real = pred * t_std + t_mean
                    y_real = y * t_std + t_mean
                    
                    loss = criterion(pred, y)
                    loss_real = criterion(pred_real, y_real)
                    # Decompose losses
                    l1_loss = l1_criterion(pred, y)
                    l1_loss_real = l1_criterion(pred_real, y_real)
                    grad_loss = criterion.gradient_loss(pred, y, default_weights)
                    grad_loss_real = criterion.gradient_loss(pred_real, y_real, default_weights)
                    mse_loss = mse_criterion(pred, y)
                    mse_loss_real = mse_criterion(pred_real, y_real)
                
                val_metrics['loss'] += loss.item()
                val_metrics['l1'] += l1_loss.item()
                val_metrics['grad'] += grad_loss.item()
                val_metrics['rmse'] += mse_loss.item()
                
                val_metrics_real['loss'] += loss_real.item()
                val_metrics_real['l1'] += l1_loss_real.item()
                val_metrics_real['grad'] += grad_loss_real.item()
                val_metrics_real['rmse'] += mse_loss_real.item()
        
        # Compute average metrics
        n_train = len(train_loader)
        n_val = len(val_loader)
        
        train_m = Metrics(
            loss=train_metrics['loss'] / n_train,
            l1=train_metrics['l1'] / n_train,
            grad=train_metrics['grad'] / n_train,
            rmse=np.sqrt(train_metrics['rmse'] / n_train)
        )
        
        val_m = Metrics(
            loss=val_metrics['loss'] / n_val,
            l1=val_metrics['l1'] / n_val,
            grad=val_metrics['grad'] / n_val,
            rmse=np.sqrt(val_metrics['rmse'] / n_val)
        )
        
        val_m_real = Metrics(
            loss=val_metrics_real['loss'] / n_val,
            l1=val_metrics_real['l1'] / n_val,
            grad=val_metrics_real['grad'] / n_val,
            rmse=np.sqrt(val_metrics_real['rmse'] / n_val)
        )
        
        # Update learning rate
        scheduler.step(val_m.loss)
        
        # Print training info
        print(f"\n[Epoch {epoch:03d}] lr={optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Train: loss={train_m.loss:.5f}, l1={train_m.l1:.5f}, "
              f"grad={train_m.grad:.5f}, rmse={train_m.rmse:.5f}")
        print(f"  Val:   loss={val_m.loss:.5f}, l1={val_m.l1:.5f}, "
              f"grad={val_m.grad:.5f}, rmse={val_m.rmse:.5f}")
        print(f"Validation set real-scale transmission loss: loss={val_m_real.loss:.5f}, l1={val_m_real.l1:.5f}, "
              f"grad={val_m_real.grad:.5f}, rmse={val_m_real.rmse:.5f}")
        print(f"  Time: {time.time()-t0:.1f}s")
        
        # Save history
        history['train_loss'].append(train_m.loss)
        history['train_loss_l1'].append(train_m.l1)
        history['train_loss_grad'].append(train_m.grad)
        history['val_loss'].append(val_m.loss)
        history['val_loss_l1'].append(val_m.l1)
        history['val_loss_grad'].append(val_m.grad)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val': best_val,
            'history': history
        }
        torch.save(checkpoint, os.path.join(out_dir, "last.pt"))
        
        # Save best model
        if val_m.loss < best_val - 1e-6:
            best_val = val_m.loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            print(f"✅ New best model: val_loss={best_val:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered (patience={patience})")
                break
    
    # Test best model
    print("\n" + "="*50)
    print("Testing best model...")
    print("="*50)
    
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
    model.eval()
    
    test_metrics = {'loss': 0, 'l1': 0, 'grad': 0, 'rmse': 0}
    test_metrics_real = {'loss': 0, 'l1': 0, 'rmse': 0}
    
    with torch.no_grad():
        for x1, x2, y in test_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred = model(x1, x2)
                
                # Normalized scale losses
                loss = criterion(pred, y)
                l1_loss = l1_criterion(pred, y)
                mse_loss = mse_criterion(pred, y)
                
                # Real scale losses
                pred_real = pred * t_std + t_mean
                y_real = y * t_std + t_mean
                loss_real = criterion(pred_real, y_real)
                l1_loss_real = l1_criterion(pred_real, y_real)
                mse_loss_real = mse_criterion(pred_real, y_real)
            
            test_metrics['loss'] += loss.item()
            test_metrics['l1'] += l1_loss.item()
            test_metrics['rmse'] += torch.sqrt(mse_loss).item()
            
            test_metrics_real['loss'] += loss_real.item()
            test_metrics_real['l1'] += l1_loss_real.item()
            test_metrics_real['rmse'] += torch.sqrt(mse_loss_real).item()
    
    n_test = len(test_loader)
    print(f"\n[TEST Normalized scale]")
    print(f"  Loss: {test_metrics['loss']/n_test:.5f}")
    print(f"  L1:   {test_metrics['l1']/n_test:.5f}")
    print(f"  RMSE: {test_metrics['rmse']/n_test:.5f}")
    
    print(f"\n[TEST Real transmission loss scale]")
    print(f"  Loss: {test_metrics_real['loss']/n_test:.5f}")
    print(f"  L1:   {test_metrics_real['l1']/n_test:.5f}")
    print(f"  RMSE: {test_metrics_real['rmse']/n_test:.5f}")
    
    # Save normalization parameters and history
    np.save(os.path.join(out_dir, "x1_mean.npy"), x1_mean)
    np.save(os.path.join(out_dir, "x1_std.npy"), x1_std)
    np.save(os.path.join(out_dir, "t_mean.npy"), t_mean)
    np.save(os.path.join(out_dir, "t_std.npy"), t_std)
    np.save(os.path.join(out_dir, "training_history.npy"), history)
    
    print("\nTraining complete! Model and parameters saved to:", out_dir)


if __name__ == "__main__":
    main()
