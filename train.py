import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from videomamba import videomamba_middle  # Adjust based on model size needs
from kinetics_sparse import VideoClsDataset_sparse

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and DataLoader
    print("Loading training data...")
    train_dataset = VideoClsDataset_sparse(
        anno_path=args.train_anno,
        prefix='',
        split=' ',
        mode='train',
        clip_len=args.clip_len,
        frame_sample_rate=args.frame_sample_rate,
        crop_size=224,
        short_side_size=256,
        new_height=256,
        new_width=340,
        keep_aspect_ratio=True,
        num_segment=1,
        num_crop=1,
        args=args
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print("Loading validation data...")
    val_dataset = VideoClsDataset_sparse(
        anno_path=args.val_anno,
        prefix='',
        split=' ',
        mode='validation',
        clip_len=args.clip_len,
        frame_sample_rate=args.frame_sample_rate,
        crop_size=224,
        short_side_size=256,
        new_height=256,
        new_width=340,
        keep_aspect_ratio=True,
        num_segment=1,
        num_crop=1,
        args=args
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model
    print("Initializing model...")
    model = videomamba_middle(num_classes=2, num_frames=args.clip_len) # 2 classes: ADL, Fall
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for i, (inputs, labels, _, _) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Time: {time.time() - start_time:.2f}s")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100 * val_correct / val_total
        print(f"Epoch [{epoch+1}/{args.epochs}] Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fall-Mamba")
    parser.add_argument("--train_anno", type=str, required=True, help="Path to training annotation file")
    parser.add_argument("--val_anno", type=str, required=True, help="Path to validation annotation file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--clip_len", type=int, default=8, help="Number of frames per clip")
    parser.add_argument("--frame_sample_rate", type=int, default=1, help="Frame sample rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=1, help="Epoch interval to save checkpoint")
    # Add dummy args for compatibility with kinetics_sparse.py if needed
    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument("--reprob", type=float, default=0.0)
    parser.add_argument("--aa", type=str, default='rand-m7-n4-mstd0.5-inc1')
    parser.add_argument("--train_interpolation", type=str, default='bicubic')
    parser.add_argument("--data_set", type=str, default='Kinetics-400') 
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
