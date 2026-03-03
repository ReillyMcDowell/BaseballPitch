import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from torch.utils.data import Dataset, DataLoader


class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=16,
                 resize=(112, 112)):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Y'know how a lot of AI Images have that yellow tint?
            # We're normalizing to prevent that and speed up training
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_paths[idx])
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate stride to sample exactly 'num_frames' across
        # the video
        stride = max(1, total_frames // self.num_frames)
        for i in range(self.num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * stride)
            ret, frame = cap.read()
            if not ret:
                # If video ends early, pad with a blank frame
                frame = torch.zeros((3, *self.resize))
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.resize)
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        # Stack frames into [Channels, Frames, H, W]
        video_tensor = torch.stack(frames, dim=1)
        return video_tensor, self.labels[idx]


def get_paths_and_labels(root_dir):
    video_paths = []
    labels = []
    class_to_idx = {
        cls: i for i, cls in enumerate(sorted(os.listdir(root_dir)))
    }
    for cls in class_to_idx:
        cls_dir = os.path.join(root_dir, cls)
        for vid in os.listdir(cls_dir):
            video_paths.append(os.path.join(cls_dir, vid))
            labels.append(class_to_idx[cls])
    return video_paths, labels, class_to_idx


def build_model(num_classes=18):
    """Load pretrained R(2+1)D, freeze backbone, unfreeze layer4,
    replace fc head."""
    weights = R2Plus1D_18_Weights.DEFAULT
    model = r2plus1d_18(weights=weights)

    # Freeze ALL layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last residual block so it can adapt to
    # pitch-specific motion patterns
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace classification head (400 -> num_classes)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(  # type: ignore[assignment]
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    return model


def train_model(model, train_loader, val_loader, optimizer, criterion,
                device, num_epochs=30, patience=5,
                save_path='modules/naive_classification/best_pitch_model.pt'):
    """
    Full training loop with validation, logging, early stopping,
    and model checkpointing.
    """
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # ---- Training Phase ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}"
                      f" - Loss: {loss.item():.4f}")

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- Validation Phase ----
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(videos)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * videos.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}  "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  "
              f"Val Acc: {val_acc:.4f}")

        # ---- Checkpointing ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved! "
                  f"(Val Acc: {val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement for "
                  f"{epochs_without_improvement}/{patience} epochs")

        # ---- Early Stopping ----
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {epoch + 1} epochs.")
            break

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")
    print(f"Best model saved to: {save_path}")
    return best_val_acc


if __name__ == '__main__':
    # ---- GPU Setup ----
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU required.")
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # ---- Model ----
    model = build_model(num_classes=18)
    model = model.to(device)

    # Verify model is on GPU
    print(f"Model device: {next(model.parameters()).device}")

    # ---- Optimizer & Loss ----
    optimizer = torch.optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 1e-3},
    ])
    criterion = nn.CrossEntropyLoss().to(device)

    # ---- Data ----
    train_paths, train_labels, class_map = get_paths_and_labels(
        'modules/naive_classification/train'
    )
    val_paths, val_labels, _ = get_paths_and_labels(
        'modules/naive_classification/val'
    )

    train_dataset = VideoDataset(train_paths, train_labels)
    val_dataset = VideoDataset(val_paths, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ---- Train ----
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=30,
        patience=5
    )
