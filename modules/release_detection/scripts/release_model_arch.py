import torch
import torch.nn as nn

class PitchReleaseCSN(nn.Module):
    """Lightweight 3D CNN for pitch release detection"""
    def __init__(self, handedness_dim=1, num_classes=2, pretrained=False):
        super().__init__()
        
        # Lightweight 3D CNN backbone (much smaller than CSN-50)
        self.conv_blocks = nn.Sequential(
            # Block 1: (B, 3, 16, 224, 224) -> (B, 64, 8, 112, 112)
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # Block 2: (B, 64, 8, 56, 56) -> (B, 128, 4, 28, 28)
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # Block 3: (B, 128, 4, 28, 28) -> (B, 256, 2, 14, 14)
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # Block 4: (B, 256, 2, 14, 14) -> (B, 512, 1, 7, 7)
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fusion head with handedness
        # Predict both: (1) which frame has release (16 outputs) and (2) confidence that release exists
        self.fusion_head = nn.Sequential(
            nn.Linear(512 + handedness_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # Frame prediction: which of 16 frames contains release point
        self.frame_head = nn.Linear(256, 16)
        
        # Binary classification: does this clip contain a release?
        self.has_release_head = nn.Linear(256, num_classes)
    
    def forward(self, video, handedness):
        """
        Args:
            video: (B, C, T, H, W) - batch of video clips
            handedness: (B, handedness_dim) - handedness features
        Returns:
            frame_logits: (B, 16) - logits for which frame contains release
            has_release_logits: (B, 2) - logits for whether release exists
        """
        # Extract features
        x = self.conv_blocks(video)
        
        # Global average pooling
        video_features = self.global_pool(x)
        video_features = video_features.view(video_features.size(0), -1)  # (B, 512)
        
        # Concatenate with handedness
        fused = torch.cat([video_features, handedness], dim=1)
        
        # Shared fusion features
        fusion_features = self.fusion_head(fused)
        
        # Predict which frame (0-15) has release
        frame_logits = self.frame_head(fusion_features)
        
        # Predict if release exists
        has_release_logits = self.has_release_head(fusion_features)
        
        return frame_logits, has_release_logits