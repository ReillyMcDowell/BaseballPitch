import torch
import torch.nn as nn
from pytorchvideo.models.csn import create_csn

class PitchReleaseCSN(nn.Module):
    def __init__(self, handedness_dim=1, num_classes=2, pretrained=False):
        super().__init__()
        
        # 1. Load CSN backbone (we'll use it for feature extraction)
        # Note: CSN-50 outputs 2048 features, CSN-152 outputs 2048 features
        csn_model = create_csn(
            input_channel=3,
            model_depth=50,
            model_num_class=400,
            norm=nn.BatchNorm3d,
            activation=nn.ReLU,
        )
        
        # Store the backbone without the final projection layer
        # We'll extract features before the classification head
        self.backbone = csn_model
        self.backbone_features = 2048
        
        # 2. Custom feature extractor wrapper
        # This will help us get features instead of classifications
        self.feature_extractor = nn.Sequential()
        for name, module in csn_model.named_children():
            # Stop before the final projection/classification layer
            if 'projection' in name.lower() or 'fc' in name.lower():
                break
            self.feature_extractor.add_module(name, module)
        
        # 3. Fusion Layer
        # Input: 2048 (Video Features) + handedness_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(self.backbone_features + handedness_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, video, handedness):
        """
        Args:
            video: (B, C, T, H, W) - batch of video clips
            handedness: (B, handedness_dim) - handedness features
        Returns:
            (B, num_classes) - classification logits
        """
        # Extract video features using the feature extractor
        video_features = self.feature_extractor(video)  # (B, C', T', H', W')
        
        # Global average pooling to get (B, 2048)
        video_features = torch.nn.functional.adaptive_avg_pool3d(video_features, (1, 1, 1))
        video_features = video_features.view(video_features.size(0), -1)  # (B, 2048)
        
        # Concatenate with handedness
        fused = torch.cat([video_features, handedness], dim=1)  # (B, 2048 + handedness_dim)
        
        # Final classification
        output = self.fusion_head(fused)  # (B, num_classes)
        
        return output