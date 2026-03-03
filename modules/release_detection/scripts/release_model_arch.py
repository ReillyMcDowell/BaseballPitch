import torch
import torch.nn as nn
import torchvision

class PitchReleaseCSN(nn.Module):
    """Transfer learning from pretrained R3D (ResNet 3D) for pitch release detection"""
    def __init__(self, handedness_dim=1, num_classes=2, pretrained=True):
        super().__init__()
        
        # Load pretrained R3D_18 model (trained on Kinetics-400 action recognition)
        # This gives us strong temporal feature extraction out of the box
        base_model = torchvision.models.video.r3d_18(weights='DEFAULT' if pretrained else None)
        
        # Remove the final classification layer
        # R3D architecture: stem -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])  # Remove fc layer
        
        # The backbone outputs (B, 512, 1, 1, 1) after avgpool
        # We'll flatten to (B, 512)
        
        # Freeze early layers (only fine-tune later layers)
        # This prevents overfitting on small dataset
        for name, param in self.backbone.named_parameters():
            # Freeze stem, layer1, layer2 (early feature extractors)
            if any(layer in name for layer in ['stem', 'layer1', 'layer2']):
                param.requires_grad = False
            # Fine-tune layer3 and layer4 (high-level features)
            else:
                param.requires_grad = True
        
        # Fusion head with handedness
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
        # Extract features with pretrained backbone
        x = self.backbone(video)
        
        # Flatten: (B, 512, 1, 1, 1) -> (B, 512)
        video_features = x.view(x.size(0), -1)
        
        # Concatenate with handedness
        fused = torch.cat([video_features, handedness], dim=1)
        
        # Shared fusion features
        fusion_features = self.fusion_head(fused)
        
        # Predict which frame (0-15) has release
        frame_logits = self.frame_head(fusion_features)
        
        # Predict if release exists
        has_release_logits = self.has_release_head(fusion_features)
        
        return frame_logits, has_release_logits