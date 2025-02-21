import torch.nn as nn
import torch

class Darknet19Detector(nn.Module):
    """
    Darknet-19 based object detection model.
    """
    def __init__(self, num_classes=4,num_of_anchors=4):
        """
        Initializes the Darknet-19 detector.

        Args:
            num_classes (int): Number of object classes.
            num_of_anchors (int): Number of anchor boxes per grid cell.
        """
        super(Darknet19Detector, self).__init__()

        self.num_classes = num_classes
        self.num_of_anchors = num_of_anchors
        
        # Darknet-19 backbone  (same as classifier up to the last 3x3x512 layer)
        self.backbone = nn.Sequential(
            # Layer 1: Convolutional + MaxPool
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2: Convolutional + MaxPool
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3: Convolutional
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 4: Convolutional (1x1)
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 5: Convolutional + MaxPool
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 6: Convolutional
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 7: Convolutional (1x1)
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 8: Convolutional + MaxPool
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 9: Convolutional
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 10: Convolutional (1x1)
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 11: Convolutional
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 12: Convolutional (1x1)
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 13: Convolutional (the last 3x3x512 layer)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Passthrough layer (from the 3x3x512 layer)
        self.passthrough = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Stage 2: High-level features
        self.stage2_conv1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 14: Convolutional
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 15: Convolutional (1x1)
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 16: Convolutional
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 17: Convolutional (1x1)
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            
            # Layer 18: Convolutional
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Stage 2: Additional convolutional layers: Two 3x3 convolutional layers with 1024 filters
        self.stage2_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        # Stage 3: Final detection layers
        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=1024 + 256 , out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            
            nn.Conv2d(in_channels=1024, out_channels=num_of_anchors*(5 + num_classes), kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input image tensor.

        Returns:
            Tensor: Prediction output containing bounding box coordinates,
                    objectness scores, and class probabilities.
        """
        # Forward pass through the backbone
        features = self.backbone(x)
        
        # High-level features
        high_level_features = self.stage2_conv1(features)
        high_level_features = self.stage2_conv2(high_level_features)
        
        # Passthrough layer
        passthrough_features = self.passthrough(features)
        batch_size, num_channel, height, width = passthrough_features.data.size()

        # Reshape: split H and W into (H/2,2) and (W/2,2)
        passthrough_features = passthrough_features.view(batch_size, num_channel, height // 2, 2, width // 2, 2)
        # Permute to bring the 2's into the channel dimension
        passthrough_features = passthrough_features.permute(0, 1, 3, 5, 2, 4).contiguous()
        # Combine the channel with the two new dimensions: channels * 2 * 2 = channels * 4
        passthrough_features = passthrough_features.view(batch_size, num_channel * 4, height // 2, width // 2)
        
        # Concatenate passthrough and high-level features
        combined_features = torch.cat((high_level_features, passthrough_features), 1)
        
        x = self.stage3(combined_features)

        return x
