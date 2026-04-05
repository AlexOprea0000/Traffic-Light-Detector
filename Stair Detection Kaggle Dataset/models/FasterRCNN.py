import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Backbone(nn.Module):
    def __init__(self, pretrained=True, trainable_layers=3):
        super(Backbone, self).__init__()
        
        # Load ResNet50 with pre-trained weights
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        
        # Extract layers up to the final convolutional block
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Optionally freeze some layers for transfer learning
        layers_to_freeze = len(list(self.feature_extractor.children())) - trainable_layers
        for i, layer in enumerate(self.feature_extractor.children()):
            if i < layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.feature_extractor(x)



class CustomBackbone(Backbone):
    def __init__(self, pretrained=True, trainable_layers=3, num_channels=512):
        super(CustomBackbone, self).__init__(pretrained, trainable_layers)
        
        # Add a 1x1 convolutional layer to reduce the output channels
        self.conv1x1 = nn.Conv2d(2048, num_channels, kernel_size=1)
    
    def forward(self, x):
        features = super(CustomBackbone, self).forward(x)
        return self.conv1x1(features)


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DetectionHead, self).__init__()

        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Bounding box regression head
        self.shared = nn.Sequential(
              nn.Conv2d(in_channels, 256, 3, padding=1),
              nn.ReLU()
                       )
        # self.bbox_head = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(in_channels, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 4)  # Output: [x_min, y_min, x_max, y_max]
        # )
        self.bbox_head = nn.Conv2d(256, 4, 1)
        
        # Class prediction head
        # self.class_head = nn.Sequential(
        #         nn.Conv2d(in_channels, 256, 3, padding=1),
        #         nn.ReLU(),
        #         nn.Conv2d(256, 4, 1)
        #                       )
        self.class_head = nn.Conv2d(256, num_classes, 1)

    
    def forward(self, x):
        # avgpool the feature map to get a fixed-size representation
        #x=self.avgpool(x)
        # Bounding box regression
        x=self.shared(x)
        bbox_predictions = self.bbox_head(x)
        
        # Class predictions
        class_logits = self.class_head(x)
        
        return bbox_predictions, class_logits


class TrafficLightColourDetector(nn.Module):
    def __init__(self, backbone, num_classes):
        super(TrafficLightColourDetector, self).__init__()
        self.backbone = backbone
        self.detection_head = DetectionHead(in_channels=2048, num_classes=num_classes)
    
    def forward(self, x):
       
        
        # Extract features using the backbone
        features = self.backbone(x)
        
        # Forward pass through the detection head
        bbox_predictions, class_logits = self.detection_head(features)
        
        return bbox_predictions, class_logits