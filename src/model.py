import torch
import torch.nn as nn
import torchvision.models as models

class PlantDiseaseModel(nn.Module):
    """
    Модель для классификации болезней растений на основе ResNet18
    """
    def __init__(self, num_classes=38, pretrained=True):
        super(PlantDiseaseModel, self).__init__()
        
        # Загружаем предобученный ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Получаем размер признаков от backbone
        num_features = self.backbone.fc.in_features
        
        # Заменяем последний слой на новый классификатор
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def get_model(num_classes, device):
    """Утилита для создания модели"""
    model = PlantDiseaseModel(num_classes=num_classes)
    model = model.to(device)
    return model