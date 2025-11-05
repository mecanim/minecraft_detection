# mmdetection/utils/fcos_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FCOS
from torchvision.models.detection.fcos import FCOSHead
from torchvision.ops import sigmoid_focal_loss
import math

class MinecraftFCOS(nn.Module):
    """FCOS модель для детекции Minecraft мобов"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Загрузка предобученной модели FCOS
        self.model = self._build_model()
    
    def _build_model(self):
        """Построение модели FCOS"""
        # Используем implementation из torchvision
        model = torchvision.models.detection.fcos_resnet50_fpn(
            pretrained=True,
            num_classes=self.config.num_classes,
            min_size=self.config.image_size[0],
            max_size=self.config.image_size[1]
        )
        
        return model
    
    def forward(self, images, targets=None):
        return self.model(images, targets)
    
    def predict(self, images, confidence_threshold=0.3):
        """Инференс с порогом уверенности"""
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)
        
        # Фильтрация по confidence
        filtered_predictions = []
        for pred in predictions:
            keep = pred['scores'] > confidence_threshold
            filtered_pred = {
                'boxes': pred['boxes'][keep],
                'scores': pred['scores'][keep],
                'labels': pred['labels'][keep]
            }
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions

class FCOSTrainer:
    """Тренер для FCOS модели"""
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Оптимизатор
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[8, 11], gamma=0.1
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        
        self.model.to(self.device)
    
    def train_epoch(self, dataloader, epoch):
        """Обучение на одной эпохе"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.config.use_amp:
                with torch.cuda.amp.autocast():
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                
                self.scaler.scale(losses).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                self.optimizer.step()
            
            total_loss += losses.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(dataloader)} | Loss: {losses.item():.4f}')
        
        self.scheduler.step()
        return total_loss / len(dataloader)