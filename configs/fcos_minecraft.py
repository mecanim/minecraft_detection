import torch
import torchvision
from torchvision.models.detection import fcos
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import os
from datetime import datetime

class FCOSMinecraftConfig:
    """Конфигурация для дообучения FCOS на Minecraft данных"""
    
    # Используем 91 класс как в COCO, но будем использовать только наши 17 классов
    num_classes = 91  # Должно совпадать с COCO!
    img_scale = (512, 512)
    max_epochs = 12
    batch_size = 2
    num_workers = 2
    checkpoint_interval = 1
    
    # Наши классы Minecraft мобов (маппинг на COCO индексы)
    minecraft_classes = [
        'pig', 'chicken', 'cow', 'creeper', 'sheep', 'skeleton', 
        'zombie', 'spider', 'turtle', 'wolf', 'llama', 'enderman', 
        'ghast', 'frog', 'fox', 'goat', 'bee'
    ]
    
    # Маппинг наших классов на COCO индексы (используем свободные индексы 80-90)
    class_mapping = {
        'pig': 80,
        'chicken': 81, 
        'cow': 82,
        'creeper': 83,
        'sheep': 84,
        'skeleton': 85,
        'zombie': 86,
        'spider': 87,
        'turtle': 88,
        'wolf': 89,
        'llama': 90,
        'enderman': 1,  # Можно повторно использовать некоторые COCO классы
        'ghast': 2,
        'frog': 3,
        'fox': 4,
        'goat': 5,
        'bee': 6
    }
    
    # Обратное маппирование для визуализации
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    # Цвета для визуализации
    palette = [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
        (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
        (165, 42, 42), (255, 77, 255), (0, 226, 252)
    ]
    
    # Аугментации для обучения
    train_transforms = A.Compose([
        A.Resize(height=img_scale[0], width=img_scale[1]),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels'], min_area=1, min_visibility=0.1))
    
    # Аугментации для валидации/тестирования
    val_transforms = A.Compose([
        A.Resize(height=img_scale[0], width=img_scale[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels'], min_area=1, min_visibility=0.1))
    
    # Оптимизатор
    optimizer_config = {
        'type': 'SGD',
        'lr': 0.0025,
        'momentum': 0.9,
        'weight_decay': 0.0001
    }
    
    # Learning rate scheduler
    scheduler_config = {
        'type': 'MultiStepLR',
        'milestones': [8, 11],
        'gamma': 0.1
    }
    
    # Mixed precision
    fp16 = True
    
    @classmethod
    def get_model(cls, pretrained=True):
        """Создание модели FCOS с правильным количеством классов"""
        if pretrained:
            # Загружаем с предобученными весами COCO (91 класс)
            model = torchvision.models.detection.fcos_resnet50_fpn(
                weights='DEFAULT'
            )
            print("✓ Модель FCOS загружена с предобученными весами COCO (91 класс)")
        else:
            # Создаем с нуля
            model = torchvision.models.detection.fcos_resnet50_fpn(
                weights=None,
                num_classes=cls.num_classes
            )
            print("✓ Модель FCOS создана с нуля")
        
        return model