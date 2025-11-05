# mmdetection/utils/minecraft_dataset.py

import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image

class MinecraftDataset(Dataset):
    """Датасет для Minecraft мобов с маппингом на COCO классы"""
    
    def __init__(self, data_dir, transform=None, split='train', config=None):
        """
        Args:
            data_dir (str): Путь к директории с данными
            transform: Аугментации
            split (str): train/val/test
            config: Конфигурация с маппингом классов
        """
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.split = split
        self.config = config
        
        # Поиск файлов
        self.image_files = []
        self.annotation_files = []
        
        images_dir = os.path.join(self.data_dir, 'images')
        annotations_dir = os.path.join(self.data_dir, 'annotations')
        
        # Если нет поддиректорий, ищем в корне
        if not os.path.exists(images_dir):
            images_dir = self.data_dir
        if not os.path.exists(annotations_dir):
            annotations_dir = self.data_dir
            
        # Сбор пар файлов
        for file in os.listdir(images_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(file)[0]
                xml_file = base_name + '.xml'
                xml_path = os.path.join(annotations_dir, xml_file)
                
                if os.path.exists(xml_path):
                    self.image_files.append(os.path.join(images_dir, file))
                    self.annotation_files.append(xml_path)
        
        print(f"Загружено {len(self.image_files)} изображений для {split}")
        
        # Используем маппинг из конфига
        if config and hasattr(config, 'class_mapping'):
            self.class_mapping = config.class_mapping
            self.idx_to_class = config.idx_to_class
        else:
            # Маппинг по умолчанию
            self.class_mapping = {
                'pig': 80, 'chicken': 81, 'cow': 82, 'creeper': 83, 'sheep': 84,
                'skeleton': 85, 'zombie': 86, 'spider': 87, 'turtle': 88, 'wolf': 89,
                'llama': 90, 'enderman': 1, 'ghast': 2, 'frog': 3, 'fox': 4, 'goat': 5, 'bee': 6
            }
            self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Загрузка изображения
        image_path = self.image_files[idx]
        image = cv2.imread(image_path)
        if image is None:
            print(f"Ошибка загрузки изображения: {image_path}")
            # Возвращаем пустые данные
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        
        # Загрузка аннотаций
        annotation_path = self.annotation_files[idx]
        boxes, labels = self.parse_annotation(annotation_path, original_width, original_height)
        
        # Если нет объектов, создаем пустые тензоры
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64) if len(boxes) > 0 else torch.zeros((0,), dtype=torch.int64)
        }
        
        # Применение аугментаций
        if self.transform:
            try:
                # Конвертируем boxes в формат albumentations [x_min, y_min, x_max, y_max]
                boxes_list = boxes.tolist() if len(boxes) > 0 else []
                labels_list = labels.tolist() if len(labels) > 0 else []
                
                transformed = self.transform(
                    image=image,
                    bboxes=boxes_list,
                    labels=labels_list
                )
                image = transformed['image']
                
                # Обновляем targets после аугментаций
                if len(transformed['bboxes']) > 0:
                    target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
                    target['labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)
                    target['area'] = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
                    target['iscrowd'] = torch.zeros((len(target['boxes']),), dtype=torch.int64)
                else:
                    # Если после аугментаций не осталось объектов
                    target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                    target['labels'] = torch.zeros((0,), dtype=torch.int64)
                    target['area'] = torch.zeros((0,), dtype=torch.float32)
                    target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)
                    
            except Exception as e:
                print(f"Ошибка при аугментации изображения {image_path}: {e}")
                # В случае ошибки используем оригинальное изображение без аугментаций
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, target
    
    def parse_annotation(self, xml_path, img_width, img_height):
        """Парсинг XML аннотации с маппингом классов на COCO индексы"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in self.class_mapping:
                print(f"Неизвестный класс '{class_name}' в файле {xml_path}, пропускаем")
                continue
                
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            # Нормализация координат к [0, 1]
            xmin_norm = xmin / img_width
            ymin_norm = ymin / img_height
            xmax_norm = xmax / img_width
            ymax_norm = ymax / img_height
            
            # Проверка и коррекция границ
            xmin_norm = max(0.0, min(1.0, xmin_norm))
            ymin_norm = max(0.0, min(1.0, ymin_norm))
            xmax_norm = max(0.0, min(1.0, xmax_norm))
            ymax_norm = max(0.0, min(1.0, ymax_norm))
            
            # Проверка валидности bounding box
            if xmax_norm > xmin_norm and ymax_norm > ymin_norm:
                boxes.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])
                labels.append(self.class_mapping[class_name])
        
        return boxes, labels

def collate_fn(batch):
    """Collate function для DataLoader"""
    return tuple(zip(*batch))