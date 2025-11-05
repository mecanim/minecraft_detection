import os
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import Counter


class MinecraftDataLoader:
    def __init__(self, datasets_path):
        self.datasets_path = datasets_path
        self.classes = set()
        self.annotations_data = []
        
    def parse_annotation(self, xml_path):
        """Парсинг XML аннотации"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        objects_data = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            self.classes.add(class_name)
            objects_data.append({
                'class': class_name,
                'bbox': [xmin, ymin, xmax, ymax],
                'bbox_normalized': [
                    xmin/width, ymin/height, 
                    xmax/width, ymax/height
                ]
            })
            
        return {
            'filename': filename,
            'width': width,
            'height': height,
            'objects': objects_data
        }
    
    def load_dataset(self, split='train'):
        """Загрузка данных для train/valid/test"""
        split_path = os.path.join(self.datasets_path, split)
        #images_path = os.path.join(split_path, 'images')
        #annotations_path = os.path.join(split_path, 'annotations')
        
        # Проверка существования директорий
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Директория {split_path} не найдена")
        # if not os.path.exists(annotations_path):
        #     raise FileNotFoundError(f"Директория {annotations_path} не найдена")
        
        # Получение списка файлов
        image_files = [f for f in os.listdir(split_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        annotation_files = [f for f in os.listdir(split_path) 
                           if f.lower().endswith('.xml')]
        
        print(f"Найдено в {split}:")
        print(f"  - Изображений: {len(image_files)}")
        print(f"  - Аннотаций: {len(annotation_files)}")
        
        # Проверка соответствия изображений и аннотаций
        image_names = {os.path.splitext(f)[0] for f in image_files}
        annotation_names = {os.path.splitext(f)[0] for f in annotation_files}
        
        missing_annotations = image_names - annotation_names
        missing_images = annotation_names - image_names
        
        if missing_annotations:
            print(f"  ⚠️ Отсутствуют аннотации для {len(missing_annotations)} изображений")
        if missing_images:
            print(f"  ⚠️ Отсутствуют изображения для {len(missing_images)} аннотаций")
        
        # Загрузка аннотаций
        split_data = []
        for ann_file in annotation_files:
            if os.path.splitext(ann_file)[0] in image_names:
                xml_path = os.path.join(split_path, ann_file)
                annotation_data = self.parse_annotation(xml_path)
                annotation_data['split'] = split
                annotation_data['image_path'] = os.path.join(
                    split_path, 
                    annotation_data['filename']
                )
                split_data.append(annotation_data)
                self.annotations_data.append(annotation_data)
        
        print(f"  ✅ Успешно загружено: {len(split_data)} пар изображение-аннотация")
        return split_data
    
    def get_class_distribution(self):
        """Получение распределения классов"""
        class_counts = Counter()
        for ann in self.annotations_data:
            for obj in ann['objects']:
                class_counts[obj['class']] += 1
        
        return class_counts
    
    def get_dataset_stats(self):
        """Получение статистики датасета"""
        total_images = len(self.annotations_data)
        total_objects = sum(len(ann['objects']) for ann in self.annotations_data)
        avg_objects_per_image = total_objects / total_images if total_images > 0 else 0
        
        # Статистика по размерам bounding boxes
        bbox_widths = []
        bbox_heights = []
        bbox_areas = []
        
        for ann in self.annotations_data:
            for obj in ann['objects']:
                xmin, ymin, xmax, ymax = obj['bbox']
                width = xmax - xmin
                height = ymax - ymin
                area = width * height
                
                bbox_widths.append(width)
                bbox_heights.append(height)
                bbox_areas.append(area)
        
        stats = {
            'total_images': total_images,
            'total_objects': total_objects,
            'avg_objects_per_image': avg_objects_per_image,
            'num_classes': len(self.classes),
            'classes': list(self.classes),
            'bbox_stats': {
                'width_mean': np.mean(bbox_widths),
                'width_std': np.std(bbox_widths),
                'height_mean': np.mean(bbox_heights),
                'height_std': np.std(bbox_heights),
                'area_mean': np.mean(bbox_areas),
                'area_std': np.std(bbox_areas),
            }
        }
        
        return stats


def visualize_sample(image_path, annotation_data, save_path=None):
    """Визуализация примера изображения с bounding boxes"""
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Изображение не найдено: {image_path}")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    
    for obj in annotation_data['objects']:
        class_name = obj['class']
        xmin, ymin, xmax, ymax = obj['bbox']
        
        # Рисование bounding box
        rect = patches.Rectangle(
            (xmin, ymin), xmax-xmin, ymax-ymin,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Добавление подписи класса
        ax.text(
            xmin, ymin-5, class_name,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
            fontsize=10, color='white', weight='bold'
        )
    
    ax.set_title(f"Пример изображения: {os.path.basename(image_path)}")
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.tight_layout()
    plt.show()


def plot_class_distribution(class_counts, save_path=None):
    """Визуализация распределения классов"""
    classes, counts = zip(*class_counts.most_common())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Добавление значений на столбцы
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.title('Распределение классов в датасете', fontsize=14, fontweight='bold')
    plt.xlabel('Классы', fontsize=12)
    plt.ylabel('Количество объектов', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()