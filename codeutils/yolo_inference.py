# mmdetection/utils/yolo_inference.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from ultralytics import YOLO

class YOLOInferenceTester:
    def __init__(self, model_type='yolov8s'):
        """
        Инициализация тестера YOLO инференса
        
        Args:
            model_type (str): Тип YOLO модели
        """
        self.model_type = model_type
        self.model = None
        
    def load_model(self):
        """Загрузка YOLO модели"""
        try:
            self.model = YOLO(f'{self.model_type}.pt')
            print(f"YOLO модель {self.model_type} загружена")
        except Exception as e:
            print(f"Ошибка загрузки YOLO модели: {e}")
            self.model = None
    
    def find_test_image(self):
        """Поиск тестового изображения"""
        possible_paths = [
            'datasets/minecraft/test/images',
            'datasets/minecraft/test',
            'datasets/minecraft/valid/images',
            'datasets/minecraft/valid'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                image_files = [f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                if image_files:
                    return os.path.join(path, random.choice(image_files))
        
        # Если изображения не найдены, создаем mock изображение
        return self.create_mock_image()
    
    def create_mock_image(self):
        """Создание mock изображения для тестирования"""
        mock_path = 'artifacts/inference/mock_yolo_test.jpg'
        os.makedirs('artifacts/inference', exist_ok=True)
        
        # Создание простого изображения
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(mock_path, img)
        print(f"Создано mock изображение для YOLO: {mock_path}")
        return mock_path
    
    def test_inference(self, save_dir='artifacts/inference/yolo_val/'):
        """
        Тестирование инференса YOLO
        
        Args:
            save_dir (str): Директория для сохранения результатов
            
        Returns:
            str: Путь к директории с результатами
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if self.model is None:
            self.load_model()
        
        image_path = self.find_test_image()
        print(f"Тестовое изображение для YOLO: {image_path}")
        
        if self.model is None:
            return self.simple_inference(image_path, save_dir)
        else:
            return self.full_inference(image_path, save_dir)
    
    def simple_inference(self, image_path, save_dir):
        """Упрощенный инференс для демонстрации"""
        # Загрузка изображения
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Создание фиктивных результатов
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # Добавление демонстрационных bounding boxes
        h, w = image.shape[:2]
        demo_boxes = [
            {'coords': [w//4, h//4, w//2, h//2], 'label': 'sheep', 'score': 0.92},
            {'coords': [w//2, h//3, 3*w//4, 2*h//3], 'label': 'creeper', 'score': 0.88},
        ]
        
        for box in demo_boxes:
            x1, y1, x2, y2 = box['coords']
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
            
            ax.text(x1, y1-10, f"{box['label']} {box['score']:.2f}",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.7),
                   fontsize=12, color='white', weight='bold')
        
        ax.set_title("YOLOv8 Inference Demo (Pretrained Model)\nNote: Using demo boxes - real inference requires YOLO installation")
        ax.axis('off')
        
        save_path = os.path.join(save_dir, 'yolo_demo_result.jpg')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Демо результат YOLO сохранен: {save_path}")
        return save_dir
    
    def full_inference(self, image_path, save_dir):
        """Полный инференс с YOLO"""
        try:
            # Выполнение инференса
            results = self.model(image_path)
            
            # Сохранение результатов
            for i, result in enumerate(results):
                result.save(filename=os.path.join(save_dir, f'yolo_result_{i}.jpg'))
            
            print(f"Результаты YOLO сохранены в: {save_dir}")
            return save_dir
            
        except Exception as e:
            print(f"Ошибка при YOLO инференсе: {e}")
            return self.simple_inference(image_path, save_dir)