import os
import yaml
from ultralytics import YOLO
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import cv2

class YOLOTrainer:
    """Тренер для модели YOLO"""
    
    def __init__(self, model_type='yolov8s', data_yaml='datasets/minecraft/data_coco.yaml'):
        self.model_type = model_type
        self.data_yaml = data_yaml
        self.log_dir = 'artifacts/yolo'
        os.makedirs(self.log_dir, exist_ok=True)
        
    def prepare_data_yaml(self):
        """Подготовка YAML файла для данных"""
        data_config = {
            'path': os.path.abspath('datasets/minecraft'),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'names': {
                0: 'pig', 1: 'chicken', 2: 'cow', 3: 'creeper', 4: 'sheep',
                5: 'skeleton', 6: 'zombie', 7: 'spider', 8: 'turtle', 9: 'wolf',
                10: 'llama', 11: 'enderman', 12: 'ghast', 13: 'frog', 14: 'fox',
                15: 'goat', 16: 'bee'
            },
            'nc': 17
        }
        
        # Сохранение YAML файла
        yaml_path = os.path.join(self.log_dir, 'data_coco.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        return yaml_path
    
    def train(self, epochs=50, imgsz=512, batch=16, patience=10):
        """Обучение YOLO модели"""
        print("Начало обучения YOLO...")
        
        # Подготовка данных
        data_yaml = self.prepare_data_yaml()
        
        # Загрузка модели
        model = YOLO(f'{self.model_type}.pt')
        
        # Параметры обучения
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'patience': patience,
            'save': True,
            'exist_ok': True,
            'project': self.log_dir,
            'name': 'train',
            'verbose': True,
            # Аугментации
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 2.0,
            'perspective': 0.0001,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }
        
        # Запуск обучения
        results = model.train(**train_args)
        
        # Копирование результатов
        self.copy_training_results()
        
        return results
    
    def copy_training_results(self):
        """Копирование результатов обучения"""
        # YOLO сохраняет результаты в runs/detect/train
        source_dir = 'runs/detect/train'
        if os.path.exists(source_dir):
            # Копирование весов
            weights_src = os.path.join(source_dir, 'weights')
            if os.path.exists(weights_src):
                weights_dst = os.path.join(self.log_dir, 'weights')
                if os.path.exists(weights_dst):
                    shutil.rmtree(weights_dst)
                shutil.copytree(weights_src, weights_dst)
            
            # Копирование результатов
            results_csv = os.path.join(source_dir, 'results.csv')
            if os.path.exists(results_csv):
                shutil.copy2(results_csv, os.path.join(self.log_dir, 'results.csv'))
    
    def visualize_metrics(self):
        """Визуализация метрик YOLO"""
        results_csv = os.path.join(self.log_dir, 'results.csv')
        
        if not os.path.exists(results_csv):
            print("Файл results.csv не найден")
            return
        
        # Загрузка данных
        df = pd.read_csv(results_csv)
        
        # Визуализация метрик
        plt.figure(figsize=(15, 10))
        
        # Box loss
        plt.subplot(2, 3, 1)
        plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
        plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Box Loss')
        plt.title('Box Loss')
        plt.legend()
        plt.grid(True)
        
        # Objectness loss
        plt.subplot(2, 3, 2)
        plt.plot(df['epoch'], df['train/obj_loss'], label='Train Obj Loss')
        plt.plot(df['epoch'], df['val/obj_loss'], label='Val Obj Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Objectness Loss')
        plt.title('Objectness Loss')
        plt.legend()
        plt.grid(True)
        
        # Classification loss
        plt.subplot(2, 3, 3)
        plt.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss')
        plt.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Classification Loss')
        plt.title('Classification Loss')
        plt.legend()
        plt.grid(True)
        
        # Precision
        plt.subplot(2, 3, 4)
        plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.title('Precision')
        plt.grid(True)
        
        # Recall
        plt.subplot(2, 3, 5)
        plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.title('Recall')
        plt.grid(True)
        
        # mAP
        plt.subplot(2, 3, 6)
        plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
        plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.title('mAP Metrics')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'yolo_training_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Вывод финальных метрик
        if not df.empty:
            final_metrics = df.iloc[-1]
            print("\nФинальные метрики YOLO:")
            print(f"Precision: {final_metrics['metrics/precision(B)']:.4f}")
            print(f"Recall: {final_metrics['metrics/recall(B)']:.4f}")
            print(f"mAP@0.5: {final_metrics['metrics/mAP50(B)']:.4f}")
            print(f"mAP@0.5:0.95: {final_metrics['metrics/mAP50-95(B)']:.4f}")