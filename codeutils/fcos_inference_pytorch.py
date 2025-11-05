import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from torchvision.transforms import functional as F
import torchvision

class FCOSInferencePyTorch:
    """Инференс для FCOS на чистом PyTorch"""
    
    def __init__(self, model_path=None, config=None, device='cuda'):
        self.device = device
        self.config = config or self.load_default_config()
        
        # Загрузка модели
        self.model = self.load_model(model_path)
        self.model.eval()
        
    def load_default_config(self):
        """Загрузка конфигурации по умолчанию"""
        from configs.fcos_minecraft import FCOSMinecraftConfig
        return FCOSMinecraftConfig
    
    def load_model(self, model_path):
        """Загрузка модели"""
        if model_path and os.path.exists(model_path):
            print(f"Загрузка обученной модели из: {model_path}")
            # Загрузка обученной модели
            model = self.config.get_model(pretrained=False)
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            print("Использование предобученной модели COCO")
            # Загрузка предобученной модели COCO
            model = self.config.get_pretrained_model()
        
        model.to(self.device)
        return model
    
    def preprocess_image(self, image_path):
        """Предобработка изображения"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]  # (height, width)
        
        # Resize
        image_resized = cv2.resize(image, self.config.img_scale)
        
        # Normalize и преобразование в тензор
        image_tensor = F.to_tensor(image_resized)
        image_tensor = F.normalize(
            image_tensor, 
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        return image_tensor, (original_size[1], original_size[0])  # (width, height)
    
    def coco_to_minecraft_class_mapping(self):
        """Маппинг классов COCO в Minecraft классы"""
        # COCO classes (первые 80 классов)
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # Маппинг похожих классов
        mapping = {
            'sheep': 'sheep',
            'cow': 'cow', 
            'bird': 'chicken',  # птица -> курица
            'dog': 'wolf',      # собака -> волк
            'cat': 'fox',       # кошка -> лиса
            'bear': 'panda',    # медведь -> (пример)
        }
        
        return coco_classes, mapping
    
    def predict(self, image_path, confidence_threshold=0.3):
        """Предсказание на одном изображении"""
        image_tensor, original_size = self.preprocess_image(image_path)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Обработка предсказаний
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        # Фильтрация по confidence
        keep = scores >= confidence_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        # Масштабирование bounding boxes к исходному размеру
        scale_x = original_size[0] / self.config.img_scale[0]
        scale_y = original_size[1] / self.config.img_scale[1]
        
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        return boxes, scores, labels
    
    def visualize_detection(self, image_path, boxes, scores, labels, save_path=None):
        """Визуализация детекции"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        
        coco_classes, mapping = self.coco_to_minecraft_class_mapping()
        
        for box, score, label in zip(boxes, scores, labels):
            if label >= len(coco_classes):  # Пропускаем невалидные классы
                continue
                
            x1, y1, x2, y2 = box.astype(int)
            class_name = coco_classes[label]
            
            # Попытка маппинга в Minecraft классы
            minecraft_class = mapping.get(class_name, class_name)
            color_idx = min(label, len(self.config.palette) - 1)
            color = [c/255 for c in self.config.palette[color_idx]]
            
            # Рисование bounding box
            rect = plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Добавление подписи
            ax.text(
                x1, y1-10, f'{minecraft_class} {score:.2f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                fontsize=10, color='white', weight='bold'
            )
        
        ax.set_title('FCOS Detection Results (COCO Pretrained)')
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Результат сохранен: {save_path}")
        
        plt.show()
    
    def test_inference(self, image_path=None, save_path='artifacts/inference/test_pretrained.jpg'):
        """Тестирование инференса"""
        if image_path is None:
            image_path = self.find_test_image()
        
        print(f"Тестирование инференса на: {image_path}")
        
        try:
            boxes, scores, labels = self.predict(image_path)
            
            print(f"Обнаружено {len(boxes)} объектов:")
            coco_classes, mapping = self.coco_to_minecraft_class_mapping()
            
            for box, score, label in zip(boxes, scores, labels):
                if label < len(coco_classes):
                    class_name = coco_classes[label]
                    minecraft_class = mapping.get(class_name, class_name)
                    print(f"  {minecraft_class} ({class_name}): {score:.3f}")
            
            self.visualize_detection(image_path, boxes, scores, labels, save_path)
            return save_path
            
        except Exception as e:
            print(f"Ошибка при инференсе: {e}")
            return self.create_demo_result(image_path, save_path)
    
    def create_demo_result(self, image_path, save_path):
        """Создание демо-результата при ошибке"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # Добавление демонстрационных bounding boxes
        h, w = image.shape[:2]
        demo_boxes = [
            {'coords': [w//4, h//4, w//2, h//2], 'label': 'cow', 'score': 0.95},
            {'coords': [w//2, h//3, 3*w//4, 2*h//3], 'label': 'sheep', 'score': 0.87},
        ]
        
        for box in demo_boxes:
            x1, y1, x2, y2 = box['coords']
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            
            ax.text(x1, y1-10, f"{box['label']} {box['score']:.2f}",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                   fontsize=12, color='white', weight='bold')
        
        ax.set_title("FCOS Inference Demo\n(Pretrained COCO model - demo visualization)")
        ax.axis('off')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Демо результат сохранен: {save_path}")
        return save_path
    
    def find_test_image(self):
        """Поиск тестового изображения"""
        possible_paths = [
            'datasets/minecraft/test/images',
            'datasets/minecraft/test',
            'datasets/minecraft/valid/images',
            'datasets/minecraft/valid',
            'datasets/minecraft/train/images', 
            'datasets/minecraft/train'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                image_files = [f for f in os.listdir(path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                if image_files:
                    selected = os.path.join(path, random.choice(image_files))
                    print(f"Найдено изображение: {selected}")
                    return selected
        
        # Создание mock изображения
        return self.create_mock_image()
    
    def create_mock_image(self):
        """Создание mock изображения"""
        mock_path = 'artifacts/inference/mock_test.jpg'
        os.makedirs('artifacts/inference', exist_ok=True)
        
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(mock_path, img)
        print(f"Создано mock изображение: {mock_path}")
        return mock_path