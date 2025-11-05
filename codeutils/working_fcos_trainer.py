import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime

class WorkingFCOSTrainer:
    """Рабочий тренер для модели FCOS"""
    
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Логирование
        self.log_dir = 'artifacts/fcos'
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Простой оптимизатор
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0001
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=8, gamma=0.1)
        
        # История лоссов
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, epoch):
        """Обучение на одной эпохе"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} Training')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            try:
                # Фильтруем батч - оставляем только с объектами
                valid_images = []
                valid_targets = []
                
                for i, (img, target) in enumerate(zip(images, targets)):
                    if isinstance(target, dict) and len(target['boxes']) > 0:
                        valid_images.append(img)
                        valid_targets.append(target)
                
                if len(valid_images) == 0:
                    continue
                
                # Перемещаем на устройство
                valid_images = [img.to(self.device) for img in valid_images]
                valid_targets = [{k: v.to(self.device) for k, v in t.items()} for t in valid_targets]
                
                self.optimizer.zero_grad()
                
                # Прямой проход
                loss_dict = self.model(valid_images, valid_targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Обратный проход
                losses.backward()
                self.optimizer.step()
                
                total_loss += losses.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': losses.item()})
                
            except Exception as e:
                # print(f"Ошибка в батче {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, epoch):
        """Валидация"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} Validation')
            for batch_idx, (images, targets) in enumerate(pbar):
                try:
                    # Фильтруем батч
                    valid_images = []
                    valid_targets = []
                    
                    for i, (img, target) in enumerate(zip(images, targets)):
                        if isinstance(target, dict) and len(target['boxes']) > 0:
                            valid_images.append(img)
                            valid_targets.append(target)
                    
                    if len(valid_images) == 0:
                        continue
                    
                    valid_images = [img.to(self.device) for img in valid_images]
                    valid_targets = [{k: v.to(self.device) for k, v in t.items()} for t in valid_targets]
                    
                    loss_dict = self.model(valid_images, valid_targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    total_loss += losses.item()
                    num_batches += 1
                    
                except Exception as e:
                    # print(f"Ошибка в валидационном батче {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self):
        """Полный цикл обучения"""
        print("Начало обучения FCOS...")
        print(f"Используется {self.config.num_classes} классов (COCO)")
        print(f"Наши классы маппятся на индексы: {self.config.class_mapping}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config.max_epochs + 1):
            print(f"\n--- Epoch {epoch}/{self.config.max_epochs} ---")
            
            # Обучение
            train_loss = self.train_epoch(epoch)
            
            # Валидация
            val_loss = self.validate(epoch)
            
            # Обновление learning rate
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Сохранение чекпоинта
            if epoch % self.config.checkpoint_interval == 0:
                model_path = os.path.join(self.log_dir, f'fcos_epoch_{epoch}.pth')
                torch.save(self.model.state_dict(), model_path)
                print(f"Модель сохранена: {model_path}")
        
        # Сохранение финальной модели
        final_path = os.path.join(self.log_dir, 'fcos_final.pth')
        torch.save(self.model.state_dict(), final_path)
        
        training_time = time.time() - start_time
        
        # Визуализация лоссов
        self.plot_losses()
        
        print(f"\nОбучение завершено за {training_time:.2f} секунд")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'training_time': training_time
        }
    
    def plot_losses(self):
        """Визуализация лоссов"""
        if len(self.train_losses) == 0:
            print("Нет данных для визуализации")
            return
            
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('FCOS Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.show()