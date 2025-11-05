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

class MinimalFCOSTrainer:
    """Минимальный тренер для модели FCOS - максимально простой"""
    
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
            lr=0.01,  # Более высокий LR для быстрого обучения
            momentum=0.9,
            weight_decay=0.0001
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=8, gamma=0.1)
        
    def train_epoch(self, epoch):
        """Обучение на одной эпохе - максимально простой"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} Training')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            try:
                # Простая фильтрация - берем только те, где есть объекты
                valid_indices = []
                for i, target in enumerate(targets):
                    if isinstance(target, dict) and len(target['boxes']) > 0:
                        valid_indices.append(i)
                
                if len(valid_indices) == 0:
                    continue
                
                # Отбираем только валидные
                batch_images = [images[i] for i in valid_indices]
                batch_targets = [targets[i] for i in valid_indices]
                
                # Перемещаем на устройство
                batch_images = [img.to(self.device) for img in batch_images]
                batch_targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch_targets]
                
                self.optimizer.zero_grad()
                
                loss_dict = self.model(batch_images, batch_targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                self.optimizer.step()
                
                total_loss += losses.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': losses.item()})
                
            except Exception as e:
                # print(f"Ошибка в батче {batch_idx}: {e}")
                continue
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate(self, epoch):
        """Валидация - максимально простой"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} Validation')
            for batch_idx, (images, targets) in enumerate(pbar):
                try:
                    # Простая фильтрация
                    valid_indices = []
                    for i, target in enumerate(targets):
                        if isinstance(target, dict) and len(target['boxes']) > 0:
                            valid_indices.append(i)
                    
                    if len(valid_indices) == 0:
                        continue
                    
                    batch_images = [images[i] for i in valid_indices]
                    batch_targets = [targets[i] for i in valid_indices]
                    
                    batch_images = [img.to(self.device) for img in batch_images]
                    batch_targets = [{k: v.to(self.device) for k, v in t.items()} for t in batch_targets]
                    
                    loss_dict = self.model(batch_images, batch_targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    total_loss += losses.item()
                    num_batches += 1
                    
                except Exception as e:
                    # print(f"Ошибка в валидационном батче {batch_idx}: {e}")
                    continue
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def train(self):
        """Полный цикл обучения"""
        print("Начало обучения FCOS (минимальная версия)...")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(1, self.config.max_epochs + 1):
            print(f"\n--- Epoch {epoch}/{self.config.max_epochs} ---")
            
            # Обучение
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)
            
            # Валидация
            val_loss = self.validate(epoch)
            val_losses.append(val_loss)
            
            # Обновление learning rate
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Сохранение модели
            if epoch % 3 == 0:
                model_path = os.path.join(self.log_dir, f'fcos_epoch_{epoch}.pth')
                torch.save(self.model.state_dict(), model_path)
                print(f"Модель сохранена: {model_path}")
        
        # Сохранение финальной модели
        final_path = os.path.join(self.log_dir, 'fcos_final.pth')
        torch.save(self.model.state_dict(), final_path)
        
        # Визуализация
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Обучение завершено!")