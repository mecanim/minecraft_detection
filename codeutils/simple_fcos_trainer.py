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

class SimpleFCOSTrainer:
    """Упрощенный тренер для модели FCOS"""
    
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Логирование
        self.log_dir = 'artifacts/fcos'
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, 'log.json')
        self.log_data = {
            'training_start': datetime.now().isoformat(),
            'config': {
                'max_epochs': config.max_epochs,
                'batch_size': config.batch_size,
                'img_scale': config.img_scale,
                'classes': config.classes,
                'num_classes': config.num_classes
            },
            'epochs': []
        }
        
        # Простой оптимизатор
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.optimizer_config['lr'],
            momentum=config.optimizer_config['momentum'],
            weight_decay=config.optimizer_config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=config.scheduler_config['milestones'],
            gamma=config.scheduler_config['gamma']
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 else None
        
    def prepare_batch(self, images, targets):
        """Подготовка батча для обучения - исправленная версия"""
        valid_images = []
        valid_targets = []
        
        for i, (image, target) in enumerate(zip(images, targets)):
            # Проверяем что target - это словарь и есть bounding boxes
            if isinstance(target, dict) and len(target['boxes']) > 0:
                valid_images.append(image)
                valid_targets.append(target)
        
        return valid_images, valid_targets
    
    def train_epoch(self, epoch):
        """Обучение на одной эпохе"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        epoch_losses = {
            'loss_classifier': 0,
            'loss_box_reg': 0,
            'loss_centerness': 0,
            'total_loss': 0
        }
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} Training')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            try:
                # Подготавливаем батч
                valid_images, valid_targets = self.prepare_batch(images, targets)
                
                # Пропускаем если нет валидных данных
                if len(valid_images) == 0:
                    # print(f"Пропущен пустой батч {batch_idx}")
                    continue
                    
                # Перемещаем на устройство
                valid_images = [image.to(self.device) for image in valid_images]
                valid_targets = [{k: v.to(self.device) for k, v in t.items()} for t in valid_targets]
                    
                self.optimizer.zero_grad()
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        loss_dict = self.model(valid_images, valid_targets)
                        losses = sum(loss for loss in loss_dict.values())
                    
                    self.scaler.scale(losses).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss_dict = self.model(valid_images, valid_targets)
                    losses = sum(loss for loss in loss_dict.values())
                    losses.backward()
                    self.optimizer.step()
                
                # Сбор статистики
                total_loss += losses.item()
                num_batches += 1
                
                for key in loss_dict:
                    if key in epoch_losses:
                        epoch_losses[key] += loss_dict[key].item()
                epoch_losses['total_loss'] += losses.item()
                
                pbar.set_postfix({'loss': losses.item()})
                
            except Exception as e:
                print(f"Ошибка в тренировочном батче {batch_idx}: {e}")
                continue
        
        if num_batches > 0:
            # Усреднение лоссов
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
        else:
            epoch_losses = {key: 0 for key in epoch_losses}
            print(f"Эпоха {epoch}: нет валидных батчей для обучения")
        
        return epoch_losses
    
    def validate(self, epoch):
        """Валидация - исправленная версия"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        val_losses = {
            'loss_classifier': 0,
            'loss_box_reg': 0,
            'loss_centerness': 0,
            'total_loss': 0
        }
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} Validation')
            for batch_idx, (images, targets) in enumerate(pbar):
                try:
                    # Подготавливаем батч
                    valid_images = []
                    valid_targets = []
                    
                    for i, (image, target) in enumerate(zip(images, targets)):
                        # Проверяем что target - это словарь и есть bounding boxes
                        if isinstance(target, dict) and len(target['boxes']) > 0:
                            valid_images.append(image)
                            valid_targets.append(target)
                    
                    # Пропускаем если нет валидных данных
                    if len(valid_images) == 0:
                        # print(f"Пропущен пустой валидационный батч {batch_idx}")
                        continue
                        
                    # Перемещаем на устройство
                    valid_images = [image.to(self.device) for image in valid_images]
                    valid_targets = [{k: v.to(self.device) for k, v in t.items()} for t in valid_targets]
                    
                    if self.scaler:
                        with torch.cuda.amp.autocast():
                            loss_dict = self.model(valid_images, valid_targets)
                            losses = sum(loss for loss in loss_dict.values())
                    else:
                        loss_dict = self.model(valid_images, valid_targets)
                        losses = sum(loss for loss in loss_dict.values())
                    
                    total_loss += losses.item()
                    num_batches += 1
                    
                    for key in loss_dict:
                        if key in val_losses:
                            val_losses[key] += loss_dict[key].item()
                    val_losses['total_loss'] += losses.item()
                    
                except Exception as e:
                    print(f"Ошибка в валидационном батче {batch_idx}: {e}")
                    # Добавим отладочную информацию
                    print(f"  Тип images: {type(images)}")
                    print(f"  Тип targets: {type(targets)}")
                    if isinstance(targets, list) and len(targets) > 0:
                        print(f"  Тип первого target: {type(targets[0])}")
                        if isinstance(targets[0], dict):
                            print(f"  Ключи первого target: {targets[0].keys()}")
                    continue
        
        if num_batches > 0:
            # Усреднение лоссов
            for key in val_losses:
                val_losses[key] /= num_batches
        else:
            val_losses = {key: 0 for key in val_losses}
            print(f"Эпоха {epoch}: нет валидных батчей для валидации")
        
        return val_losses
    
    def save_logs(self):
        """Сохранение логов в JSON"""
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def train(self):
        """Полный цикл обучения"""
        print("Начало обучения FCOS...")
        print(f"Количество классов: {self.config.num_classes}")
        print(f"Размер изображения: {self.config.img_scale}")
        print(f"Количество эпох: {self.config.max_epochs}")
        
        start_time = time.time()
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config.max_epochs + 1):
            print(f"\n--- Epoch {epoch}/{self.config.max_epochs} ---")
            
            # Обучение
            train_losses = self.train_epoch(epoch)
            
            # Валидация
            val_losses = self.validate(epoch)
            
            # Обновление learning rate
            self.scheduler.step()
            
            # Логирование
            epoch_log = {
                'epoch': epoch,
                'lr': self.scheduler.get_last_lr()[0],
                'train_losses': train_losses,
                'val_losses': val_losses,
                'timestamp': datetime.now().isoformat()
            }
            self.log_data['epochs'].append(epoch_log)
            
            print(f"Train Loss: {train_losses['total_loss']:.4f}")
            print(f"Val Loss: {val_losses['total_loss']:.4f}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Сохранение чекпоинта
            if epoch % self.config.checkpoint_interval == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_losses['total_loss'],
                    'val_loss': val_losses['total_loss']
                }
                
                checkpoint_path = os.path.join(self.log_dir, f'fcos_epoch_{epoch}.pth')
                torch.save(checkpoint, checkpoint_path)
                print(f"Чекпоинт сохранен: {checkpoint_path}")
            
            # Сохранение лучшей модели
            if val_losses['total_loss'] < best_val_loss and val_losses['total_loss'] > 0:
                best_val_loss = val_losses['total_loss']
                best_model_path = os.path.join(self.log_dir, 'fcos_best.pth')
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Лучшая модель сохранена: {best_model_path} (loss: {best_val_loss:.4f})")
            
            # Сохранение логов
            self.save_logs()
        
        # Сохранение финальной модели
        final_model_path = os.path.join(self.log_dir, 'fcos_final.pth')
        torch.save(self.model.state_dict(), final_model_path)
        
        training_time = time.time() - start_time
        self.log_data['training_end'] = datetime.now().isoformat()
        self.log_data['total_training_time'] = training_time
        self.save_logs()
        
        print(f"\nОбучение завершено за {training_time:.2f} секунд")
        print(f"Логи сохранены: {self.log_file}")
        
        # Визуализация лоссов
        self.plot_losses()
        
        return self.log_data
    
    def plot_losses(self):
        """Визуализация лоссов"""
        if len(self.log_data['epochs']) == 0:
            print("Нет данных для визуализации")
            return
            
        epochs = [log['epoch'] for log in self.log_data['epochs']]
        train_losses = [log['train_losses']['total_loss'] for log in self.log_data['epochs']]
        val_losses = [log['val_losses']['total_loss'] for log in self.log_data['epochs']]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Train Total Loss')
        plt.plot(epochs, val_losses, 'r-', label='Val Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Loss')
        plt.legend()
        plt.grid(True)
        
        # Детальные лоссы
        plt.subplot(2, 2, 2)
        train_classifier = [log['train_losses']['loss_classifier'] for log in self.log_data['epochs']]
        train_box_reg = [log['train_losses']['loss_box_reg'] for log in self.log_data['epochs']]
        train_centerness = [log['train_losses']['loss_centerness'] for log in self.log_data['epochs']]
        
        plt.plot(epochs, train_classifier, 'b-', label='Classifier Loss')
        plt.plot(epochs, train_box_reg, 'r-', label='Box Reg Loss')
        plt.plot(epochs, train_centerness, 'g-', label='Centerness Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Components')
        plt.legend()
        plt.grid(True)
        
        # Learning rate
        plt.subplot(2, 2, 3)
        lrs = [log['lr'] for log in self.log_data['epochs']]
        plt.plot(epochs, lrs, 'purple')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        
        # Validation losses
        plt.subplot(2, 2, 4)
        val_classifier = [log['val_losses']['loss_classifier'] for log in self.log_data['epochs']]
        val_box_reg = [log['val_losses']['loss_box_reg'] for log in self.log_data['epochs']]
        val_centerness = [log['val_losses']['loss_centerness'] for log in self.log_data['epochs']]
        
        plt.plot(epochs, val_classifier, 'b-', label='Classifier Loss')
        plt.plot(epochs, val_box_reg, 'r-', label='Box Reg Loss')
        plt.plot(epochs, val_centerness, 'g-', label='Centerness Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss Components')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()