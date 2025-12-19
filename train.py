"""
训练脚本
用于训练EAKM-Net模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import os
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys
import os

# 添加当前目录到路径，以便直接运行脚本
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from .eakm_net import create_eakm_net
    from .data_preprocessing import FakeNewsDataset
    from .cfnd_dataset import CFNDDataset, create_cfnd_dataloaders
    from .knowledge_enhancement import KnowledgeBase
except ImportError:
    from eakm_net import create_eakm_net
    from data_preprocessing import FakeNewsDataset
    from cfnd_dataset import CFNDDataset, create_cfnd_dataloaders
    from knowledge_enhancement import KnowledgeBase


def collate_fn(batch):
    """自定义批处理函数"""
    texts = [item['text'] for item in batch]
    images = torch.stack([item['image'] for item in batch])
    entities = [item['entities'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    
    return texts, images, entities, labels


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (texts, images, entities, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(texts, images, entities)
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='[Val]')
        for texts, images, entities, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            logits = model(texts, images, entities)
            loss = criterion(logits, labels)
            
            # 统计
            running_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return avg_loss, accuracy, precision, recall, f1, auc


def train(
    data_path: str,
    model_save_dir: str = './checkpoints',
    batch_size: int = 4,
    num_epochs: int = 50,
    learning_rate: float = 2e-5,
    text_model: str = 'bert-base-uncased',
    image_backbone: str = 'resnet50',
    device: str = 'cuda',
    resume_from: str = None,
    dataset_type: str = 'csv',  # 'csv' 或 'cfnd'
    use_entities: bool = True
):
    """
    训练主函数
    
    Args:
        data_path: 数据路径（CSV文件路径或CFND数据集目录）
        dataset_type: 数据集类型 ('csv' 或 'cfnd')
        use_entities: 是否使用实体识别
    """
    
    # 创建保存目录
    os.makedirs(model_save_dir, exist_ok=True)
    
    # 设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建知识库
    knowledge_base = KnowledgeBase()
    
    # 创建数据集
    print(f"加载数据集 (类型: {dataset_type})...")
    
    if dataset_type.lower() == 'cfnd':
        # 使用CFND数据集
        train_loader, val_loader, _ = create_cfnd_dataloaders(
            data_dir=data_path,
            batch_size=batch_size,
            num_workers=0,
            use_entities=use_entities,
            image_size=224
        )
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
    else:
        # 使用CSV数据集（原有方式）
        dataset = FakeNewsDataset(
            data_path=data_path,
            text_col='text',
            image_col='image',
            label_col='label',
            max_length=512,
            image_size=224,
            use_entities=use_entities
        )
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    print("创建模型...")
    model = create_eakm_net(
        text_model=text_model,
        image_backbone=image_backbone,
        num_classes=2,
        knowledge_base=knowledge_base
    )
    model = model.to(device)
    
    # 加载检查点
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        print(f"从 {resume_from} 恢复训练...")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    
    # 训练循环
    best_auc = 0.0
    train_losses = []
    val_losses = []
    val_aucs = []
    
    print("\n开始训练...")
    for epoch in range(start_epoch, num_epochs):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        
        # 打印结果
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
              f"Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'val_f1': val_f1
            }
            best_model_path = os.path.join(model_save_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            print(f"保存最佳模型 (AUC: {best_auc:.4f})")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(model_save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
    
    print(f"\n训练完成！最佳AUC: {best_auc:.4f}")
    print(f"模型保存在: {os.path.join(model_save_dir, 'best_model.pth')}")


def main():
    parser = argparse.ArgumentParser(description='训练EAKM-Net模型')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='数据路径（CSV文件路径或CFND数据集目录）')
    parser.add_argument('--dataset_type', type=str, default='csv', 
                       choices=['csv', 'cfnd'], help='数据集类型 (csv/cfnd)')
    parser.add_argument('--model_save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='学习率')
    parser.add_argument('--text_model', type=str, default='bert-base-uncased', 
                       help='文本模型（CFND数据集建议使用bert-base-chinese）')
    parser.add_argument('--image_backbone', type=str, default='resnet50', help='图像骨干网络')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--resume_from', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--use_entities', action='store_true', default=True, 
                       help='是否使用实体识别')
    parser.add_argument('--no_entities', dest='use_entities', action='store_false',
                       help='禁用实体识别')
    
    args = parser.parse_args()
    
    train(
        data_path=args.data_path,
        model_save_dir=args.model_save_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        text_model=args.text_model,
        image_backbone=args.image_backbone,
        device=args.device,
        resume_from=args.resume_from,
        dataset_type=args.dataset_type,
        use_entities=args.use_entities
    )


if __name__ == "__main__":
    main()


