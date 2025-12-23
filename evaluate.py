"""
评估脚本
用于评估训练好的EAKM-Net模型
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# 设置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

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


def evaluate(
    model_path: str,
    data_path: str,
    batch_size: int = 4,
    text_model: str = 'bert-base-uncased',
    image_backbone: str = 'resnet50',
    device: str = 'cuda',
    save_results: bool = True,
    output_dir: str = './results'
):
    """评估模型"""
    
    # 设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    knowledge_base = KnowledgeBase()
    model = create_eakm_net(
        text_model=text_model,
        image_backbone=image_backbone,
        num_classes=2,
        knowledge_base=knowledge_base
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型检查点信息:")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Val AUC: {checkpoint.get('val_auc', 'N/A'):.4f}")
    print(f"  - Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f}")
    
    # 数据加载
    print("加载数据...")
    # 判断是CFND数据集还是CSV文件
    if os.path.isdir(data_path) and os.path.exists(os.path.join(data_path, 'test_data.csv')):
        # CFND数据集
        _, _, test_loader = create_cfnd_dataloaders(
            data_dir=data_path,
            batch_size=batch_size,
            num_workers=0,
            use_entities=True,
            image_size=224
        )
        test_dataset = test_loader.dataset  # 为了统一后续使用
        print(f"测试集大小: {len(test_loader.dataset)}")
    else:
        # CSV文件
        test_dataset = FakeNewsDataset(
            data_path=data_path,
            text_col='text',
            image_col='image',
            label_col='label',
            max_length=512,
            image_size=224,
            use_entities=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"测试集大小: {len(test_dataset)}")
    
    # 评估
    print("开始评估...")
    all_preds = []
    all_probs = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='评估中')
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
    
    # 计算指标
    avg_loss = running_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"损失 (Loss):        {avg_loss:.4f}")
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1分数 (F1-Score):  {f1:.4f}")
    print(f"AUC:                {auc:.4f}")
    print("\n混淆矩阵:")
    print(cm)
    
    # 保存结果
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存文本结果
        results_file = os.path.join(output_dir, 'evaluation_results.txt')
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("EAKM-Net 评估结果\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"模型路径: {model_path}\n")
            f.write(f"测试集大小: {len(test_dataset)}\n\n")
            f.write(f"损失 (Loss):        {avg_loss:.4f}\n")
            f.write(f"准确率 (Accuracy):  {accuracy:.4f}\n")
            f.write(f"精确率 (Precision): {precision:.4f}\n")
            f.write(f"召回率 (Recall):    {recall:.4f}\n")
            f.write(f"F1分数 (F1-Score):  {f1:.4f}\n")
            f.write(f"AUC:                {auc:.4f}\n\n")
            f.write("混淆矩阵:\n")
            f.write(str(cm) + "\n")
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['真实', '虚假'],
                    yticklabels=['真实', '虚假'])
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        # 绘制ROC曲线
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
        plt.xlabel('假正例率 (FPR)')
        plt.ylabel('真正例率 (TPR)')
        plt.title('ROC曲线')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
        plt.close()
        
        print(f"\n结果已保存到: {output_dir}")
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm
    }


def main():
    parser = argparse.ArgumentParser(description='评估EAKM-Net模型')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--data_path', type=str, required=True, help='测试数据文件路径（CSV）')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--text_model', type=str, default='bert-base-uncased', help='文本模型')
    parser.add_argument('--image_backbone', type=str, default='resnet50', help='图像骨干网络')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='./results', help='结果保存目录')
    
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model_path,
        data_path=args.data_path,
        batch_size=args.batch_size,
        text_model=args.text_model,
        image_backbone=args.image_backbone,
        device=args.device,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()


