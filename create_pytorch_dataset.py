import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchtext
from torchtext.vocab import build_vocab_from_iterator
from torchtext import _extension
# 设置随机种子
torch.manual_seed(42)

# --- 步骤1: 加载MFCC特征和元数据 ---
def load_features_and_metadata(mfcc_path, metadata_path):
    """
    加载预处理的MFCC特征和元数据
    返回:
        features (np.ndarray): MFCC特征数组 (num_samples, max_length, n_mfcc)
        df (pd.DataFrame): 包含'path', 'sentence'等列的DataFrame
    """
    features = np.load(mfcc_path)
    df = pd.read_csv(metadata_path)
    return features, df
from collections import defaultdict
import spacy
# --- 步骤2: 构建词汇表 ---
def build_vocab(text_series):
    """
    从文本数据构建词汇表
    返回:
        vocab: torchtext的词汇表对象
        tokenizer: 分词函数
    """
    # 法语分词器（需安装spacy法语模型
    tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='fr_core_news_sm')
    
    # 构建词汇表
    def yield_tokens():
        for text in text_series:
            yield tokenizer(text.lower())  # 转为小写
    
    vocab = build_vocab_from_iterator(
        yield_tokens(),
        specials=['<pad>', '<unk>', '<sos>', '<eos>']
    )
    vocab.set_default_index(vocab['<unk>'])
    return vocab, tokenizer

# --- 步骤3: 定义PyTorch数据集 ---
class FrenchSpeechDataset(Dataset):
    def __init__(self, features, df, vocab, tokenizer, max_text_length=100):
        """
        参数:
            features: MFCC特征数组 (num_samples, max_length, n_mfcc)
            df: 包含文本的DataFrame
            vocab: 词汇表对象
            tokenizer: 分词函数
            max_text_length: 文本最大长度（超过则截断）
        """
        self.features = features
        self.texts = df['sentence'].values
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 获取MFCC特征
        mfcc = torch.FloatTensor(self.features[idx])  # (max_length, n_mfcc)
        
        # 处理文本
        text = self.texts[idx]
        tokens = self.tokenizer(text.lower())
        token_ids = [self.vocab['<sos>']] + [self.vocab[token] for token in tokens] + [self.vocab['<eos>']]
        
        # 裁剪或填充文本
        if len(token_ids) > self.max_text_length:
            token_ids = token_ids[:self.max_text_length]
        else:
            token_ids = token_ids + [self.vocab['<pad>']] * (self.max_text_length - len(token_ids))
        
        return {
            'input_features': mfcc,
            'target_text': torch.LongTensor(token_ids),
            'original_text': text
        }


if __name__ == "__main__":
    # 加载数据
    MFCC_PATH = "processed/mfcc_features.npy"
    METADATA_PATH = "processed/metadata.csv"
    features, df = load_features_and_metadata(MFCC_PATH, METADATA_PATH)
    
    # 构建词汇表
    vocab, tokenizer = build_vocab(df['sentence'])
    print(f"词汇表大小: {len(vocab)}")
    
    # 初始化数据集
    dataset = FrenchSpeechDataset(features, df, vocab, tokenizer)
    
    # 划分训练集和验证集
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        random_state=42
    )
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    
    # 创建DataLoader
    batch_size = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: {
            'input_features': torch.stack([item['input_features'] for item in batch]),
            'target_text': torch.stack([item['target_text'] for item in batch]),
            'original_text': [item['original_text'] for item in batch]
        }
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 测试一个batch
    sample_batch = next(iter(train_loader))
    print("\nBatch示例:")
    print("MFCC特征形状:", sample_batch['input_features'].shape)  # (batch_size, max_length, n_mfcc)
    print("文本ID形状:", sample_batch['target_text'].shape)      # (batch_size, max_text_length)
    print("原文:", sample_batch['original_text'][0])
    
    # 保存词汇表（后续推理使用）
    torch.save(vocab, "processed/vocab.pth")
    print("\n词汇表已保存到 processed/vocab.pth")