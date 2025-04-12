import pandas as pd
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# 设置随机种子（确保可复现性）
torch.manual_seed(42)


# 加载CSV文件
df = pd.read_csv('merged_data.csv')
# y,sr=librosa.load(os.path.join('fr/clips',df.iloc[0]['path']),sr=16000)
# y_trimmed,_=librosa.effects.trim(y,top_db=20)
# mfcc=librosa.feature.mfcc(
#     y=y_trimmed,
#     sr=sr,
#     n_mfcc=13,
#     hop_length=int(sr*0.01),
#     n_fft=int(sr*0.025)
# )
# mfcc=(mfcc-np.mean(mfcc))/np.std(mfcc)
# max_length=500
# if mfcc.shape[1]<max_length:
#     mfcc=np.pad(mfcc,
#                 pad_width=((0,0),(0,max_length-mfcc.shape[1])),
#                 mode='constant'
#                 )
# else:
#     mfcc=mfcc[:,:max_length]


def extract_mfcc(audio_path, n_mfcc=13, max_length=500):
    """提取MFCC特征并填充/裁剪到固定长度"""
    y, sr = librosa.load(os.path.join('fr/clips', audio_path), sr=16000)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)  # 降噪
    
    # 提取MFCC
    mfcc = librosa.feature.mfcc(
        y=y_trimmed, 
        sr=sr, 
        n_mfcc=n_mfcc,
        hop_length=int(sr * 0.01),  # 10ms帧移
        n_fft=int(sr * 0.025)       # 25ms帧长
    )
    
    # 归一化
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    
    # 处理时序长度
    if mfcc.shape[1] < max_length:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_length - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_length]
    
    return mfcc.T  # 形状: (时间步, 特征维度)

# # 测试单个样本
# sample_mfcc = extract_mfcc(df.iloc[0]['path'])
# print("\nMFCC形状:", sample_mfcc.shape)  # 应输出 (500, 13)
from tqdm import tqdm  # 进度条工具
def process_all_audio(csv_path, clips_dir, output_dir):
    """处理所有音频并保存MFCC特征和元数据"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载CSV文件
    df = pd.read_csv(csv_path)
    
    # 预分配数组（假设 max_length=500, n_mfcc=13）
    all_mfccs = np.zeros((len(df), 500, 13))  # 形状: (样本数, 时间步, 特征维度)
    
    # 逐个处理音频
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Audio"):
        audio_path = os.path.join(clips_dir, row['path'])
        
        try:
            mfcc = extract_mfcc(audio_path)
            all_mfccs[idx] = mfcc  # 存入数组
        except Exception as e:
            print(f"处理失败: {audio_path}, 错误: {e}")
            all_mfccs[idx] = np.zeros((500, 13))  # 失败时填充零
    
    # 保存MFCC特征和元数据
    np.save(os.path.join(output_dir, 'mfcc_features.npy'), all_mfccs)
    df.to_csv(os.path.join(output_dir, 'metadata.csv'), index=False)
    print(f"\nMFCC特征已保存到 {output_dir}/mfcc_features.npy")
    print(f"元数据已保存到 {output_dir}/metadata.csv")

if __name__ == "__main__":
    # 输入参数
    CSV_PATH = "merged_data.csv"          # 您的输入CSV路径
    CLIPS_DIR = "fr/clips"                   # 音频文件夹路径
    OUTPUT_DIR = "processed"              # 输出目录
    
    # 运行处理
    process_all_audio(CSV_PATH, CLIPS_DIR, OUTPUT_DIR)