import pandas as pd
import os
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

# 加载数据
df_durations = pd.read_csv(r'fr\clip_durations.tsv', sep='\t')
df_validated = pd.read_csv(r'fr\validated.tsv', sep='\t')

# 合并数据（通过文件名关联）
df = pd.merge(
    left=df_validated[['path', 'sentence']],  # 只保留需要的列
    right=df_durations,
    left_on='path',
    right_on='clip',
    how='inner'
)

# 检查无效数据
print(f"总样本数: {len(df)}")
print(f"缺失值检查:\n{df.isnull().sum()}")


output_path='./merged_data.csv'
if not os.path.exists(output_path):
    df.to_csv(output_path,index=False)
else:
    print(f'文件{output_path}存在，未进行覆盖')
