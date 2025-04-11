import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split


# Data preprocessing: Transformation
# 变换图像，增强模型泛化能力
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.Pad(3),
                                       transforms.RandomRotation(10),
                                       transforms.CenterCrop(64),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=mean, std=std)
                                       ])

test_transforms = transforms.Compose([transforms.Pad(3),
                                      transforms.CenterCrop(64),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std)
                                      ])

# Custom dataset prepration in Pytorch format
class SkinData(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df['path'][index]).resize((64, 64))
        y = torch.tensor(int(self.df['target'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y

def split(df):
    # 将数据划分为训练集和测试集，并重置索引。
    train, test = train_test_split(df, test_size=0.2)
    train = train.reset_index()
    test = test.reset_index()
    return train, test

# 独立同分布（iid）数据集（用于划分不同客户端的数据集）
def dataset_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

