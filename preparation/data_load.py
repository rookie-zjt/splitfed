import os
from glob import glob

import pandas as pd
from sklearn.model_selection import train_test_split


def get_dataset(name='HAM10000'):
    global df
    if name == 'HAM10000':
        df = pd.read_csv('data/HAM10000_metadata.csv')
        print(df.head())
        # 名字全称（非必要）
        lesion_type = {
            'nv': 'Melanocytic nevi',
            'mel': 'Melanoma',
            'bkl': 'Benign keratosis-like lesions ',
            'bcc': 'Basal cell carcinoma',
            'akiec': 'Actinic keratoses',
            'vasc': 'Vascular lesions',
            'df': 'Dermatofibroma'
        }

        # merging both folders of HAM1000 dataset -- part1 and part2 -- into a single directory
        imageid_path = {os.path.splitext(os.path.basename(x))[0]: x
                        for x in glob(os.path.join("data", '*', '*.jpg'))}

        # print("path---------------------------------------", imageid_path.get)
        df['path'] = df['image_id'].map(imageid_path.get)
        df['cell_type'] = df['dx'].map(lesion_type.get)
        df['target'] = pd.Categorical(df['cell_type']).codes
        print(df['cell_type'].value_counts())
        print(df['target'].value_counts())

    return df

def split(df):
    # 将数据划分为训练集和测试集，并重置索引。
    train, test = train_test_split(df, test_size=0.2)
    train = train.reset_index()
    test = test.reset_index()
    return train, test
