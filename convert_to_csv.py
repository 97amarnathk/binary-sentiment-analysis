import numpy as np
import pandas as pd
import os
import io

from tqdm import tqdm

basepath = './aclImdb'
datapoints = 50000

labels = {'pos': 1, 'neg':0}

df = pd.DataFrame()

#append to dataframe
with tqdm(total = datapoints) as progress_bar:
    for folder in ('test', 'train'):
        for sub in ('pos', 'neg'):
            path = os.path.join(basepath, folder, sub)
            for file in os.listdir(path):
                with open(os.path.join(path, file), 'r', encoding = 'utf-8') as infile:
                    text_data = infile.read()
                df = df.append([[text_data, labels[sub]]], ignore_index = True)
                progress_bar.update(1)


print(df.shape)
df.columns = ['review', 'sentiment']

#shuffle data
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))

#export as csv
df.to_csv('./movie_data.csv', index = False, encoding = 'utf-8')