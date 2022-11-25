import os, glob, re
import itertools
import pandas as pd
import numpy as np
import torch
from torch.utils import data
import json

MODIS_SCALE_FACTOR = 1e-4
MODIS_FEATURES_IDX = [i for i in range(0, 7)]

class YieldDataset(data.Dataset):
    
    def __init__(self, image_folder, label_path, crop_type, year, return_id=False):
        super(YieldDataset, self).__init__()

        self.image_folder = image_folder
        self.label_path = label_path
        self.crop_type = crop_type
        self.return_id = return_id

        # read images
        self.image_paths = glob.glob(self.image_folder +'/**/*.npy',  recursive=True)
        file_names = [x.split('/')[-1].replace('.npy', '') for x in self.image_paths]
        years = np.array([int(x[:4]) for x in file_names])
        geoids = np.array([x.split('_')[-1] for x in file_names])
        df_image = pd.DataFrame({'id':geoids, 'year':years})

        # read the labels
        with open(label_path) as f:
            yield_per_parcel = json.load(f)

        years, yields, ids = [], [], []
        for item in yield_per_parcel.items():
            years.append(int(item[0].split('_')[1]))
            ids.append(item[0].split('_')[0])
            yields.append(item[1])

        df_yield = pd.DataFrame({'id':ids, 'year':years, 'yield':yields})
        df_yield = df_yield[df_yield['year'] == year]
        # keep only intersection
        self.data = pd.merge(df_yield, df_image, how='inner')
        self.len = len(self.data.index)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        row = self.data.iloc[item]

        features = np.load(self.image_folder + '{}_{}.npy'.format(row['year'], row['id']))

        #scale the MODIS bands based on the scale provided in the data documentation
        for i in MODIS_FEATURES_IDX:
            features[:,i] = features[:,i] * MODIS_SCALE_FACTOR

        if self.return_id:
            return torch.from_numpy(features), torch.tensor(row['yield']), row['id']
        else:
            return torch.from_numpy(features), torch.tensor(row['yield'])
