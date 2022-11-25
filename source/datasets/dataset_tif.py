import os, glob, re
import itertools
import pandas as pd
import numpy as np
import torch
from torch.utils import data
import rasterio as rio

class YieldDataset(data.Dataset):
    
    def __init__(self, image_folder, label_path, crop_type, return_id=False):
        """
        imagefolder (str): path to .tif satellite images
        label_file (str): path to csv containing annual yield 
                          of crop typeas labels
        crop_type (int): value corresponding to crop type 
                         e.g. 1 for corn
        """
        super(YieldDataset, self).__init__()

        self.image_folder = image_folder
        self.label_path = label_path
        self.crop_type = crop_type
        self.return_id = return_id

        # read labels, drop empty rows, create id column
        self.labels = pd.read_csv(label_path)
        self.labels =  self.labels[['Year', 'State ANSI', 'County ANSI', 'Value']].dropna()
        self.labels['GEOID'] = self.labels.apply(lambda row: self.create_id(row), axis=1)
        yield_pairs = [(id, str(y)) for id, y in zip(self.labels['GEOID'], self.labels['Year'])]

        # read images
        self.image_paths = glob.glob(self.image_folder +'/**/*.tif',  recursive=True)
        file_names = [x.split('/')[-1].replace('.tif', '') for x in self.image_paths]        
        years = np.array([x[:4] for x in file_names])
        geoids = np.array([x.split('_')[-1] for x in file_names])
        file_pairs = list(set(zip(geoids, years))) # keep only unique pairs

        # create geoid-year pair
        # keep only intersection of image files and yield
        self.pair = list(set(file_pairs) & set(yield_pairs))
        self.len = len(self.pair)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        geoid, year = self.pair[item]

        # match pattern in image path for a certain county & year
        # make sure it is sorted in correct order
        r = re.compile('.*'+year+'\d{4}_'+ geoid)
        image_paths_filtered = sorted(list(filter(r.match, self.image_paths)))

        # create the time series
        arr = []
        for i in range(len(image_paths_filtered)):
            # open image
            x_0 = rio.open(image_paths_filtered[i]).read()
            # select the crop type
            x_0 = np.where(x_0[0]==float(self.crop_type), x_0, np.nan)
            # take the average of the parcels -> change this if you want to use something different
            x_0 = np.expand_dims(np.nanmean(x_0[1:,:,:], axis=(1,2)), axis=0)
            arr.append(x_0)
        x = np.concatenate(arr, axis=0, dtype=np.float32)
        
        # get the label
        df_yield = self.labels.loc[(self.labels['GEOID']==geoid) & (self.labels['Year']==int(year))]['Value']
        y = np.nan if df_yield.empty else np.array(df_yield.values[0], dtype=np.float32)

        if self.return_id:
            return torch.from_numpy(x), torch.tensor(y), self.pair[item][0]

        else:
            return torch.from_numpy(x), torch.tensor(y)

    def create_id(self, row):
        """
        create 5-character code from state and county id
        """
        state = int(row['State ANSI'])
        county = int(row['County ANSI'])
        geoid = str(state).zfill(2) + str(county).zfill(3)

        return geoid
