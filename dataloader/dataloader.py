import os
import torch
import pandas as pd 
from PIL import Image
from torch.utils.data import Dataset
from utils.utils import from_original_index, to_original_index


class RakutenCatalogueLoader(Dataset):

    def __init__(self, csv_file, root_dir, transform  = None):
        self.product_target = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.product_target)

    def __getitem__(self, idx):

            if torch.is_tensor(idx):
                idx = idx.tolist()

            img_name = os.path.join(self.root_dir,
                                    self.product_target.iloc[idx, 0].split('/')[1])
            
            image = Image.open(img_name).convert('RGB')
        
            target = self.product_target.iloc[idx, 1]
            target = from_original_index(target)

            sample = {'image': image, 'target': target}

            if self.transform:
                sample['image'] = self.transform(sample['image'])

            return sample