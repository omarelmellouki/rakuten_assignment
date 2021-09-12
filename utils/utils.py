import os
import logging
import pandas as pd
import numpy as np 

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def clean_dataset(path):

    df = pd.read_csv(path)
    df_valid = pd.DataFrame(columns = ['image_name', 'target'])
    df.head()
    count = 0

    for idx in range(len(df)):
        path = df.iloc[idx,0].split('/')[1]
        if os.path.isfile(os.path.join('data/images', path)):
            count += 1 
            df_valid = df_valid.append(df.iloc[idx], ignore_index = True)
    df_valid.to_csv('data/corrected_data_set.csv', index = False)

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def dataset_split(path):

    df = pd.read_csv(path)
    df.head(5)

    if not (os.path.isfile(os.path.join('train_data.csv')) and \
        os.path.isfile(os.path.join('val_data.csv')) and \
        os.path.isfile(os.path.join('test_data.csv'))):

        train, validate, test = \
                  np.split(df.sample(frac=1, random_state=42), 
                           [int(.9*len(df)), int(.95*len(df))])

        train.to_csv(os.path.join(path.split('/')[0], 'train_data.csv'), index = False)
        validate.to_csv(os.path.join(path.split('/')[0], 'validate_data.csv'), index = False)
        test.to_csv(os.path.join(path.split('/')[0], 'test_data.csv'), index = False)

    return  os.path.join(path.split('/')[0], 'train_data.csv'), \
            os.path.join(path.split('/')[0], 'validate_data.csv'), \
            os.path.join(path.split('/')[0], 'test_data.csv')

def count_classes(path):

    df = pd.read_csv(path)
    classes = df['target'].unique()
    nb_classes = len(df['target'].unique())
    return classes, nb_classes

dict_index = {1560: 0, 1320: 1, 2060: 2,
        1680: 3, 1741: 4, 1481: 5, 1920: 6, 2542: 7,
        1720: 8, 2885: 9, 2582: 10, 2766: 11, 1760: 12,
        2240: 13, 1300: 14, 2080: 15, 1840: 16, 1500: 17,
        2725: 18, 1064: 19, 2200: 20, 190: 21, 50: 22, 280: 23, 
        1141: 24, 80: 25, 1146: 26, 1142: 27, 240: 28, 1143: 29, 1280: 30, 1780: 31
    }

def from_original_index(index):

    return dict_index[index]

def to_original_index(index):

    reverse = {v:k for k,v in dict_index.items()}

    return reverse[index]