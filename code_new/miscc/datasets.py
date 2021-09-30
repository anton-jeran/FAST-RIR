from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
# from PIL import Image
import soundfile as sf
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
from scipy import signal

from miscc.config import cfg


class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train',rirsize=4096): #, transform=None, target_transform=None):

        # self.transform = transform
        # self.target_transform = target_transform
        self.rirsize = rirsize
        self.data = []
        self.data_dir = data_dir       
        self.bbox = None
        
        split_dir = os.path.join(data_dir, split)

        self.filenames = self.load_filenames(split_dir)
        self.embeddings = self.load_embedding(split_dir)

    def get_RIR(self, RIR_path):
        wav,fs = sf.read(RIR_path) #Image.open(RIR_path).convert('RGB')
        length = wav.size
        # crop_length = int((16384*(80))/(64))
        crop_length = 4096 #int(16384)
        if(length<crop_length):
            zeros = np.zeros(crop_length-length)
            RIR_original = np.concatenate([wav,zeros])
        else:
            RIR_original = wav[0:crop_length]

        # resample_length = int((self.rirsize*(80))/(64))
        resample_length = int(self.rirsize)
        if(resample_length==16384):
            RIR = RIR_original
        else:
            RIR = RIR_original#signal.resample(RIR_original,resample_length)
        RIR = np.array([RIR]).astype('float32')



        # if bbox is not None:
        #     R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        #     center_x = int((2 * bbox[0] + bbox[2]) / 2)
        #     center_y = int((2 * bbox[1] + bbox[3]) / 2)
        #     y1 = np.maximum(0, center_y - R)
        #     y2 = np.minimum(height, center_y + R)
        #     x1 = np.maximum(0, center_x - R)
        #     x2 = np.minimum(width, center_x + R)
        #     RIR = RIR.crop([x1, y1, x2, y2])
        # load_size = int(self.rirsize * 76 / 64)
        # RIR = RIR.resize((load_size, load_size), PIL.Image.BILINEAR)
        # if self.transform is not None:
        #     RIR = self.transform(RIR)
        return RIR


    def load_embedding(self, data_dir):
        embedding_filename   = '/embeddings.pickle'  
        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f)
            # embeddings = np.array(embeddings)
            # # embedding_shape = [embeddings.shape[-1]]
            # print('embeddings: ', embeddings.shape)
        return embeddings

    # def load_class_id(self, data_dir, total_num):
    #     if os.path.isfile(data_dir + '/class_info.pickle'):
    #         with open(data_dir + '/class_info.pickle', 'rb') as f:
    #             class_id = pickle.load(f)
    #     else:
    #         class_id = np.arange(total_num)
    #     return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def __getitem__(self, index):
        key = self.filenames[index]

        data_dir = self.data_dir

        # captions = self.captions[key]
        embeddings = self.embeddings[key]
        RIR_name = '%s/RIR/%s.wav' % (data_dir, key)
        RIR = self.get_RIR(RIR_name)
        embedding = np.array(embeddings).astype('float32')
        # if self.target_transform is not None:
        #     embedding = self.target_transform(embedding)
        return RIR, embedding

    def __len__(self):
        return len(self.filenames)
