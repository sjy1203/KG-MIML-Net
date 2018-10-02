import torch
import torch.utils.data as data
import pickle
import numpy as np
import os

class MIMICDataset(data.Dataset):
    def __init__(self, med_data, diag_data, side_data, med_voc, diag_voc):
        self.med_data = med_data
        self.diag_data = diag_data
        self.side_data = side_data

        self.med_voc = med_voc
        self.diag_voc = diag_voc
    
    def __getitem__(self, index):
        meds = self.med_data[index]
        diags = self.diag_data[index]
        side = self.side_data[index]

        diags_input = []
        diags_input.extend([self.diag_voc(token) for token in diags])

        meds_output = []
        meds_output.append(self.med_voc('<start>'))
        meds_output.extend([self.med_voc(token) for token in meds])
        meds_output.append(self.med_voc('<end>'))

        return diags_input, side, meds_output

    def __len__(self):
        return len(self.diag_data)

def padding_batch(data):
    lengths = [len(item) for item in data]
    padding = np.zeros((len(data), max(lengths)))
    for i, item in enumerate(data):
        end = lengths[i]
        padding[i, :end] = item[:end]
    return padding, lengths

def collate_fn(data):
    data.sort(key=lambda x: len(x[2]), reverse=True)
    diags, sides, meds = zip(*data)
    diags, diags_length = padding_batch(diags)
    meds, meds_length = padding_batch(meds)
    sides = np.stack(sides, 0)

    return diags, diags_length, sides, meds, meds_length

def get_loader(batch_size, data_path, med_voc, diag_voc, shuffle=True):
    
    with open(os.path.join(data_path, 'side_data.pkl'), 'rb') as f:
        side = pickle.load(f)
    with open(os.path.join(data_path, 'diag_data.pkl'), 'rb') as f:
        diag = pickle.load(f)    
    with open(os.path.join(data_path, 'pres_data.pkl'), 'rb') as f:
        med = pickle.load(f)    
    mimic = MIMICDataset(med, diag, side, med_voc=med_voc, diag_voc=diag_voc)

    data_loader = torch.utils.data.DataLoader(dataset = mimic,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=2,
    collate_fn=collate_fn)

    return data_loader

def read_data(data_path):
    with open(os.path.join(data_path, 'diag_data.pkl'), 'rb') as f:
        diag = pickle.load(f)    
    with open(os.path.join(data_path, 'pres_data.pkl'), 'rb') as f:
        med = pickle.load(f)  
    
    return diag, med
