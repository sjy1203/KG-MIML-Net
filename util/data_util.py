import json
import pickle

def read_config(file_path):
    return json.load(open(file_path, 'r'))

def read_pkl(file_path):
    return pickle.load(open(file_path, 'rb'))

