import os
import sys
import numpy as np
import pickle
from collections import Counter, defaultdict
from sklearn.metrics import jaccard_similarity_score, average_precision_score, roc_auc_score

sys.path.insert(0,os.path.join(os.getcwd(), '..'))
from util.build_vocabulary import build_normal_vocab
from util.build_dataloader import read_data

def train(data_path='../data'):
    diags, meds = read_data(os.path.join(data_path,'train'))
    diag_voc = build_normal_vocab(os.path.join(data_path, 'unique_diag.pkl'))
    med_voc = build_normal_vocab(os.path.join(data_path, 'unique_pres.pkl'))
    
    co_matrix = defaultdict(Counter)
    for i in range(len(diags)):
        
        for diag_item in diags[i]:
            for med_item in meds[i]:
                co_matrix[diag_voc(diag_item)][med_voc(med_item)] += 1
    if not os.path.exists('saved/'):
        os.mkdir('saved/')
    with open('saved/co_matrix.pkl', 'wb') as f:
        pickle.dump(co_matrix, f)

def eval(K=5, data_path='../data'):
    diags, meds = read_data(os.path.join(data_path,'test'))
    diag_voc = build_normal_vocab(os.path.join(data_path, 'unique_diag.pkl'))
    med_voc = build_normal_vocab(os.path.join(data_path, 'unique_pres.pkl'))
    with open('saved/co_matrix.pkl', 'rb') as f:
        co_matrix = pickle.load(f)
    
    y_pred = np.zeros((len(diags), len(med_voc)))
    y_test = np.zeros((len(diags), len(med_voc)))

    for i in range(len(diags)):
        for diag_item in diags[i]:
            k_most_common = co_matrix[diag_voc(diag_item)].most_common(K)
            if len(k_most_common) == 0:
                continue
            k_most_common, _ = zip(*k_most_common)
            y_pred[i, list(k_most_common)] = 1
        for med_item in meds[i]:
            y_test[i, med_voc(med_item)] = 1
    print('K=',K)
    print('micro roc_auc_score', roc_auc_score(y_test, y_pred, average='micro'))
    print('jaccard_similarity_score', jaccard_similarity_score(y_test, y_pred))   



if __name__ == '__main__':
    train()    
    for i in range(8,9):
        eval(i)