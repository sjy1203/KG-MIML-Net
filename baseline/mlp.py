import pickle
import os
from sklearn.metrics import jaccard_similarity_score, average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.insert(0,os.path.join(os.getcwd(),'..'))
from util.build_vocabulary import build_vocab

from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, StandardScaler
import numpy as np

def _read(data_path, file_name):
    with open(os.path.join(data_path, file_name), 'rb') as f:
        data = pickle.load(f)
    return data

def _trans(data, voc):
    res = []
    for item in data:
        res.append([voc(token) for token in item])
    return res

def read_data(train_path, eval_path, diag_voc, med_voc):
    diags_train = _read(train_path, 'diag_data.pkl')
    meds_train = _read(train_path, 'pres_data.pkl')
    side_train = np.array(_read(train_path, 'side_data.pkl'))

    diags_eval = _read(eval_path, 'diag_data.pkl')
    meds_eval = _read(eval_path, 'pres_data.pkl')
    side_eval = np.array(_read(eval_path, 'side_data.pkl'))

    sides = np.concatenate([side_train,side_eval], axis=0)
    sides = StandardScaler().fit_transform(sides)
    sides = MinMaxScaler().fit_transform(sides)
    print('side size:[{}]'.format(sides.shape))

    train_len = len(diags_train)

    diags = _trans(diags_train + diags_eval, diag_voc)
    meds = _trans(meds_train + meds_eval, med_voc)

    mlb = MultiLabelBinarizer()
    diags = mlb.fit_transform(diags)
    meds = mlb.fit_transform(meds)
    diags = np.concatenate([diags,sides],axis=1)
    print('input size:',diags.shape)
    return diags[:train_len], diags[train_len:], meds[:train_len], meds[train_len:]
    

def main(model, only_eval=False, model_name='mlp.pkl', data_path='../data'):
    diag_voc = build_vocab(os.path.join(data_path, 'unique_diag.pkl'))
    med_voc = build_vocab(os.path.join(data_path, 'unique_pres.pkl'))

    print('unique med:[{}], unique diag:[{}]'.format(len(med_voc), len(diag_voc)))

    x_train, x_test, y_train, y_test = read_data(os.path.join(data_path, 'train'), os.path.join(data_path, 'eval'), diag_voc, med_voc)
    if not only_eval:
        model.fit(x_train, y_train)

        if not os.path.exists('saved'):
            os.mkdir('saved')
        with open(os.path.join('saved', model_name), 'wb') as f:
            pickle.dump(model, f) 

    with open(os.path.join('saved', model_name), 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(x_test)
    print('y_pred:',y_pred[0:2])

    print('micro roc_auc_score', roc_auc_score(y_test, y_pred, average='micro'))
    print('jaccard_similarity_score', jaccard_similarity_score(y_test, y_pred))

if __name__ == '__main__':
    #model = RandomForestClassifier(n_estimators=300, verbose=True)
    model = MLPClassifier(hidden_layer_sizes=(512,512,512), verbose=True, max_iter=500)
    main(model, model_name='mlp500.pkl', only_eval=False)

