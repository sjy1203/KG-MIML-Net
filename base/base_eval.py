import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence
import os
import json
import numpy as np
import pickle
from sklearn.metrics import jaccard_similarity_score, roc_auc_score
from .beam import SequenceGenerator

class Eval(object):
    def __init__(self, model, data_loader, med_voc, config):
        super(Eval, self).__init__()
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.med_voc = med_voc

    def eval(self, model=None):
        data_loader, config = self.data_loader, self.config
        med_voc = self.med_voc
        if model is None:
            model = self.model
            if not os.path.exists('saved/%s' % config['model_name']):
                os.mkdir('saved/%s' % config['model_name'])
            use_cuda = config['use_cuda']
            if config['resume']:
                model.load_state_dict(torch.load(open(config['resume_path'], 'rb')))
            model = model.cuda() if use_cuda else model 
            model.eval()
        

        total_step = len(data_loader)
        with open(os.path.join(config['data_path'], 'test/diag_data.pkl'), 'rb') as f:
            data = pickle.load(f)
        data_len = len(data)
        del data
        y_pred = np.zeros((data_len, len(med_voc)))
        y_gt = np.zeros((data_len, len(med_voc)))

        # seqG = SequenceGenerator(model, beam_size=3)

        for i, (diags, diags_len, sides, meds, meds_len) in enumerate(data_loader):
            # seqs = seqG.beam_search_m(sides, diags)
            # diags = np.zeros((len(diags),len(diags[0])))
            outputs = self.decode_batch(model, torch.LongTensor(diags), torch.FloatTensor(sides), max_len=20, trg_mask=diags_len)
            outputs = outputs.detach().cpu().numpy().astype(int)
            # outputs = [seq.output for seq in seqs]
            meds = np.array(meds,dtype=int)
            for j in range(len(meds)):
                y_pred[[[i*config['train']['batch_size']+j]*len(outputs[j]), outputs[j]]] = 1
                y_gt[[[i*config['train']['batch_size']+j]*len(meds[j]), meds[j]]] = 1

            if i % config['log_step'] == 0:
                print('prcessing [{} of {}]'.format(i, len(data_loader)))
        jac_score = jaccard_similarity_score(y_gt, y_pred)
        # print('micro roc_auc_score', roc_auc_score(y_gt, y_pred, average='micro'))    
        print('jaccard_similarity_score', jac_score) 
        return jac_score        
    
    def decode_batch(self, model, inputs, sides, max_len=20, trg_mask=None):
        start_token = 1
        targets = torch.ones((len(inputs), start_token), dtype=torch.long)
        for _ in range(max_len):
            outputs = model(inputs, targets, sides, trg_mask)
            outputs = model.decode(outputs) #(batch, seq, C)
            decoder_argmax = outputs.detach().cpu().numpy().argmax(axis=-1)
            next_preds = torch.LongTensor(decoder_argmax[:,-1])
            targets = torch.cat(
                (targets, next_preds.unsqueeze(1)),
                1
            )
        return targets





