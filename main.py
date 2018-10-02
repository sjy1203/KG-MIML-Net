from base.base_train import Trainer
from util.build_dataloader import get_loader
from util.build_vocabulary import build_vocab
from util.data_util import read_config
from base.base_eval import Eval
import os
from model.model import Seq2SeqAttention
import torch
import torch.nn as nn
import pickle
import argparse
import numpy as np
from util.sample import sample_pytorch 
torch.manual_seed(1203)

device = None

def _init():
    if not os.path.exists('saved/'):
        os.mkdir('saved/')
    config = read_config('config_template.json')
    med_voc = build_vocab(os.path.join(config['data_path'], 'unique_pres.pkl'), add_start_end=True)
    diag_voc = build_vocab(os.path.join(config['data_path'], 'unique_diag.pkl'))
    side_data = pickle.load(open(os.path.join(config['data_path'], 'side_data.pkl'), 'rb'))

    diag_embedding = None
    med_embedding = None

    device = torch.device('gpu:0') if config['use_cuda'] else torch.device('cpu')

    model = Seq2SeqAttention(
        src_emb_dim = config['model']['dim_word_src'],
        trg_emb_dim = config['model']['dim_word_trg'],
        src_vocab_size = len(diag_voc),
        trg_vocab_size = len(med_voc),
        src_hidden_dim = config['model']['dim'],
        trg_hidden_dim = config['model']['dim'],
        ctx_hidden_dim = config['model']['dim'],
        contextual_dim = len(side_data[0]),
        attention_mode = 'dot',
        batch_size = config['train']['batch_size'],
        pad_token_src = 0,
        pad_token_trg = 0,
        bidirectional = config['model']['bidirectional'],
        nlayers=1,
        nlayers_trg=1,
        dropout=0.,
        add_contextual_layer = config['model']['add_contextual_layer'],
        add_tree_embedding = config['model']['add_tree_embedding'],
        src_vocab = diag_voc,
        add_supervised= config['model']['add_supervised'],
        inputs_embedding = diag_embedding,
        outputs_embedding = med_embedding,
        add_tree_inputs_embedding= config['model']['add_tree_inputs_embedding'],
        add_tree_outputs_embedding= config['model']['add_tree_outputs_embedding'],
        unique_inputs_voc = diag_voc,
        unique_outputs_voc = med_voc,
        device = device
    )
    return model, diag_voc, med_voc, config   


def eval():
    model, diag_voc, med_voc, config = _init()

    data_loader = get_loader(config['train']['batch_size'], os.path.join(config['data_path'], 'test'), med_voc, diag_voc)
    print(len(data_loader))

    Eval(model, data_loader, med_voc, config).eval()


def train():
    model, diag_voc, med_voc, config = _init()

    data_loader = get_loader(config['train']['batch_size'], os.path.join(config['data_path'], 'train'), med_voc, diag_voc)
    eval_data_loader = get_loader(config['train']['batch_size'], os.path.join(config['data_path'], 'eval'), med_voc, diag_voc)
    print(len(data_loader))

    weight_mask = torch.ones(len(med_voc)).to(device)
    weight_mask[med_voc.word2idx['<pad>']] = 0
    loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).to(device)

    Trainer(model, loss_criterion, data_loader, config, eval_data_loader, med_voc).train()

def sample(patient_id):
    model, diag_voc, med_voc, config = _init()
    sample_pytorch(model, config, diag_voc, med_voc, patient_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-e", "--eval", help="mode: eval", action="store_true")
    parser.add_argument("-p", "--patient", help='patient_id',type=int, default=-1)
    args = parser.parse_args()
    if args.eval:
        print('start eval')
        eval()
    else:
        print(args.patient)
        if args.patient == -1:
            print('start train')
            train()
        else:
            sample(args.patient)
    
