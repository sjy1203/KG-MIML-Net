import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence
import os
import json
import numpy as np
from .base_eval import Eval
import pickle

class Trainer(object):
    def __init__(self, model, loss, data_loader, config, eval_data_loader, med_voc):
        super(Trainer, self).__init__()
        self.model = model
        self.data_loader = data_loader
        self.config = config
        self.criterion = loss
        self.optimizer = Adam(model.parameters(), lr=config['train']['lr'])
        self.evaluator = Eval(model, eval_data_loader, med_voc, config)

    def train(self):
        model, data_loader, criterion, optimizer, config = self.model, self.data_loader, self.criterion, self.optimizer, self.config
        if not os.path.exists('saved/%s' % config['model_name']):
            os.mkdir('saved/%s' % config['model_name'])
        use_cuda = config['use_cuda']
        if config['resume']:
            model.load_state_dict(torch.load(open(config['resume_path'], 'rb')))
        model = model.cuda() if use_cuda else model 
        criterion = criterion.cuda() if use_cuda else criterion

        total_step = len(data_loader)
        loss_records = []
        jaccard_records = []
        for epoch in range(config['train']['epoch']):
            for i, (diags, diags_len, sides, meds, meds_len) in enumerate(data_loader):
                # diags = np.zeros((len(diags),len(diags[0])))
                outputs = model(torch.LongTensor(diags), torch.LongTensor(meds[:,:-1]), torch.FloatTensor(sides), trg_mask=diags_len)
                targets = torch.LongTensor(meds[:,1:]).cuda()
            
                loss = criterion(outputs, targets)
                model.zero_grad()
                loss.backward()
                optimizer.step()

                if i % config['log_step'] == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                        epoch,
                        config['train']['epoch'],
                        i,
                        total_step,
                        loss.item()
                    ))

                # Save the model checkpoints
                if (i+1) % config['save_step'] == 0:
                    model_name =  '{}-{}-{}-{:.4f}.ckpt'.format(config['model_name'], epoch+1, i+1, loss.item())
                    torch.save(model.state_dict(), open(
                        os.path.join('saved/', config['model_name'], model_name),
                        'wb'
                        )
                    )

                    json.dump(config, open(
                        os.path.join('saved/', config['model_name'], 'config.json'),'w') 
                    )
            loss_records.append(loss.item())

            with torch.no_grad():
                score = self.evaluator.eval(model=model)
                jaccard_records.append(score)

        pickle.dump(jaccard_records, open(os.path.join('saved/', config['model_name'], 'score.pkl'),'wb'))
        pickle.dump(loss_records, open(os.path.join('saved/', config['model_name'], 'loss.pkl'),'wb'))


        torch.save(model.state_dict(), open(
            os.path.join('saved/', config['model_name'], '{}-final.ckpt'.format(config['model_name'])), 
            'wb')
        )
        json.dump(config, open(
            os.path.join('saved/', config['model_name'], 'config.json'), 'w')
        )
        



