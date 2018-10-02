import torch
from .data_util import read_pkl
import numpy as np

def sample_pytorch(model, config, diag_voc, pres_voc, patient_id, device=None):
    model.load_state_dict(torch.load(open(config['resume_path'], 'rb')))
    device = torch.device('cpu') if device is None else device
    model = model.to(device)
    side_data = read_pkl('./data/side_data.pkl')
    diag_data = read_pkl('./data/diag_data.pkl')
    pres_data = read_pkl('./data/pres_data.pkl')

    p_side = side_data[patient_id]
    p_diag = diag_data[patient_id]
    p_pres = pres_data[patient_id]

    p_inputs = [diag_voc(i) for i in p_diag]
    p_trgs = [pres_voc(i) for i in p_pres]

    with torch.no_grad():
        outputs = decode_batch(model, torch.LongTensor(p_inputs).unsqueeze(dim=0), torch.FloatTensor(p_side).unsqueeze(dim=0), max_len=20, trg_mask=[len(p_diag)])
        outputs = outputs.detach().cpu().numpy()[0]
        print('diags name:', p_diag)
        print('true drugs name:', p_pres)
        print('pred drugs name:', [pres_voc.idx2word[i] for i in outputs])
        print('true drugs idx:', p_trgs)
        print('pred drugs idx:', list(outputs))


def decode_batch(model, inputs, sides, max_len=20, trg_mask=None):
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
  


            
                
             
            