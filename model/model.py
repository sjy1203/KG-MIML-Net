import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

import torch
from .basic_model import LSTMAttentionDot, SupervisedAttention, InputsTreeEmbedding, OutputsTreeEmbedding

import sys
import os
sys.path.insert(0,os.path.join(os.getcwd(),'..'))
from util.build_vocabulary import TreeVocabulary, build_instances_tree, build_labels_tree

class Seq2SeqAttention(BaseModel):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(
        self,
        src_emb_dim,
        trg_emb_dim,
        src_vocab_size,
        trg_vocab_size,
        src_hidden_dim,
        trg_hidden_dim,
        ctx_hidden_dim,
        contextual_dim,
        attention_mode,
        batch_size,
        pad_token_src,
        pad_token_trg,
        bidirectional=True,
        nlayers=2,
        nlayers_trg=2,
        dropout=0.,
        add_contextual_layer = False,
        add_tree_embedding = False,
        src_vocab = None,
        add_supervised = False,
        inputs_embedding = None,
        outputs_embedding = None,
        add_tree_inputs_embedding = False,
        add_tree_outputs_embedding = False,
        unique_inputs_voc = None,
        unique_outputs_voc = None,
        device = None
    ):
        """Initialize model."""
        super(Seq2SeqAttention, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.ctx_hidden_dim = ctx_hidden_dim
        self.attention_mode = attention_mode
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.pad_token_src = pad_token_src
        self.pad_token_trg = pad_token_trg
        self.add_contextual_layer = add_contextual_layer
        self.add_tree_embedding = add_tree_embedding
        self.add_supervised = add_supervised
        self.add_tree_inputs_embedding = add_tree_inputs_embedding
        self.add_tree_outputs_embedding = add_tree_outputs_embedding
        self.device = torch.device('cpu') if device is None else device
        
        if add_tree_inputs_embedding:
            self.src_embedding = InputsTreeEmbedding(build_instances_tree(unique_inputs_voc), 
            unique_inputs_voc, 
            src_emb_dim
            )
        else:
            self.src_embedding = nn.Embedding(
                src_vocab_size,
                src_emb_dim,
                self.pad_token_src,
                _weight= None if inputs_embedding is None else torch.cat([torch.zeros((1, src_emb_dim)), torch.Tensor(inputs_embedding)], dim=0)
            )

        if add_tree_outputs_embedding:
            self.trg_embedding = OutputsTreeEmbedding(build_labels_tree(unique_outputs_voc), 
            unique_outputs_voc, 
            trg_emb_dim)
        else:
            self.trg_embedding = nn.Embedding(
                trg_vocab_size,
                trg_emb_dim,
                self.pad_token_trg,
                _weight= None if outputs_embedding is None else torch.cat([torch.zeros((1, trg_emb_dim)), torch.Tensor(outputs_embedding)], dim=0)
            )

        self.src_hidden_dim = src_hidden_dim // 2 \
            if self.bidirectional else src_hidden_dim
        self.encoder = nn.LSTM(
            src_emb_dim*2 if self.add_tree_inputs_embedding else src_emb_dim,
            self.src_hidden_dim,
            nlayers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

        self.decoder = LSTMAttentionDot(
            trg_emb_dim*2 if self.add_tree_outputs_embedding else trg_emb_dim,
            trg_hidden_dim,
            batch_first=True
        )

        self.encoder2decoder = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            trg_hidden_dim
        )

        if add_contextual_layer:
            self.contextual_layer = ContextualLayer(trg_hidden_dim, contextual_dim, src_hidden_dim if add_supervised else 0)
        if add_supervised:
            self.supervised_attention = SupervisedAttention()
            
        if add_tree_embedding:
            self.tree_embedding = TreeWeightedEmbedding(src_vocab, src_emb_dim, num_layers=1, device=self.device)

        self.decoder2vocab = nn.Linear(trg_hidden_dim, trg_vocab_size)

        #self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)
        h0_encoder = torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        )
        c0_encoder = torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        )

        return h0_encoder.to(self.device), c0_encoder.to(self.device)

    def forward(self, input_src, input_trg, input_contextual=None, trg_mask=None, ctx_mask=None):
        ####src embedding
        input_src_cuda = input_src.to(self.device)
        src_emb = self.src_embedding(input_src_cuda)
            
        if self.add_tree_embedding:
            src_emb = self.tree_embedding(input_src, src_emb)
            # src_emb = torch.cat([src_emb, src_emb_tree], dim=-1)

        ####trg embedding
        input_trg_cuda = input_trg.to(self.device)
        trg_emb = self.trg_embedding(input_trg_cuda)


        #### contextual information        
        input_contextual = input_contextual.to(self.device) if not input_contextual is None else input_contextual
        
    
        self.h0_encoder, self.c0_encoder = self.get_state(input_src)
        src_h, (src_h_t, src_c_t) = self.encoder(
            src_emb, (self.h0_encoder, self.c0_encoder)
        )

        

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]
        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        ctx = src_h.transpose(0, 1)

        trg_h, (_, _) = self.decoder(
            trg_emb,
            (decoder_init_state, c_t),
            ctx,
            ctx_mask
        )

        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size()[0] * trg_h.size()[1],
            trg_h.size()[2]
        )

        if self.add_contextual_layer:
            if not input_contextual is None:
                input_contextual = input_contextual.repeat(1, trg_h.size(1)).view(-1, input_contextual.size(1))

                if self.add_supervised:
                    supervised_inputs = self.supervised_attention(src_h, trg_mask) #(batch, dim)
                    supervised_inputs = supervised_inputs.repeat(1, trg_h.size(1)).view(-1, supervised_inputs.size(1))
                    trg_h_reshape = self.contextual_layer(trg_h_reshape, input_contextual, supervised_inputs)
                else:
                    trg_h_reshape = self.contextual_layer(trg_h_reshape, input_contextual)

        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size()[0],
            trg_h.size()[1],
            decoder_logit.size()[1]
        )
        return decoder_logit.permute(0,2,1) # (batch, C, seq)

    def decode(self, logits):
        """Return probability distribution over words."""
        logits = logits.permute(0,2,1) # (batch, seq, C)
        word_probs = F.softmax(logits, dim=-1) # (batch, seq, C)
        return word_probs

class ContextualLayer(nn.Module):
    def __init__(self, trg_hidden_dim, contextual_dim, src_hidden_dim):
        super(ContextualLayer, self).__init__()
        self.cat_size = trg_hidden_dim + contextual_dim + src_hidden_dim
        self.layer = nn.Sequential(
            nn.Linear(self.cat_size, self.cat_size),
            nn.ReLU(),
            nn.Linear(self.cat_size, trg_hidden_dim),
            nn.ReLU()
        )

    def forward(self, decoder_inputs, contextual_inputs, supervised_inputs=None):
        if supervised_inputs is None:
            return self.layer(torch.cat([decoder_inputs, contextual_inputs],dim=1))
        else:
            return self.layer(torch.cat([decoder_inputs, contextual_inputs, supervised_inputs],dim=1))

class TreeWeightedEmbedding(nn.Module):
    def __init__(self, input_voc, emb_dim, num_layers, device=None):
        super(TreeWeightedEmbedding, self).__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.device = torch.device('cpu') if device is None else device

        # self.tree = L3_Tree(input_voc)
        self.tree = TreeVocabulary(input_voc)
        
        self.embed = nn.Embedding(len(self.tree), emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, emb_dim, num_layers, batch_first=True)

    def forward(self, input_src, input_embeddings):
        '''
        input_src : tensor (batch, seq)
        input_embeddings : tensor cuda (batch, seq , dim)
        '''
        input_reshape = input_src.view(-1).numpy() # (batch * seq , 1)
        input_embeddings_reshape = input_embeddings.view(input_embeddings.size(0) * input_embeddings.size(1), input_embeddings.size(2)).unsqueeze(1) # (batch*seq, 1, dim)
        parent_ls = []
        for i in input_reshape:
            parents_idx = self.tree.getLeafAns(i)
            parent_ls.append(parents_idx)
        parent_tensor = torch.LongTensor(parent_ls).to(self.device) # (batch*seq, 2)
        parent_embeddings = self.embed(parent_tensor) # (batch*seq, 2, dim)
        combined_embeddings = torch.cat([parent_embeddings, input_embeddings_reshape], dim=1) #(batch*seq, 3, dim)
        outputs, _ = self.lstm(combined_embeddings) # outputs (batch*seq, 3, dim)
        outputs = outputs[:,-1,:] # (batch*seq, dim)
        outputs = outputs.view(*input_embeddings.size())
        return outputs




        




