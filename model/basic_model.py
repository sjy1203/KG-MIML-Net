import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# class Tree(object):
#     def __init__(self):
#         root = Node()
#         self.root = root
#         flatten_tree = defaultdict(list)
#         flatten_tree[root] 
#         self.flatten_tree = flatten_tree


#     def add_node(self, node, depth): 
#
class OutputsTreeEmbedding(nn.Module):
    def __init__(self, tree, voc, emb_dim):
        super(OutputsTreeEmbedding, self).__init__()
        self.pad_token_idx = 0
        self.start_token_idx = 1
        self.end_token_idx = 2
        self.tree = tree
        self.voc = voc
        self.emb_dim = emb_dim
        self.embed = nn.Embedding(len(self.voc), emb_dim, padding_idx=self.pad_token_idx)
        self.lstm = nn.LSTM(emb_dim, emb_dim, bidirectional=True, batch_first=True)
        level2_set = set()
        level3_set = set()
        level4_set = set()
        for i in range(3, len(self.voc)):
            level1_node = self.tree[self.voc.idx2word[i]]

            level2_node = level1_node.parent
            level3_node = level2_node.parent
            level4_node = level3_node.parent

            level2_set.add(level2_node)
            level3_set.add(level3_node)
            level4_set.add(level4_node)

        self.level2_set = level2_set
        self.level3_set = level3_set
        self.level4_set = level4_set

    def forward(self, input_src):
        self._update_tree_embedding()
        input_src_flatten  = [i for lst in input_src for i in lst ]
        tree_embeddings = []
        for i in input_src_flatten:
            tree_embeddings.append(self._get_tree_embeddings(int(i)))
        tree_embeddings = torch.stack(tree_embeddings, dim=0)
        _,(h_t,_) = self.lstm(tree_embeddings)
        leaf_embedding = h_t.permute(1,0,2)
        leaf_embedding = leaf_embedding.contiguous().view(leaf_embedding.size(0),-1)
        
        return leaf_embedding.view(input_src.size(0), input_src.size(1), leaf_embedding.size(1))        

    def _get_tree_embeddings(self, leaf_idx):
        if leaf_idx == self.pad_token_idx:
            return torch.zeros((5, self.emb_dim)).cuda()
        if leaf_idx == self.start_token_idx:
            return self.embed(torch.LongTensor([1]*5).cuda()).squeeze(dim=0)
        if leaf_idx == self.end_token_idx:
            return self.embed(torch.LongTensor([2]*5).cuda()).squeeze(dim=0)

        embeddings = []
        level1_node = self.tree[self.voc.idx2word[leaf_idx]]
        level2_node = level1_node.parent
        level3_node = level2_node.parent
        level4_node = level3_node.parent
        root_node = self.tree['<root>']
        embeddings.append(level1_node.embedding)
        embeddings.append(level2_node.embedding)
        embeddings.append(level3_node.embedding)
        embeddings.append(level4_node.embedding)
        embeddings.append(root_node.embedding)

        return torch.stack(embeddings,dim=0)


    def _update_tree_embedding(self):
        embeddings = self.embed(torch.LongTensor(range(0, len(self.voc))).view(1,-1).cuda()).squeeze(0) # (len(voc), emb_dim)

        for i in range(3, len(self.voc)):
            level1_node = self.tree[self.voc.idx2word[i]]
            level1_node.update_embedding(embeddings[i,:])

        for i in self.level2_set:
            i.update_embedding()
        
        for i in self.level3_set:
            i.update_embedding()

        for i in self.level4_set:
            i.update_embedding()

        self.tree['<root>'].update_embedding()
                

class InputsTreeEmbedding(nn.Module):
    def __init__(self, tree, voc, emb_dim):
        super(InputsTreeEmbedding, self).__init__()
        self.pad_token_idx = 0
        self.tree = tree
        self.voc = voc
        self.emb_dim = emb_dim
        self.embed = nn.Embedding(len(self.voc), emb_dim, padding_idx=self.pad_token_idx)
        self.lstm = nn.LSTM(emb_dim, emb_dim, bidirectional=True, batch_first=True)
        
        level2_set = set()
        level3_set = set()
        for i in range(1, len(self.voc)):
            level1_node = self.tree[self.voc.idx2word[i]]

            level2_node = level1_node.parent
            level3_node = level2_node.parent
            level2_set.add(level2_node)
            level3_set.add(level3_node)

        self.level2_set = level2_set
        self.level3_set = level3_set

    def forward(self, input_src):
        self._update_tree_embedding()
        input_src_flatten  = [i for lst in input_src for i in lst]
        tree_embeddings = []
        for i in input_src_flatten:
            tree_embeddings.append(self._get_tree_embeddings(int(i)))
        tree_embeddings = torch.stack(tree_embeddings, dim=0)
        _,(h_t,_) = self.lstm(tree_embeddings)
        leaf_embedding = h_t.permute(1,0,2)
        leaf_embedding = leaf_embedding.contiguous().view(leaf_embedding.size(0),-1)
        return leaf_embedding.view(input_src.size(0), input_src.size(1), leaf_embedding.size(1))

    def _get_tree_embeddings(self, leaf_idx):
        if leaf_idx == self.pad_token_idx:
            return torch.zeros((4, self.emb_dim)).cuda()
        embeddings = []
        level1_node = self.tree[self.voc.idx2word[leaf_idx]]
        level2_node = level1_node.parent
        level3_node = level2_node.parent
        root_node = self.tree['<root>']
        embeddings.append(level1_node.embedding)
        embeddings.append(level2_node.embedding)
        embeddings.append(level3_node.embedding)
        embeddings.append(root_node.embedding)

        return torch.stack(embeddings,dim=0)


    def _update_tree_embedding(self):
        embeddings = self.embed(torch.LongTensor(range(0, len(self.voc))).view(1,-1).cuda()).squeeze(0) # (len(voc), emb_dim)       

        for i in range(1, len(self.voc)):
            level1_node = self.tree[self.voc.idx2word[i]]
            level1_node.update_embedding(embeddings[i,:])

        for i in self.level2_set:
            i.update_embedding()
        
        for i in self.level3_set:
            i.update_embedding()

        self.tree['<root>'].update_embedding()



class TreeWeightedEmbedding(nn.Module):
    def __init__(self, input_voc, emb_dim, num_layers):
        super(TreeWeightedEmbedding, self).__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim

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
        parent_tensor = torch.LongTensor(parent_ls).cuda() # (batch*seq, 2)
        parent_embeddings = self.embed(parent_tensor) # (batch*seq, 2, dim)
        combined_embeddings = torch.cat([parent_embeddings, input_embeddings_reshape], dim=1) #(batch*seq, 3, dim)
        outputs, _ = self.lstm(combined_embeddings) # outputs (batch*seq, 3, dim)
        outputs = outputs[:,-1,:] # (batch*seq, dim)
        outputs = outputs.view(*input_embeddings.size())
        return outputs

class SupervisedAttention(nn.Module):
    def __init__(self):
        super(SupervisedAttention, self).__init__()

    def forward(self, ctx, ctx_lens=None):
        '''
        ctx (batch, seq, dim)
        '''
        max_len = max(ctx_lens)
        attn = [ list(range(i,0,-1)) + [-5]*(max_len-i) for i in ctx_lens]
        attn = torch.FloatTensor(attn).cuda()
        attn = F.softmax(attn, dim=-1).unsqueeze(dim=1) # (batch, 1,seq)
        attn_ctx = torch.bmm(attn, ctx).squeeze(1) # (batch, dim)

        return attn_ctx
        

class SoftDotAttention(nn.Module):
    """Soft Dot Attention.
    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=-1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context):
        """Propogate input through the network.
        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn

class LSTMAttentionDot(nn.Module):
    """A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, batch_first=True):
        """Initialize params."""
        super(LSTMAttentionDot, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.batch_first = batch_first

        self.input_weights = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_weights = nn.Linear(hidden_size, 4 * hidden_size)

        self.attention_layer = SoftDotAttention(hidden_size)

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""
        def recurrence(input, hidden):
            """Recurrence helper."""
            hx, cx = hidden  # n_b x hidden_dim
            gates = self.input_weights(input) + \
                self.hidden_weights(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate = F.sigmoid(ingate)
            forgetgate = F.sigmoid(forgetgate)
            cellgate = F.tanh(cellgate)
            outgate = F.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            h_tilde, alpha = self.attention_layer(hy, ctx.transpose(0, 1))

            return h_tilde, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            output.append(hidden[0])

        output = torch.stack(output, 0)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden
