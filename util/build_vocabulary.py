import pickle
from collections import defaultdict
import numpy as np
import torch

level2 = ['001-009', '010-018', '020-027', '030-041', '042', '045-049', '050-059', '060-066', '070-079', '080-088', '090-099', '100-104', '110-118', '120-129', '130-136', '137-139', '140-149', '150-159', '160-165', '170-176', '176', '179-189', '190-199', '200-208', '209', '210-229', '230-234', '235-238', '239', '240-246', '249-259', '260-269', '270-279', '280-289', '290-294', '295-299', '300-316', '317-319', '320-327', '330-337', '338', '339', '340-349', '350-359', '360-379', '380-389', '390-392', '393-398', '401-405', '410-414', '415-417', '420-429', '430-438', '440-449', '451-459', '460-466', '470-478', '480-488', '490-496', '500-508', '510-519', '520-529', '530-539', '540-543', '550-553', '555-558', '560-569', '570-579', '580-589', '590-599', '600-608', '610-611', '614-616', '617-629', '630-639', '640-649', '650-659', '660-669', '670-677', '678-679', '680-686', '690-698', '700-709', '710-719', '720-724', '725-729', '730-739', '740-759', '760-763', '764-779', '780-789', '790-796', '797-799', '800-804', '805-809', '810-819', '820-829', '830-839', '840-848', '850-854', '860-869', '870-879', '880-887', '890-897', '900-904', '905-909', '910-919', '920-924', '925-929', '930-939', '940-949', '950-957', '958-959', '960-979', '980-989', '990-995', '996-999', 'V01-V91', 'V01-V09', 'V10-V19', 'V20-V29', 'V30-V39', 'V40-V49', 'V50-V59', 'V60-V69', 'V70-V82', 'V83-V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'E000-E899', 'E000', 'E001-E030', 'E800-E807', 'E810-E819', 'E820-E825', 'E826-E829', 'E830-E838', 'E840-E845', 'E846-E849', 'E850-E858', 'E860-E869', 'E870-E876', 'E878-E879', 'E880-E888', 'E890-E899', 'E900-E909', 'E910-E915', 'E916-E928', 'E929', 'E930-E949', 'E950-E959', 'E960-E969', 'E970-E978', 'E980-E989', 'E990-E999']

def expand_level2():
    level2_expand = {}
    for i in level2:
        tokens = i.split('-')            
        if i[0] == 'V':
            if len(tokens)==1:
                level2_expand[i] = i
            else:
                for j in range(int(tokens[0][1:]), int(tokens[1][1:])+1):
                    level2_expand["V%02d" % j] = i                
        elif i[0] == 'E':
            if len(tokens)==1:
                level2_expand[i] = i
            else:
                for j in range(int(tokens[0][1:]), int(tokens[1][1:])+1):
                    level2_expand["E%03d" % j] = i                                
        else:
            if len(tokens)==1:
                level2_expand[i] = i    
            else: 
                for j in range(int(tokens[0]), int(tokens[1])+1):
                    level2_expand["%03d" % j] = i
    return level2_expand

class Node(object):
    def __init__(self, idx=-1, word='<pad>', embedding=None, parent=None, childs=None):
        self.idx = idx
        self.word = word
        self.embedding = embedding
        self.parent = parent
        self.childs = childs

    def __eq__(self, other):
        return self.word == other.word

    def __str__(self):
        return self.word

    def __hash__(self):
        return hash(self.word)
   
    def add_child(self, node):
        if self.childs is None:
            self.childs = []
        if not node in self.childs:
            self.childs.append(node)

    def update_embedding(self, embedding=None):
        if self.childs is None:
            # leaf node
            if not embedding is None:
                self.embedding = embedding
        else:
            # parent node
            child_embedding = torch.stack([child.embedding for child in self.childs], dim=0)
            self.embedding = torch.mean(child_embedding, dim=0)

def build_labels_tree(voc):
    tree_dict = defaultdict(Node)
    for i in range(3, len(voc)):
        level1_name = voc.idx2word[i]
        level2_name = level1_name[:4]
        level3_name = level1_name[:3]
        level4_name = level1_name[:1]
        level5_name = '<root>'

        level1_node = Node(idx=i, word=level1_name)
        level2_node = tree_dict[level2_name]
        level3_node = tree_dict[level3_name] 
        level4_node = tree_dict[level4_name] 
        level5_node = tree_dict[level5_name]

        if level2_node.idx == -1:
            level2_node = Node(idx=0, word=level2_name)
        if level3_node.idx == -1:
            level3_node = Node(idx=0, word=level3_name)
        if level4_node.idx == -1:
            level4_node = Node(idx=0, word=level4_name) 
        if level5_node.idx == -1:
            level5_node = Node(idx=0, word=level5_name) 

        level1_node.parent = level2_node
        level2_node.parent = level3_node
        level3_node.parent = level4_node
        level4_node.parent = level3_node

        level2_node.add_child(level1_node)
        level3_node.add_child(level2_node)
        level4_node.add_child(level3_node)
        level5_node.add_child(level4_node)

        tree_dict[level1_name] = level1_node
        tree_dict[level2_name] = level2_node
        tree_dict[level3_name] = level3_node
        tree_dict[level4_name] = level4_node
        tree_dict[level5_name] = level5_node
    return tree_dict        

def build_instances_tree(voc):
    tree_dict = defaultdict(Node)
    level3_map = expand_level2()

    for i in range(1, len(voc)):
        level1 = voc.idx2word[i]
        level2 = level1[:4] if level1[0] == 'E' else level1[:3]
        level3 = level3_map[level2]
        level4 = '<root>'

        level1_name = level1
        level2_name = '{}_{}'.format(level2, 2)
        level3_name = '{}_{}'.format(level3, 3)
        level4_name = level4

        level1_node = Node(idx=i, word=level1_name)
        level2_node = tree_dict[level2_name]
        level3_node = tree_dict[level3_name] 
        level4_node = tree_dict[level4_name]

        if level2_node.idx == -1:
            level2_node = Node(idx=0, word=level2_name)
        if level3_node.idx == -1:
            level3_node = Node(idx=0, word=level3_name)
        if level4_node.idx == -1:
            level4_node = Node(idx=0, word=level4_name)
        
        level1_node.parent = level2_node
        level2_node.parent = level3_node
        level3_node.parent = level4_node

        level2_node.add_child(level1_node)
        level3_node.add_child(level2_node)
        level4_node.add_child(level3_node)

        tree_dict[level1_name] = level1_node
        tree_dict[level2_name] = level2_node
        tree_dict[level3_name] = level3_node
        tree_dict[level4_name] = level4_node
    return tree_dict

class TreeVocabulary(object):
    def __init__(self, leaf_voc):
        self.leaf_voc = leaf_voc
        self.root = '<root>'
        self.map = expand_level2()
        self.map['<pad>'] = '<pad>'

        tree_voc = Vocabulary()
        tree_voc.add_word('<pad>')
        tree_voc.add_word(self.root)

        for i in level2:
            tree_voc.add_word(i)

        for i in list(leaf_voc.word2idx.keys()):
            if i[0] == 'E':
                tree_voc.add_word(i[:4])
            else:
                tree_voc.add_word(i[:3])
        self.tree_voc = tree_voc
        
    def getLeafAns(self, idx):
        if idx == self.leaf_voc('<pad>'):
            return [idx] * 3
        leaf_word = self.leaf_voc.idx2word[idx]
        parent_of_leaf = self.tree_voc(leaf_word[:4] if leaf_word[0]=='E' else leaf_word[:3])
        pp_of_leaf = self.tree_voc(self.map[self.tree_voc.idx2word[parent_of_leaf]])
        return [self.tree_voc('<root>'), pp_of_leaf, parent_of_leaf]   

    def __len__(self):
        return len(self.tree_voc)

# class L3_Tree(object):
#     def __init__(self, child_voc):
#         self.root = '<root>'
#         self.root_idx = 1

#         parent_voc = Vocabulary()
#         parent_voc.add_word('<pad>')
#         parent_voc.add_word(self.root)
#         for i in level2:
#             parent_voc.add_word(i)
#         self.parent_voc = parent_voc
#         self.map = expand_level2()
#         self.map['<pad>'] = '<pad>'
        
#         self.child_voc = child_voc
    
#     def get_parent_idx(self, idx):
#         child_word = self.child_voc.idx2word[idx]
#         parent_word = self.map[child_word]
#         parent_idx = self.parent_voc(parent_word)
#         return parent_idx
    
class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __call__(self, word):
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_normal_vocab(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    vocab = Vocabulary()
    
    for i in data:
        vocab.add_word(i)

    return vocab    

def build_vocab(path, add_start_end=False):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    if add_start_end:
        vocab.add_word('<start>')
        vocab.add_word('<end>')
    
    for i in data:
        vocab.add_word(i)

    return vocab

    