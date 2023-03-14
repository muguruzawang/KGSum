import sys 
sys.path.append("..")

import torch
import dgl
import numpy as np
import json
import pickle
import random
from itertools import combinations
from nltk.corpus import stopwords
from collections import Counter
from tools.logger import *
from utils.logging import init_logger, logger
import nltk
from module import vocabulary
from module import data
from torch.autograd import Variable

import time

import pdb

NODE_TYPE = {'word':0, 'sentence': 1, 'doc':2, 'entity': 3, 'relation':4, 'type':5 ,'root':6 }
FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']
FILTERWORD.extend(punctuations)

### in this version, we construct graph with three types of nodes: sentence, entity, relation.

def load_to_cuda(batch,device):
    batch = {'ent_text': batch['ent_text'].to(device,non_blocking=True), 'rel': batch['rel'].to(device,non_blocking=True),  \
             'text': batch['text'].to(device,non_blocking=True), 'text_extend': batch['text_extend'].to(device,non_blocking=True), \
             'graph': batch['graph'], 'raw_ent_text': batch['raw_ent_text'], 'raw_sent_input': batch['raw_sent_input'], 'ent_num':batch['ent_num'], \
             'raw_tgt_text': batch['raw_tgt_text'], 'examples':batch['examples'], 'tgt': batch['tgt'].to(device, non_blocking=True), \
             'sent_num':batch['sent_num'],  'edges':batch['edges'].to(device, non_blocking=True), 'extra_zeros':batch['extra_zeros'], \
             'article_oovs':batch['article_oovs'],'tgt_extend': batch['tgt_extend'].to(device, non_blocking=True), \
             'ent_text_extend': batch['ent_text_extend'].to(device,non_blocking=True), 'raw_temp':batch['raw_temp'],\
             'ent_score': batch['ent_score'].to(device,non_blocking=True),\
             'template_input':batch['template_input'].to(device,non_blocking=True),\
             'template_target':batch['template_target'].to(device,non_blocking=True),\
             'enttypes':batch['enttypes'].to(device,non_blocking=True),\
             'type':batch['type'].to(device,non_blocking=True)}
    batch['extra_zeros'] = batch['extra_zeros'].to(device, non_blocking=True) if batch['extra_zeros'] != None else batch['extra_zeros']
    
    return batch 

def readJson(fname):
    data = []
    with open(fname, 'r',encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def readText(fname):
    data = []
    with open(fname, 'r', encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data

def write_txt(batch, seqs, w_file, args):
    # converting the prediction to real text.
    ret = []
    for b, seq in enumerate(seqs):
        txt = []
        for token in seq:
            if int(token) not in [args.wordvocab.word2id(x) for x in ['<PAD>', '<BOS>', '<EOS>']]:
                txt.append(args.wordvocab.id2word(int(token)))
            if int(token) == args.wordvocab.word2id('<EOS>'):
                break
        w_file.write(' '.join([str(x) for x in txt])+'\n')
        ret.append([' '.join([str(x) for x in txt])])
    return ret 


def replace_ent(x, ent, V):
    # replace the entity
    mask = x>=V
    if mask.sum()==0:
        return x
    nz = mask.nonzero()
    fill_ent = ent[nz, x[mask]-V]
    x = x.masked_scatter(mask, fill_ent)
    return x


###这是建立长度为lens的mask矩阵的操作
def len2mask(lens, device):
    #得到最大的长度n
    max_len = max(lens)
    #构造维度为[len(lens),maxlen]的矩阵
    mask = torch.arange(max_len, device=device).unsqueeze(0).expand(len(lens), max_len)

    ####最终会得到类似[[ 0, 0, 0，1，1],
        #[0, 0, 1, 1, 1],
        #[0, 1, 1, 1, 1]]的矩阵
    #作者这里用0来表示实际的单词，用1来填充
    mask = mask >= torch.LongTensor(lens).to(mask).unsqueeze(1)
    return mask


### for roberta, use pad_id = 1 to pad tensors.
def pad(var_len_list, out_type='list', flatten=False):
    if flatten:
        lens = [len(x) for x in var_len_list]
        var_len_list = sum(var_len_list, [])
    max_len = max([len(x) for x in var_len_list])
    if out_type=='list':
        if flatten:
            return [x+['<PAD>']*(max_len-len(x)) for x in var_len_list], lens
        else:
            return [x+['<PAD>']*(max_len-len(x)) for x in var_len_list]
    if out_type=='tensor':
        if flatten:
            return torch.stack([torch.cat([x, \
            torch.ones([max_len-len(x)]+list(x.shape[1:])).type_as(x)], 0) for x in var_len_list], 0), lens
        else:
            return torch.stack([torch.cat([x, \
            torch.ones([max_len-len(x)]+list(x.shape[1:])).type_as(x)], 0) for x in var_len_list], 0)

def pad_sent_entity(var_len_list, pad_id,bos_id,eos_id, flatten=False):
    def _pad_(data,height,width,pad_id,bos_id,eos_id):
        rtn_data = []
        for para in data:
            if torch.is_tensor(para):
                para = para.numpy().tolist()
            if len(para) > width:
                para = para[:width]
            else:
                para += [pad_id] * (width - len(para))
            rtn_data.append(para)
        rtn_length = [len(para) for para in data]
        x = []
        '''
        x.append(bos_id)
        x.append(eos_id)
        '''
        x.extend([pad_id] * (width))
        rtn_data = rtn_data + [x] * (height - len(data))
        # rtn_data = rtn_data + [[pad_id] * width] * (height - len(data))
        rtn_length = rtn_length + [0] * (height - len(data))
        if len(rtn_data) == 0:
            rtn_data.append([])
        return rtn_data, rtn_length
    
    if flatten:
        var_len = [len(x) for x in var_len_list]
        max_nsent = max(var_len)
        max_ntoken = max([max([len(p) for p in x]) for x in var_len_list])
        _pad_var_list = [_pad_(ex, max_nsent, max_ntoken, pad_id, bos_id, eos_id) for ex in var_len_list]
        pad_var_list = torch.stack([torch.tensor(e[0]) for e in _pad_var_list])
        return pad_var_list, var_len

    else:
        max_nsent = len(var_len_list)
        max_ntoken = max([len(x) for x in var_len_list])
        
        _pad_var_list = _pad_(var_len_list, max_nsent,max_ntoken, pad_id, bos_id, eos_id)
        pad_var_list = torch.tensor(_pad_var_list[0]).transpose(0, 1)
        return pad_var_list

def pad_edges(batch_example):
    max_nsent = max([len(ex.raw_sent_input) for ex in batch_example])
    max_nent = max([len(ex.raw_ent_text) for ex in batch_example])
    edges = torch.zeros(len(batch_example),max_nsent,max_nent)
    for index,ex in enumerate(batch_example):
        for key in ex.entities:
            if int(key) >= ex.doc_max_len:
                break
            if ex.entities[key] != []:
                for x in ex.entities[key]:
                    #e = at_least(x.lower().split())
                    e = at_least(x.lower())
                    entNo = ex.raw_ent_text.index(e)
                    sentNo = int(key)

                    edges[index][sentNo][entNo] = 1
    return edges

def at_least(x):
    # handling the illegal data
    if len(x) == 0:
        return ['<UNK>']
    else:
        return x

class Example(object):
    def __init__(self, text, label, summary,template, entities, relations, types, \
        clusters, entscore, sent_max_len, doc_max_len, wordvocab,  rel_vocab, type_vocab):
        #data format is as follows:
        # text: [[],[],[]] list(list(string)) for multi-document; one per article sentence. each token is separated by a single space
        # entities: {"0":[],"1":[]...}, a dict correponding to all the sentences, one list per sentence
        # relations: list(list()) the inner list correspongding to a 3-element tuple, [ent1, relation, ent2]
        # types: list  one type per entity
        # clusters: list(list) the inner list is the set of all the co-reference lists

        # filterwords are only used in graph building process
        # all the text words should be in the range of word_vocab, or it will be [UNK]

        self.wordvocab = wordvocab
        self.type_vocab = type_vocab
        start_decoding = wordvocab.word2id(vocabulary.START_DECODING)
        stop_decoding = wordvocab.word2id(vocabulary.STOP_DECODING)

        self.sent_max_len = sent_max_len
        self.doc_max_len = doc_max_len

        self.text = text
        self.summary = ' '.join(summary).lower()
        abstract_words = self.summary.split() # list of strings
        abs_ids = [wordvocab.word2id(w) for w in abstract_words] # list of word ids; OOVs are represented by the id for UNK token

        #self.template = ' '.join(template).lower()
        self.template = template.lower()
        template_words = self.template.split() # list of strings
        template_ids = [wordvocab.word2id(w) for w in template_words] # list of word ids; OOVs are represented by the id for UNK token

        self.entities = entities
        self.raw_sent_len = []
        self.raw_sent_input = []
        if isinstance(text, list) and isinstance(text[0], list):
            self.original_article_sents = []
            for doc in text:
                self.original_article_sents.extend(doc)
        self.original_abstract = "\n".join(summary)

        self.enc_sent_input = []
        self.raw_sent_input = []
        for sent in self.original_article_sents:
            article_words = sent.lower().split()
            if len(article_words) > sent_max_len:
                article_words = article_words[:sent_max_len]
            self.raw_sent_input.append(article_words)
            self.enc_sent_input.append([wordvocab.word2id(w) for w in article_words]) # list of word ids; OOVs are represented by the id for UNK token
            if len(self.raw_sent_input) >= self.doc_max_len:
                break
        
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, 500, start_decoding, stop_decoding)
        self.template_input, self.template_target = self.get_dec_inp_targ_seqs(template_ids, 500, start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)
        
        self.enc_sent_input_extend = []
        self.article_oovs = []
    
        ###add pointer-generator mode
        for enc_sent in self.raw_sent_input:
            enc_input_extend_vocab, self.article_oovs = data.article2ids(enc_sent, wordvocab, self.article_oovs)
            #self.article_oovs.extend(oovs)
            self.enc_sent_input_extend.append(enc_input_extend_vocab)
        
        # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
        abs_ids_extend_vocab = data.abstract2ids(abstract_words, wordvocab, self.article_oovs)
        template_extend_vocab = data.abstract2ids(template_words, wordvocab, self.article_oovs)

        # Overwrite decoder target sequence so it uses the temp article OOV ids
        _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, 500, start_decoding, stop_decoding)
        _, self.template_target = self.get_dec_inp_targ_seqs(template_extend_vocab, 500, start_decoding, stop_decoding)

        ###  truncate doc length to threshold
        '''
        if len(self.original_article_sents) >= self.doc_max_len:
            self.raw_sent_input = self.original_article_sents[:self.doc_max_len]
        else:
            self.raw_sent_input = self.original_article_sents
        ''' 
        # Store the label
        self.label = label
        label_shape = (len(self.original_article_sents), len(label))  # [N, len(label)]

        self.label_matrix = np.zeros(label_shape, dtype=int)
        if label != []:
            self.label_matrix[np.array(label), np.arange(len(label))] = 1  # label_matrix[i][j]=1 indicate the i-th sent will be selected in j-th step
        
        #assert min([len(x) for x in self.raw_ent_text])>0, str(self.raw_ent_text)
        for i in range(types.count([])):
            types.remove([])

        self.raw_ent_type = types # <method> .. <>
        self.raw_ent_text = []
        self.ent_enc_input = []
        self.ent_type = []
        index = -1
        for key in entities:
            if int(key) >= self.doc_max_len:
                break
            for x in entities[key]:
                '''
                e = at_least(x.lower().split())
                if e not in self.raw_ent_text:
                    self.raw_ent_text.append(e)
                '''
                e = at_least(x.lower())
                if e not in self.raw_ent_text:
                    self.raw_ent_text.append(e)
                    e_split = e.split()
                    e_enc = [wordvocab.word2id(w) for w in e_split]
                    self.ent_enc_input.append(e_enc)

                    index += 1
                    thetype = self.raw_ent_type[index].lower()
                    typeid = wordvocab.word2id('<'+thetype+'>')
                    self.ent_type.append([typeid]*len(e_split))
        
        self.ent_score = []

        self.enc_ent_input_extend = []

        for enc_ent in self.raw_ent_text:
            enc_entsplit = enc_ent.split()
            self.ent_score.append([entscore[enc_ent]]*len(enc_entsplit))
            enc_ent_extend = data.abstract2ids(enc_entsplit, wordvocab, self.article_oovs)
            #self.article_oovs.extend(oovs)
            self.enc_ent_input_extend.append(enc_ent_extend)

                    
        self.raw_rel = [] 
        for r in relations:
            if len(r) != 3:
                continue
            ent1 = r[0].lower()
            ent2 = r[2].lower()
            if ent1 in self.raw_ent_text and ent2 in self.raw_ent_text:
                self.raw_rel.append([ent1, r[1], ent2])

        for cluster in clusters:
            cluster = list(set(cluster))
            combs = list(combinations(cluster,2))
            for comb in combs:
                comb1 = comb[0].lower()
                comb2 = comb[1].lower()
                if comb1 in self.raw_ent_text and comb2 in self.raw_ent_text: 
                    self.raw_rel.append([comb1, 'Coreference', comb2])
        
        self.types = ['Task', 'Method', 'Metric', 'Material', 'OtherScientificTerm', 'Generic']

    def __str__(self):
        return '\n'.join([str(k)+':\t'+str(v) for k, v in self.__dict__.items()])

    def __len__(self):
        return len(self.raw_text)
    
    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len: # truncate
            inp = inp[:max_len]
            target = target[:max_len] # no end_token
        else: # no truncation
            target.append(stop_id) # end token
        assert len(inp) == len(target)
        return inp, target

class ExampleSet(torch.utils.data.Dataset):
    def __init__(self, text_path, ert_path, template_path, entscore_path, 
                    wordvocab, rel_vocab, type_vocab,sent_max_len, doc_max_len, device=None):
        super(ExampleSet, self).__init__()
        self.device = device
        self.rel_vocab = rel_vocab
        self.wordvocab = wordvocab
        self.type_vocab = type_vocab

        self.sent_max_len = sent_max_len
        self.doc_max_len = doc_max_len

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        self.json_text_list = readJson(text_path) ###将训练数据读出来（text: , summary: ,label:）
        self.json_ert_list = readJson(ert_path)
        self.json_template = readJson(template_path)
        self.entscore_list = readJson(entscore_path)
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.json_text_list))
        self.size = len(self.json_text_list) ###训练集的大小
        
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.json_text_list))
        self.size = len(self.json_text_list) ###训练集的大小

    def get_example(self, index):
        json_text = self.json_text_list[index]
        json_entity = self.json_ert_list[index]
        json_entscore = self.entscore_list[index]
        json_template = self.json_template[index]
        #e["summary"] = e.setdefault("summary", [])
        example = Example(json_text['text'],json_text['label'], json_text['summary'],json_template['summary'], json_entity['entities'],json_entity['relations'], \
                json_entity['types'],json_entity['clusters'], json_entscore, self.sent_max_len, \
                self.doc_max_len, self.wordvocab, self.rel_vocab, self.type_vocab)
        return example
    
    def pad_label_m(self, label_matrix):
        label_m = label_matrix[:self.doc_max_len, :self.doc_max_len]
        N, m = label_m.shape
        if m < self.doc_max_len:
            pad_m = np.zeros((N, self.doc_max_len - m))
            return np.hstack([label_m, pad_m])
        return label_m

    def __getitem__(self, index):
        item = self.get_example(index)
        #enc_sent_input_pad是包含所有经过pad后的句子列表，这一步是对句子进行裁剪，只取前max个句子
        graph = self.build_graph(item)
        if len(item.raw_ent_text) != 0:
            ex_data = self.get_tensor(item)
        else:
            ex_data = {}
        return graph, ex_data

    def __len__(self):
        return self.size
    
    def MapEnt2Sent(self,ex):
        ent2sent = {}
        
        for key in ex.entities:
            if int(key) >= self.doc_max_len:
                break
            if ex.entities[key] != []:
                for x in ex.entities[key]:
                    #e = at_least(x.lower().split())
                    e = at_least(x.lower())
                    entNo = ex.raw_ent_text.index(e)
                    sentNo = int(key)

                    if entNo not in ent2sent:
                        ent2sent[entNo] = [sentNo]
                    else:
                        ent2sent[entNo].append(sentNo)
            
        return ent2sent
    
    def entity_edge_sent(self, G, sent2nid, ent2sent, ent2nid,ex):
        for i in range(len(ex.raw_ent_text)):
            ent_nid = ent2nid[i]
            if i in ent2sent.keys():
                for nid in ent2sent[i]:
                    sent_nid = sent2nid[nid]
                    G.add_edges(ent_nid, sent_nid)
                    G.add_edges(sent_nid, ent_nid)
        return G 
            
    def entity_edge_rel(self, G, ent2nid, rel2nid, ex):
        for i, (ent1,rel,ent2) in enumerate(ex.raw_rel):
            ent1id, ent2id = ex.raw_ent_text.index(ent1), ex.raw_ent_text.index(ent2)
            ent1_nid, ent2_nid = ent2nid[ent1id], ent2nid[ent2id]
            rel1_nid, rel2_nid = rel2nid[i], rel2nid[i]+1

            G.add_edges(ent1_nid, rel1_nid)
            G.add_edges(rel1_nid, ent2_nid)
            G.add_edges(ent2_nid, rel2_nid)
            G.add_edges(rel2_nid, ent1_nid)
        return G


    def entity_edge_type(self, G, ent2nid, type2nid, ex):
        index = -1
        for key in ex.entities:
            if int(key) >= self.doc_max_len:
                break
            if ex.entities[key] != []:
                for x in ex.entities[key]:
                    e = at_least(x.lower())
                    entNo = ex.raw_ent_text.index(e)
                    ent_nid = ent2nid[entNo]
                    index += 1
                    thetype = ex.raw_ent_type[index]
                    type_id = ex.types.index(thetype)
                    type_nid = type2nid[type_id]

                    # src,dst are both torch.tensor()
                    src,dst = G.edges()
                    if type_nid not in dst.numpy()[np.argwhere(src.numpy() == ent_nid)]:
                        G.add_edges(ent_nid, type_nid)
                        G.add_edges(type_nid, ent_nid)   
        
        return G

    ###这里是建立graph的步骤
    def build_graph(self,ex):
        graph = dgl.DGLGraph()
        #graph = dgl.graph()

        graph.set_n_initializer(dgl.init.zero_initializer)

        s_nodes = len(ex.raw_sent_input)

        label = self.pad_label_m(ex.label_matrix)
        graph.add_nodes(s_nodes, {'type': torch.ones(s_nodes) * NODE_TYPE['sentence'], 'label': torch.LongTensor(label)})
        
        sent2nid = [i for i in range(s_nodes)]

        ########################################################################################
        # the above code add word , sentence, doc nodes to graph
        # the following code will add entity , relation, type nodes to graph

        ent_len = len(ex.raw_ent_text)
        rel_len = len(ex.raw_rel) # treat the repeated relation as different nodes, refer to the author's code

        graph.add_nodes(ent_len, {'type': torch.ones(ent_len) * NODE_TYPE['entity']})
        graph.add_nodes(1, {'type': torch.ones(1) * NODE_TYPE['root']})
        graph.add_nodes(rel_len*2, {'type': torch.ones(rel_len*2) * NODE_TYPE['relation']})
        graph.add_nodes(len(ex.types), {'type': torch.ones(len(ex.types)) * NODE_TYPE['type']})

        ent2nid = [i + s_nodes for i in range(ent_len)]
        root2nid = s_nodes + ent_len
        rel2nid = [i + s_nodes + ent_len + 1 for i in range(rel_len*2)]
        type2nid = [i + s_nodes + ent_len + rel_len*2 + 1 for i in range(len(ex.types))]

        ########################################################################################
        # now we add edges bwtween nodes, including the following types of edges
        # word2sent, sent2word
        # sent2doc, doc2sent
        # entity2sent, sent2entity
        # entity2rel, rel2entity
        # entity2type,  type2entity

        #pdb.set_trace()
        ent2sent = self.MapEnt2Sent(ex)
        graph = self.entity_edge_sent(graph, sent2nid, ent2sent, ent2nid,ex)
        graph = self.entity_edge_rel(graph, ent2nid, rel2nid,ex)
        graph = self.entity_edge_type(graph, ent2nid, type2nid,ex)

        
        #将root节点和其他节点连接
        graph.add_edges(root2nid, torch.arange(root2nid))
        #反向继续连接
        graph.add_edges(torch.arange(root2nid), root2nid)
        

        return graph

    def get_tensor(self, ex):
        
        rel_data = ['--root--'] + sum([[x[1],x[1]+'_INV'] for x in ex.raw_rel], [])
        rel = [self.wordvocab.word2id('<'+x.lower()+'>') for x in rel_data]
        type_ = [self.wordvocab.word2id('<'+x.lower()+'>') for x in ex.types]
        
        _cached_tensor = {'ent_text': [torch.LongTensor(x) for x in ex.ent_enc_input],  'rel': torch.LongTensor(rel), \
                            'text': [torch.LongTensor(x) for x in ex.enc_sent_input], 'text_extend':[torch.LongTensor(x) for x in ex.enc_sent_input_extend], \
                            'raw_ent_text': ex.raw_ent_text, 'summary':ex.summary, 'tgt':torch.LongTensor(ex.dec_input), \
                            'raw_sent_input': ex.raw_sent_input, 'example':ex, 'tgt_extend': torch.LongTensor(ex.target), 'oovs':ex.article_oovs, \
                            'ent_text_extend': [torch.LongTensor(x) for x in ex.enc_ent_input_extend],'type': torch.LongTensor(type_), \
                            'ent_score':[torch.FloatTensor(x) for x in ex.ent_score],'raw_temp':ex.template,\
                            'template_input':torch.LongTensor(ex.template_input),\
                            'template_target':torch.LongTensor(ex.template_target),\
                            'enttypes':[torch.LongTensor(x) for x in ex.ent_type]}
        return _cached_tensor

    def batch_fn(self, samples):
        batch_ent_text, batch_rel, batch_text, batch_raw_ent_text, batch_raw_sent_input, batch_examples,batch_raw_temp, \
            batch_tgt, batch_tgt_extend,batch_raw_tgt_text, batch_graph2, batch_oovs, batch_text_extend, \
            batch_ent_text_extend, batch_entscore,batch_template_input,batch_template_target,b_enttypes, batch_type =  [], [], [], [], [], [], [], [], [], [], [],[],[],[],[],[],[],[],[]
        batch_graph, batch_ex = map(list, zip(*samples))
        #pdb.set_trace()
        #pdb.set_trace()
        for graph,ex_data in zip(batch_graph,batch_ex):
            if ex_data != {}:
                batch_ent_text.append(ex_data['ent_text'])
                batch_rel.append(ex_data['rel'])
                batch_text.append(ex_data['text'])
                batch_raw_ent_text.append(ex_data['raw_ent_text'])
                batch_raw_sent_input.append(ex_data['raw_sent_input'])
                batch_examples.append(ex_data['example'])
                batch_tgt.append(ex_data['tgt'])
                batch_raw_tgt_text.append(ex_data['summary'])
                batch_graph2.append(graph)
                batch_oovs.append(ex_data['oovs'])
                batch_tgt_extend.append(ex_data['tgt_extend'])
                batch_text_extend.append(ex_data['text_extend'])
                batch_ent_text_extend.append(ex_data['ent_text_extend'])
                batch_entscore.append(ex_data['ent_score'])
                batch_template_input.append(ex_data['template_input'])
                batch_template_target.append(ex_data['template_target'])
                batch_raw_temp.append(ex_data['raw_temp'])

                b_enttypes.append(ex_data['enttypes'])
                batch_type.append(ex_data['type'])

        pad_id = self.wordvocab.word2id('<PAD>')
        bos_id = self.wordvocab.word2id('<BOS>')
        eos_id = self.wordvocab.word2id('<EOS>')
        
        batch_ent_text,ent_num = pad_sent_entity(batch_ent_text, pad_id, bos_id,eos_id, flatten = True)
        batch_rel = pad(batch_rel, out_type='tensor')
        '''
        batch_tgt = pad_sent_entity(batch_tgt, pad_id,bos_id,eos_id, flatten = False)
        batch_tgt_extend = pad_sent_entity(batch_tgt_extend, pad_id,bos_id,eos_id, flatten = False)
        '''
        batch_text,sent_num = pad_sent_entity(batch_text, pad_id,bos_id,eos_id, flatten = True)
        b_enttypes,_ = pad_sent_entity(b_enttypes, pad_id,bos_id,eos_id, flatten = True)

        batch_text_extend,sent_num = pad_sent_entity(batch_text_extend, pad_id,bos_id,eos_id, flatten = True)
        batch_ent_text_extend,_ = pad_sent_entity(batch_ent_text_extend, pad_id,bos_id,eos_id, flatten = True)
        batch_entscore,_ = pad_sent_entity(batch_entscore, 0,bos_id,eos_id, flatten = True) # entscore的pad_id为0
        batch_edges = pad_edges(batch_examples)
        '''
        batch_template_input = pad_sent_entity(batch_template_input, pad_id,bos_id,eos_id, flatten = False)
        batch_template_target = pad_sent_entity(batch_template_target, pad_id,bos_id,eos_id, flatten = False)
        
        '''
        batch_tgt_all = []
        batch_size = len(batch_tgt)
        batch_tgt_all.extend(batch_tgt)
        batch_tgt_all.extend(batch_tgt_extend)
        batch_tgt_all.extend(batch_template_input)
        batch_tgt_all.extend(batch_template_target)
        batch_tgt_all = pad_sent_entity(batch_tgt_all, pad_id,bos_id,eos_id, flatten = False)
        batch_tgt = batch_tgt_all.transpose(0,1)[:batch_size].transpose(0,1)
        batch_tgt_extend = batch_tgt_all.transpose(0,1)[batch_size:batch_size*2].transpose(0,1)
        batch_template_input = batch_tgt_all.transpose(0,1)[batch_size*2:batch_size*3].transpose(0,1)
        batch_template_target = batch_tgt_all.transpose(0,1)[batch_size*3:].transpose(0,1)
        
        batch_type = pad(batch_type, out_type='tensor')
        
        max_art_oovs = max([len(oovs) for oovs in batch_oovs])
        extra_zeros = None
        batch_size = len(batch_graph2)
        if max_art_oovs > 0:
            #extra_zeros = Variable(torch.zeros((batch_size, max_art_oovs)))
            extra_zeros = torch.zeros((batch_size, max_art_oovs))
        
        batch_graph = dgl.batch(batch_graph2)

        return {'ent_text': batch_ent_text, 'rel': batch_rel, 'text': batch_text, 'text_extend':batch_text_extend, 'extra_zeros':extra_zeros, \
             'graph': batch_graph, 'raw_ent_text': batch_raw_ent_text, 'raw_sent_input': batch_raw_sent_input, 'ent_num':ent_num, \
            'examples':batch_examples, 'tgt': batch_tgt, 'sent_num':sent_num, 'edges':batch_edges, 'raw_tgt_text': batch_raw_tgt_text, \
            'article_oovs':batch_oovs, 'tgt_extend': batch_tgt_extend, 'ent_text_extend':batch_ent_text_extend,'ent_score':batch_entscore,\
            'template_input':batch_template_input,'template_target':batch_template_target,\
            'enttypes':b_enttypes, 'type': batch_type, 'raw_temp':batch_raw_temp}
        
if __name__ == '__main__' :
    pass

