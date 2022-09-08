#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to read the train/eval/test data from file and process it, and read the vocab data from file and process it"""

import re
import os
from nltk.corpus import stopwords

import glob
import copy
import random
import time
import json
import pickle
import nltk
import collections
from collections import Counter
from itertools import combinations
import numpy as np
from random import shuffle
import pdb

import torch
import torch.utils.data
import torch.nn.functional as F

from tools.logger import *

import dgl
from dgl.data.utils import save_graphs, load_graphs
from itertools import combinations

FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']
FILTERWORD.extend(punctuations)


######################################### Example #########################################

class Example(object):
    """Class representing a train/val/test example for single-document extractive summarization."""

    def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, label):
        """ Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        :param article_sents: list(strings) for single document or list(list(string)) for multi-document; one per article sentence. each token is separated by a single space.
        :param abstract_sents: list(strings); one per abstract sentence. In each sentence, each token is separated by a single space.
        :param vocab: Vocabulary object
        :param sent_max_len: int, max length of each sentence
        :param label: list, the No of selected sentence, e.g. [1,3,5]
        """

        self.sent_max_len = sent_max_len
        self.enc_sent_len = []
        self.enc_sent_input = []
        self.enc_sent_input_pad = []

        # Store the original strings
        self.original_article_sents = article_sents
        self.original_abstract = "\n".join(abstract_sents)

        # Process the article
        # 多文档摘要
        if isinstance(article_sents, list) and isinstance(article_sents[0], list):  # multi document
            self.original_article_sents = []
            ###将多文档的二级列表拓展为一级列表
            for doc in article_sents:
                self.original_article_sents.extend(doc)
        for sent in self.original_article_sents:
            article_words = sent.split() ###通过split将单词分割出来
            ### 存储句子的本来长度
            self.enc_sent_len.append(len(article_words))  # store the length before padding
            ### 存储句子的单词id
            self.enc_sent_input.append([vocab.word2id(w.lower()) for w in article_words])  # list of word ids; OOVs are represented by the id for UNK token ###存储为id
        self._pad_encoder_input(vocab.word2id('[PAD]'))

        # Store the label
        self.label = label
        label_shape = (len(self.original_article_sents), len(label))  # [N, len(label)]
        # label_shape = (len(self.original_article_sents), len(self.original_article_sents))
        self.label_matrix = np.zeros(label_shape, dtype=int)
        if label != []:
            self.label_matrix[np.array(label), np.arange(len(label))] = 1  # label_matrix[i][j]=1 indicate the i-th sent will be selected in j-th step

    def _pad_encoder_input(self, pad_id):
        """
        :param pad_id: int; token pad id
        :return: 
        """
        max_len = self.sent_max_len
        for i in range(len(self.enc_sent_input)):
            article_words = self.enc_sent_input[i].copy()
            if len(article_words) > max_len:
                article_words = article_words[:max_len]
            if len(article_words) < max_len:
                article_words.extend([pad_id] * (max_len - len(article_words)))
            self.enc_sent_input_pad.append(article_words)


class Example2(Example):
    """Class representing a train/val/test example for multi-document extractive summarization."""

    def __init__(self, article_sents, abstract_sents, vocab, sent_max_len, label):
        """ Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

        :param article_sents: list(list(string)) for multi-document; one per article sentence. each token is separated by a single space.
        :param abstract_sents: list(strings); one per abstract sentence. In each sentence, each token is separated by a single space.
        :param vocab: Vocabulary object
        :param sent_max_len: int, max length of each sentence
        :param label: list, the No of selected sentence, e.g. [1,3,5]
        """

        super().__init__(article_sents, abstract_sents, vocab, sent_max_len, label)
        cur = 0
        self.original_articles = []
        self.article_len = []
        self.enc_doc_input = []
        for doc in article_sents:
            if len(doc) == 0:
                continue
            docLen = len(doc)
            ###将每篇文档的语句展成一句话
            self.original_articles.append(" ".join(doc))
            self.article_len.append(docLen)
            self.enc_doc_input.append(catDoc(self.enc_sent_input[cur:cur + docLen]))
            cur += docLen


######################################### ExampleSet #########################################

class ExampleSet(torch.utils.data.Dataset):
    """ Constructor: Dataset of example(object) for single document summarization"""

    def __init__(self, data_path, vocab, doc_max_timesteps, sent_max_len, filter_word_path, w2s_path, entity_path, entityvocab):
        """ Initializes the ExampleSet with the path of data
        
        :param data_path: string; the path of data
        :param vocab: object;
        :param doc_max_timesteps: int; the maximum sentence number of a document, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param filter_word_path: str; file path, the file must contain one word for each line and the tfidf value must go from low to high (the format can refer to script/lowTFIDFWords.py) 
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2sTFIDF.py)
        """

        self.vocab = vocab
        self.sent_max_len = sent_max_len
        self.doc_max_timesteps = doc_max_timesteps

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        self.example_list = readJson(data_path) ###将训练数据读出来（text: , summary: ,label:）
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.example_list))
        self.size = len(self.example_list) ###训练集的大小

        logger.info("[INFO] Loading filter word File %s", filter_word_path)
        tfidf_w = readText(filter_word_path) 
        self.filterwords = FILTERWORD  ##停用词
        self.filterids = [vocab.word2id(w.lower()) for w in FILTERWORD]  ###id
        self.filterids.append(vocab.word2id("[PAD]"))   # keep "[UNK]" but remove "[PAD]"  ###再加上pad
        lowtfidf_num = 0
        pattern = r"^[0-9]+$"
        for w in tfidf_w:
            if vocab.word2id(w) != vocab.word2id('[UNK]'):
                self.filterwords.append(w)
                self.filterids.append(vocab.word2id(w))
                # if re.search(pattern, w) == None:  # if w is a number, it will not increase the lowtfidf_num
                    # lowtfidf_num += 1
                lowtfidf_num += 1
            if lowtfidf_num > 5000:
                break

        logger.info("[INFO] Loading word2sent TFIDF file from %s!" % w2s_path)
        self.w2s_tfidf = readJson(w2s_path)

        logger.info("[INFO] Loading entity file from %s!" % entity_path)
        self.entity2sen = readJson(entity_path)
        self.entityvocab = entityvocab

    def get_example(self, index):
        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"])
        return example

    def pad_label_m(self, label_matrix):
        label_m = label_matrix[:self.doc_max_timesteps, :self.doc_max_timesteps]
        N, m = label_m.shape
        if m < self.doc_max_timesteps:
            pad_m = np.zeros((N, self.doc_max_timesteps - m))
            return np.hstack([label_m, pad_m])
        return label_m

    def AddWordNode(self, G, inputid):
        wid2nid = {}
        nid2wid = {}
        nid = 0

        ###两个字典，word2id id2word
        for sentid in inputid:
            for wid in sentid:
                if wid not in self.filterids and wid not in wid2nid.keys():
                    wid2nid[wid] = nid
                    nid2wid[nid] = wid
                    nid += 1

        w_nodes = len(nid2wid)

        ###增加w_nodes个新节点
        G.add_nodes(w_nodes)
        ###对节点的所有特征进行初始化，0初始化，但是不太合理呀
        G.set_n_initializer(dgl.init.zero_initializer)
        #ndata是节点特征 edata是边的特征
        G.ndata["unit"] = torch.zeros(w_nodes)  ###单词节点unit=0
        G.ndata["id"] = torch.LongTensor(list(nid2wid.values()))  ###单词节点id 
        G.ndata["dtype"] = torch.zeros(w_nodes) ###单词节点dtype = 0

        return wid2nid, nid2wid
    

    def CreateGraph(self, input_pad, label, w2s_w, entity_json):
        """ Create a graph for each document
        
        :param input_pad: list(list); [sentnum, wordnum]
        :param label: list(list); [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}; for each sentence and each word, the tfidf between them
        :return: G: dgl.DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
                entity: unit=2,dtype=1,words=tensor
            edge:
                word2sent, sent2word:  tffrac=int, dtype=0
                sent2entity,entity2sent: dtype=1
        """

        G = dgl.DGLGraph()
        ###此处在图中加入单词节点和特征
        wid2nid, nid2wid = self.AddWordNode(G, input_pad)
        ### 单词节点数
        w_nodes = len(nid2wid)

        ###加上句子节点
        N = len(input_pad)
        G.add_nodes(N)
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        sentid2nid = [i + w_nodes for i in range(N)]

        G.set_e_initializer(dgl.init.zero_initializer)
        for i in range(N):
            c = Counter(input_pad[i])
            sent_nid = sentid2nid[i]
            sent_tfw = w2s_w[str(i)]
            for wid in c.keys():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    tfidf = sent_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    G.add_edges(wid2nid[wid], sent_nid,
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edges(sent_nid, wid2nid[wid],
                                data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
            
            # The two lines can be commented out if you use the code for your own training, since HSG does not use sent2sent edges. 
            # However, if you want to use the released checkpoint directly, please leave them here.
            # Otherwise it may cause some parameter corresponding errors due to the version differences.
            
            '''
            G.add_edges(sent_nid, sentid2nid, data={"dtype": torch.ones(N)})
            G.add_edges(sentid2nid, sent_nid, data={"dtype": torch.ones(N)})
            '''
        G.nodes[sentid2nid].data["words"] = torch.LongTensor(input_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]

        return G

    ### 最主要的就是这个函数，会自动调用，返回一组（G,index）
    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return 
            G: graph for the example
            index: int; the index of the example in the dataset
        """
        item = self.get_example(index)
        #enc_sent_input_pad是包含所有经过pad后的句子列表，这一步是对句子进行裁剪，只取前max个句子
        input_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
        label = self.pad_label_m(item.label_matrix)
        w2s_w = self.w2s_tfidf[index]
        entity_json = self.entity2sen[index]
        G = self.CreateGraph(input_pad, label, w2s_w, entity_json)

        return G, index

    ###返回数据集的总长度
    def __len__(self):
        return self.size


class MultiExampleSet(ExampleSet):
    """ Constructor: Dataset of example(object) for multiple document summarization"""
    def __init__(self, data_path, vocab, doc_max_timesteps, sent_max_len, filter_word_path, w2s_path, w2d_path, entity_path, entityvocab, ert_path):
        """ Initializes the ExampleSet with the path of data

        :param data_path: string; the path of data
        :param vocab: object;
        :param doc_max_timesteps: int; the maximum sentence number of a document, each example should pad sentences to this length
        :param sent_max_len: int; the maximum token number of a sentence, each sentence should pad tokens to this length
        :param filter_word_path: str; file path, the file must contain one word for each line and the tfidf value must go from low to high (the format can refer to script/lowTFIDFWords.py) 
        :param w2s_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2sTFIDF.py)
        :param w2d_path: str; file path, each line in the file contain a json format data (which can refer to the format can refer to script/calw2dTFIDF.py)
        """

        super().__init__(data_path, vocab, doc_max_timesteps, sent_max_len, filter_word_path, w2s_path, entity_path, entityvocab) ###构造方法的重构

        ###相比于word2sent 还多了一个word2document
        logger.info("[INFO] Loading word2doc TFIDF file from %s!" % w2d_path)
        self.w2d_tfidf = readJson(w2d_path)
        self.ert = readJson(ert_path)

    def get_example(self, index):
        e = self.example_list[index]
        e["summary"] = e.setdefault("summary", [])
        example = Example2(e["text"], e["summary"], self.vocab, self.sent_max_len, e["label"])
        return example
    
    def AddEntityNode(self, G, sent_pad, entity_json, wsd_nodes):
        entid2nid = {}
        nid2entid = {}
        nid = wsd_nodes

        '''
        print('sent_pad的大小为： '+str(len(sent_pad))+'\n')
        print('entity_json的大小为： '+str(len(entity_json))+'\n')
        print(entity_json)
        print('\n')
        '''
        for sentid in range(len(sent_pad)):
            for entity in entity_json[str(sentid)]:
                if entity in self.entityvocab.entity_list():
                    entid = self.entityvocab.entity2id(entity)
                    if entity not in entid2nid.keys():
                        entid2nid[entid] = nid
                        nid2entid[nid] = entid
                        nid += 1
            
        ent_nodes = len(nid2entid)

        ###增加w_nodes个新节点
        G.add_nodes(ent_nodes)
        #G.set_n_initializer(dgl.init.zero_initializer)

        G.ndata["unit"][wsd_nodes:] = torch.ones(ent_nodes) * 2
        G.ndata["dtype"][wsd_nodes:] = torch.ones(ent_nodes) * 3
        G.ndata["id"][wsd_nodes:] = torch.LongTensor(list(nid2entid.values()))

        return entid2nid, nid2entid

    ###add relation node as levi graph manner
    def AddLeviRelationNode(self, G, wsde_nodes, relations, clusters):
        relid2nid = {}
        nid2relid = {}
        nid = wsde_nodes
        relid2ent = []
        ###计算需要添加多少个关系节点，每个关系节点有正反两个节点
        '''
        rel_nodes = 0
        for rel in relations:
            relid2nid[rel_nodes] = nid
            nid2relid[nid] = rel_nodes
            relid2ent.append()
            rel_nodes += 1
            nid += 1
            relid2nid[rel_nodes] = nid
            nid2relid[nid] = rel_nodes
        '''
        rel_nodes = 0
        for relation in relations:
            ent1 = relation[0]
            ent2 = relation[2]
            if ent1 in self.entityvocab.entity_list() and ent2 in self.entityvocab.entity_list():
                rel_nodes += 2
        for cluster in clusters:
            cluster = list(set(cluster))
            for ent in set_cluster:
                if ent not in self.entityvocab.entity_list():
                    cluster.remove(ent)
            rel_nodes += len(list(combinations(set_cluster,2)))*2

        ###增加rel_nodes个新节点
        G.add_nodes(rel_nodes)
        #G.set_n_initializer(dgl.init.zero_initializer)

        G.ndata["unit"][wsde_nodes:] = torch.ones(ent_nodes) * 3
        G.ndata["dtype"][wsde_nodes:] = torch.ones(ent_nodes) * 4
        '''
        G.ndata["id"][wsde_nodes:] = torch.LongTensor(list(nid2entid.values()))
        '''
        return


    def MapSent2Doc(self, article_len, sentNum):
        sent2doc = {}
        doc2sent = {}
        sentNo = 0
        for i in range(len(article_len)):
            doc2sent[i] = []
            for j in range(article_len[i]):
                sent2doc[sentNo] = i
                doc2sent[i].append(sentNo)
                sentNo += 1

                '''
                难不成是这里的问题困扰了我这半个月，sentNum最大为50，而sentNo从0开始计数，sentNo为51的时候，才会满足条件，退出；
                而这时sent2doc加入了多一个文档的编号， doc2sent[i].append(50),实际上编号50的语句是不会出现在数据集中的；sent2doc里的文档会多一
                我把> 替换为 >=.再试试
                '''

                if sentNo >= sentNum:  
                    return sent2doc
        return sent2doc

    def CreateGraph(self, docLen, sent_pad, doc_pad, label, w2s_w, w2d_w, entity_json, relations, clusters):
        """ Create a graph for each document

        :param docLen: list; the length of each document in this example
        :param sent_pad: list(list), [sentnum, wordnum]
        :param doc_pad: list, [document, wordnum]
        :param label: list(list), [sentnum, sentnum]
        :param w2s_w: dict(dict) {str: {str: float}}, for each sentence and each word, the tfidf between them
        :param w2d_w: dict(dict) {str: {str: float}}, for each document and each word, the tfidf between them
        :return: G: dgl.DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
                document: unit=1, dtype=2
                entity: unit=2,dtype=3,words=tensor
                relation: unit=3, dtype=4
            edge:
                word2sent, sent2word: tffrac=int, dtype=0
                word2doc, doc2word: tffrac=int, dtype=0
                sent2doc: dtype=2
                sent2entity,entity2sent: dtype=1
                ent2rel,rel2sent: dtype=3
        """
        # add word nodes
        G = dgl.DGLGraph()
        ###此处在图中加入单词节点和特征
        wid2nid, nid2wid = self.AddWordNode(G, sent_pad)
        w_nodes = len(nid2wid)

        # add sent nodes
        ###句子结点数
        N = len(sent_pad)
        ###增加n个句子节点
        G.add_nodes(N)
        ###句子节点unit = 1 dtype = 1
        G.ndata["unit"][w_nodes:] = torch.ones(N)
        G.ndata["dtype"][w_nodes:] = torch.ones(N)
        ### sentid的编号从单词节点id之后开始
        sentid2nid = [i + w_nodes for i in range(N)]
        ###单词节点加句子节点总数
        ws_nodes = w_nodes + N

        # add doc nodes
        ###增加文档节点
        sent2doc = self.MapSent2Doc(docLen, N)  ###这是将所有句子映射到对应的文档id中
        article_num = len(set(sent2doc.values()))
        G.add_nodes(article_num)
        ###文档节点unit = 1 dtype = 2
        G.ndata["unit"][ws_nodes:] = torch.ones(article_num)
        G.ndata["dtype"][ws_nodes:] = torch.ones(article_num) * 2
        ### docid的编号从单词节点和句子id之后开始
        docid2nid = [i + ws_nodes for i in range(article_num)]

        '''
        加上entity节点
        '''
        wsd_nodes = ws_nodes + article_num
        entid2nid, nid2entid = self.AddEntityNode(G, sent_pad, entity_json, wsd_nodes)


        '''
        加上relation节点
        '''
        wsde_nodes = wsd_nodes + len(nid2entid)
        AddLeviRelationNode(self, G, wsde_nodes, relations, clusters)

        # add sent edges
        ### 增加句子和单词之间的边
        for i in range(N):
            c = Counter(sent_pad[i]) #计数器,统计列表中的单词出现次数
            sent_nid = sentid2nid[i] #当前句子的nodeid
            sent_tfw = w2s_w[str(i)] #当前句子的tfidf值

            temp = 0.1
            for wid, cnt in c.items():###cnt is not needed
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in sent_tfw.keys():
                    tfidf = sent_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10

                    temp = tfidf_box
                    # w2s s2w
                    ###增加无向边
                    G.add_edge(wid2nid[wid], sent_nid,
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edge(sent_nid, wid2nid[wid],
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
            # s2d
            docid = sent2doc[i]
            docnid = docid2nid[docid]
            ###增加句子和文档之间的边

            ###我把这一行注释掉，不清楚是不是这一行引起了建图之中的问题
            ###不能把这一行注释掉，注释了就找不到dnode的前辈节点snode了
            #G.add_edge(sent_nid, docnid, data={dtype": torch.Tensor([2])})
            ###我加上一个无关紧要的tffrac值呢
            G.add_edge(sent_nid, docnid, data={"tffrac": torch.LongTensor([temp]), "dtype": torch.Tensor([2])})

        # add doc edges
        for i in range(article_num):
            c = Counter(doc_pad[i]) #统计文档中单词出现的次数
            doc_nid = docid2nid[i]
            doc_tfw = w2d_w[str(i)]
            for wid, cnt in c.items():
                if wid in wid2nid.keys() and self.vocab.id2word(wid) in doc_tfw.keys():
                    # w2d d2w
                    tfidf = doc_tfw[self.vocab.id2word(wid)]
                    tfidf_box = np.round(tfidf * 9)  # box = 10
                    G.add_edge(wid2nid[wid], doc_nid,
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})
                    G.add_edge(doc_nid, wid2nid[wid],
                               data={"tffrac": torch.LongTensor([tfidf_box]), "dtype": torch.Tensor([0])})

        G.nodes[sentid2nid].data["words"] = torch.LongTensor(sent_pad)  # [N, seq_len]
        G.nodes[sentid2nid].data["position"] = torch.arange(1, N + 1).view(-1, 1).long()  # [N, 1]
        G.nodes[sentid2nid].data["label"] = torch.LongTensor(label)  # [N, doc_max]

        # add sent2entity edges
        ### 增加句子和实体之间的边
        for i in range(N):
            sent_nid = sentid2nid[i] #当前句子的nodeid
            sent_ent = entity_json[str(i)] #当前句子的tfidf值

            if sent_ent == []:
                continue
            c = Counter(sent_ent)

            for entity, cnt in c.items():
                if entity in self.entityvocab.entity_list():
                    entid = self.entityvocab.entity2id(entity)
                    if entid in entid2nid.keys():
                        ###这里还是要暂定一个边权，都定为1吧
                        value = 1.0
                        value_box = np.round(value * 9) 
                        G.add_edge(entid2nid[entid], sent_nid,
                                data={"tffrac": torch.LongTensor([value_box]), "dtype": torch.Tensor([1])})
                        G.add_edge(sent_nid, entid2nid[entid],
                                data={"tffrac": torch.LongTensor([value_box]), "dtype": torch.Tensor([1])})
        
        # 增加文档和实体的边
        for i in range(article_num):
            entitys = []
            for key in entity_json.keys():
                if int(key) >= len(sent2doc.keys()):
                    break
                if sent2doc[int(key)] == i:
                    entitys.extend(entity_json[key])
            
            if entitys == []:
                continue
            c = Counter(entitys) #统计文档中单词出现的次数
            doc_nid = docid2nid[i]
            for entity, cnt in c.items():
                if entity in self.entityvocab.entity_list():
                    entid = self.entityvocab.entity2id(entity)
                    if entid in entid2nid.keys():
                        ###这里还是要暂定一个边权，都定为1吧
                        value = 1.0
                        value_box = np.round(value * 9) 
                        G.add_edge(entid2nid[entid], doc_nid,
                                data={"tffrac": torch.LongTensor([value_box]), "dtype": torch.Tensor([1])})
                        G.add_edge(doc_nid, entid2nid[entid],
                                data={"tffrac": torch.LongTensor([value_box]), "dtype": torch.Tensor([1])})

        # 增加实体和关系的边
        rel_nodes = 0
        for rel in relations:
            ent1 = rel[0]
            ent2 = rel[2]
            ###如果两个实体都在entity_list内
            if ent1 in self.entityvocab.entity_list() and ent2 in self.entityvocab.entity_list():
                ent1id = self.entityvocab.entity2id(ent1)
                ent2id = self.entityvocab.entity2id(ent2)
                value = 1.0
                value_box = np.round(value * 9) 
                G.add_edge(entid2nid[ent1id], wsde_nodes+rel_nodes,
                        data={"tffrac": torch.LongTensor([value_box]), "dtype": torch.Tensor([3])})
                G.add_edge(wsde_nodes+rel_nodes, entid2nid[ent2id],
                        data={"tffrac": torch.LongTensor([value_box]), "dtype": torch.Tensor([3])})
                rel_nodes += 1

                G.add_edge(entid2nid[ent2id], wsde_nodes+rel_nodes,
                        data={"tffrac": torch.LongTensor([value_box]), "dtype": torch.Tensor([3])})
                G.add_edge(wsde_nodes+rel_nodes, entid2nid[ent1id],
                        data={"tffrac": torch.LongTensor([value_box]), "dtype": torch.Tensor([3])})
                rel_nodes += 1

        ###添加co-reference关系
        for cluster in clusters:
            cluster = list(set(cluster))
            for ent in set_cluster:
                if ent not in self.entityvocab.entity_list():
                    cluster.remove(ent)
            combs = list(combinations(cluster,2))
            for comb in combs:
                ent1 = comb[0]
                ent2 = comb[2]
                ###如果两个实体都在entity_list内
                if ent1 in self.entityvocab.entity_list() and ent2 in self.entityvocab.entity_list():
                    ent1id = self.entityvocab.entity2id(ent1)
                    ent2id = self.entityvocab.entity2id(ent2)
                    value = 1.0
                    value_box = np.round(value * 9) 
                    G.add_edge(entid2nid[ent1id], wsde_nodes+rel_nodes,
                            data={"tffrac": torch.LongTensor([value_box]), "dtype": torch.Tensor([3])})
                    G.add_edge(wsde_nodes+rel_nodes, entid2nid[ent2id],
                            data={"tffrac": torch.LongTensor([value_box]), "dtype": torch.Tensor([3])})
                    rel_nodes += 1

                    G.add_edge(entid2nid[ent2id], wsde_nodes+rel_nodes,
                            data={"tffrac": torch.LongTensor([value_box]), "dtype": torch.Tensor([3])})
                    G.add_edge(wsde_nodes+rel_nodes, entid2nid[ent1id],
                            data={"tffrac": torch.LongTensor([value_box]), "dtype": torch.Tensor([3])})
                    rel_nodes += 1
                    
        return G

    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return 
            G: graph for the example
            index: int; the index of the example in the dataset
        """
        item = self.get_example(index)
        sent_pad = item.enc_sent_input_pad[:self.doc_max_timesteps]
        enc_doc_input = item.enc_doc_input
        article_len = item.article_len
        label = self.pad_label_m(item.label_matrix)

        relations = self.ert[index]['relations']
        clusters = self.ert[index]['clusters']

        entity_json = self.entity2sen[index]
        G = self.CreateGraph(article_len, sent_pad, enc_doc_input, label, self.w2s_tfidf[index], self.w2d_tfidf[index], entity_json, relations, clusters)

        return G, index


class LoadHiExampleSet(torch.utils.data.Dataset):
    def __init__(self, data_root):
        super().__init__()
        self.data_root = data_root
        self.gfiles = [f for f in os.listdir(self.data_root) if f.endswith("graph.bin")]
        logger.info("[INFO] Start loading %s", self.data_root)

    def __getitem__(self, index):
        graph_file = os.path.join(self.data_root, "%d.graph.bin" % index)
        g, label_dict = load_graphs(graph_file)
        # print(graph_file)
        return g[0], index

    def __len__(self):
        return len(self.gfiles)


######################################### Tools #########################################


import dgl


def catDoc(textlist):
    res = []
    for tlist in textlist:
        res.extend(tlist)
    return res


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


def graph_collate_fn(samples):
    '''
    :param batch: (G, input_pad)
    :return: 
    '''
    graphs, index = map(list, zip(*samples))
    graph_len = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graphs]  # sent node of graph
    sorted_len, sorted_index = torch.sort(torch.LongTensor(graph_len), dim=0, descending=True)
    batched_graph = dgl.batch([graphs[idx] for idx in sorted_index])  ###将一个batch的graph打包成一个batch_graph，最后返回的是一张图，好处在于：任何用于操作一张小图的代码可以被直
                                                                      ## 接使用在一个图批量上。其次，由于DGL能够并行处理图中节点和边上的计算，因此同一批量内的图样本都可以被并行计算
    return batched_graph, [index[idx] for idx in sorted_index]
