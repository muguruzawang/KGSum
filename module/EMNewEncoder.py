from module.transformer_encoder import NewTransformerEncoder, BertLSTMEncoder
from module.neural import PositionwiseFeedForward, sequence_mask
from module.roberta import RobertaEmbedding
from modules import GraphTrans

import torch.nn as nn
import torch
import pdb

from module.utlis_dataloader import *

class EMEncoder(nn.Module):
    def __init__(self, args, device, src_embeddings, padding_idx, bert_model):
        super(EMEncoder, self).__init__()
        self.args = args
        self.padding_idx = padding_idx
        self.device = device
        self.embeddings = src_embeddings
        # self._TFembed = nn.Embedding(50, self.args.emb_size) # box=10 , embed_size = 256

        if args.use_bert:
            self.bert_model = bert_model
            self.sent_encoder = BertLSTMEncoder(self.bert_model)
            self.entity_encoder = BertLSTMEncoder(self.bert_model)
        else:
            self.sent_encoder = NewTransformerEncoder(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                                      self.args.ff_size, self.args.enc_dropout, src_embeddings, self.device)
            self.entity_encoder = NewTransformerEncoder(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                                      self.args.ff_size, self.args.enc_dropout, src_embeddings, self.device)

        self.graph_enc = GraphTrans(args)

        self.layer_norm = nn.LayerNorm(self.args.emb_size, eps=1e-6)
        # self.layer_norm2 = nn.LayerNorm(self.args.emb_size, eps=1e-6)
        self.feed_forward = PositionwiseFeedForward(self.args.enc_hidden_size, self.args.ff_size, self.args.enc_dropout)

        self.rel_emb = nn.Embedding(len(args.rel_vocab), args.enc_hidden_size, padding_idx=self.padding_idx)
        nn.init.xavier_normal_(self.rel_emb.weight)

    def forward(self, batch):
        """
        :param src:  batch_size x n_paras x n_tokens
        :param cluster: batch_size x n_clusters x n_cluster_tokens
        :param edge: batch_size x n_paras x n_clusters
        :return:
        """
        src = batch['text']
        ent = batch['ent_text']
        #print(src.size())

        batch_size, n_sents, n_tokens = src.size()
        #print(ent.size())
        n_ent, n_ent_tokens = ent.size(1), ent.size(2)
        sent_feature, sent_context, _ = self.sent_encoder(src)      #
        ent_feature, ent_context, __ = self.entity_encoder(ent)
        rel_feature = self.embeddings(batch['rel'])
        type_feature = self.embeddings(batch['type'])
        
        ###################################################
        
        ent_num_mask = len2mask(batch['ent_num'], self.args.device)
        # batch['ent_text']是对实体进行0填充后的状态，所以后三行得到的**_mask都是[0,0,0,0,...,,1,1,1]这样的数据
        rel_mask = batch['rel']==1 # 0 means the <PAD>
        sent_num_mask = len2mask(batch['sent_num'], self.args.device)
        
        g_sent, g_ent = self.graph_enc(sent_feature, sent_num_mask, ent_feature, ent_num_mask, rel_feature, rel_mask,type_feature, batch['graph'], batch['sent_num'], batch['ent_num'])

        _sent_state = g_sent.unsqueeze(2)
        sent_context = self.feed_forward(sent_context + _sent_state)         # batch_size x n_paras x n_tokens x hidden
        mask_para = ~(src.data.eq(self.padding_idx).bool())
        mask_para = mask_para[:, :, :, None].float()
        sent_context = sent_context * mask_para

        _ent_state = g_ent.unsqueeze(2)
        ent_context = self.feed_forward(ent_context + _ent_state)         # batch_size x n_paras x n_tokens x hidden
        mask_ent = ~(ent.data.eq(self.padding_idx).bool())
        mask_ent = mask_ent[:, :, :, None].float()
        ent_context = ent_context * mask_ent
        #sent_context = sent_context.transpose(0, 1).contiguous()
        # to be consistent with the predictor
        # para_context = para_context.view(batch_size, n_paras * n_tokens, -1).transpose(0, 1).contiguous()       # (n_paras*n_tokens) x batch_size x hidden
        
        return g_sent, sent_context, g_ent, ent_context
        