import math

import torch.nn as nn
import torch
from torch.nn import init

from module.attention  import MultiHeadedAttention, MultiHeadedPooling
from module.neural import PositionwiseFeedForward, PositionalEncoding, sequence_mask
from module.roberta import RobertaEmbedding
import pdb

INIT = 1e-2

class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, inputs, mask):
        """
        Transformer Encoder Layer definition.

        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`

        Returns:
            (`FloatTensor`):

            * outputs `[batch_size x src_len x model_dim]`
        """
        input_norm = self.layer_norm(inputs)
        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerPoolingLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerPoolingLayer, self).__init__()

        self.pooling_attn = MultiHeadedPooling(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        context = self.pooling_attn(inputs, inputs,
                                    mask=mask)
        out = self.dropout(context)

        return self.feed_forward(out)


class NewTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, device):
        super(NewTransformerEncoder, self).__init__()
        self.device = device
        self.d_model = d_model
        self.heads = heads
        self.d_per_head = self.d_model // self.heads
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, int(self.embeddings.embedding_dim / 2))
        self.transformer_local = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.d_per_head, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.pooling = MultiHeadedPooling(heads, d_model, dropout=dropout, use_final_linear=False)

    def forward(self, src, lengths=None):
        """ See :obj:`EncoderBase.forward()`"""
    
        batch_size, n_blocks, n_tokens = src.size()
        emb = self.embeddings(src)

        padding_idx = self.embeddings.padding_idx
        mask_local = ~(src.data.eq(padding_idx).view(batch_size * n_blocks, n_tokens).bool())
        mask_block = (torch.sum(mask_local.view(batch_size, n_blocks, n_tokens), -1) > 0).bool()

        local_pos_emb = self.pos_emb.pe[:, :n_tokens].unsqueeze(1).expand(batch_size, n_blocks, n_tokens,
                                                                          int(self.embeddings.embedding_dim / 2))
        inter_pos_emb = self.pos_emb.pe[:, :n_blocks].unsqueeze(2).expand(batch_size, n_blocks, n_tokens,
                                                                          int(self.embeddings.embedding_dim / 2))
        combined_pos_emb = torch.cat([local_pos_emb, inter_pos_emb], -1)
        emb = emb * math.sqrt(self.embeddings.embedding_dim)
        emb = emb + combined_pos_emb
        emb = self.pos_emb.dropout(emb)

        word_vec = emb.view(batch_size * n_blocks, n_tokens, -1)

        # word_vec.shape = [200,72,256]  mask-local = [200,72]   之所以为~mask_local是因为masked_fill是对1的地方进行0填充
        for i in range(self.num_layers):
            word_vec = self.transformer_local[i](word_vec, word_vec, ~mask_local)  # all_sents * max_tokens * dim
        word_vec = self.layer_norm1(word_vec)
        mask_inter = (~mask_block).unsqueeze(1).expand(batch_size, self.heads, n_blocks).contiguous()
        # [32,1,50]
        mask_inter = mask_inter.view(batch_size * self.heads, 1, n_blocks).bool()
        block_vec = self.pooling(word_vec, word_vec, ~mask_local)
        block_vec = block_vec.view(-1, self.d_per_head)
        block_vec = self.layer_norm2(block_vec)
        block_vec = self.dropout2(block_vec)
        block_vec = block_vec.view(batch_size * n_blocks, 1, -1)
        src_features = self.feed_forward(word_vec + block_vec)
        block_vec = block_vec.view(batch_size, n_blocks, -1)

        #[200,72,1]   mask_local.shape=[200,72]  这就是200句话，每句话里的单词和mask
        mask_hier = mask_local[:, :, None].float()
        src_features = src_features * mask_hier
        #[4,50,72,256]
        src_features = src_features.view(batch_size, n_blocks, n_tokens, -1)

        return block_vec, src_features, mask_hier

class NewTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(NewTransformerEncoderLayer, self).__init__()
        self.d_model, self.heads = d_model, heads
        self.d_per_head = self.d_model // self.heads
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.d_per_head, eps=1e-6)

        self.pooling = MultiHeadedPooling(heads, d_model, dropout=dropout, use_final_linear=False)

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        :param inputs: [ num_of_paras_in_one_batch x seq_len x d_model]
        :param mask: [ num_of_paras_in_one_batch x seq_len ]
        :return:
        """
        batch_size, seq_len, d_model = inputs.size()
        input_norm = self.layer_norm(inputs)
        mask_local = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm, mask=mask_local)
        para_vec = self.dropout(context) + inputs
        para_vec = self.pooling(para_vec, para_vec, mask)
        para_vec = self.layer_norm2(para_vec)
        para_vec = para_vec.view(batch_size, -1)
        return para_vec

class BertLSTMEncoder(nn.Module):
    def __init__(self, bert_model, padding_idx=1, dropout=0.1, n_layer=1,
                 bidirectional=True, n_hidden=256, heads=8):
        super(BertLSTMEncoder, self).__init__()
        self.bert_model = bert_model
        
        self.embedding= self.bert_model._embedding
        self.embedding.weight.requires_grad = False
        self.emb_dim = self.embedding.weight.size(1)
        
        self.padding_idx = padding_idx
        self.dropout = dropout
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.bidirectional = bidirectional
        self.heads = heads
        self.d_per_head = self.n_hidden // self.heads
        self.layer_norm1 = nn.LayerNorm(self.d_per_head, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = PositionwiseFeedForward(self.n_hidden, self.n_hidden, dropout)

        # initial encoder LSTM states are learned parameters
        state_layer = n_layer * (2 if bidirectional else 1)
        self._init_enc_h = nn.Parameter(
            torch.Tensor(state_layer, n_hidden)
        )
        self._init_enc_c = nn.Parameter(
            torch.Tensor(state_layer, n_hidden)
        )
        init.uniform_(self._init_enc_h, -INIT, INIT)
        init.uniform_(self._init_enc_c, -INIT, INIT)
        self.enc_lstm = nn.LSTM(self.emb_dim, self.n_hidden, self.n_layer,
                                bidirectional=self.bidirectional, dropout=self.dropout)
        self.pooling = MultiHeadedPooling(self.heads, self.n_hidden, dropout=dropout, use_final_linear=False)
        self.projection = nn.Linear(2 * self.n_hidden, self.n_hidden)



    def forward(self, src):
        """
        src: batch_size x n_paras x n_tokens
        """
        batch_size, n_sents, n_tokens = src.size()
        #找出src中的padding_idx,取非，即非pad的词为1，pad为0
        mask_local = ~(src.data.eq(self.padding_idx).view(batch_size * n_sents, n_tokens).bool())
        #返回的是每个sent的长度
        src_len = torch.sum(mask_local, -1).long()
        #维度变成二维，重新组织
        src = src.view(batch_size * n_sents, -1)

        # size = (state_layer,batch_size*n_sents,n_hidden)
        size = (
            self._init_enc_h.size(0),
            len(src_len),
            self._init_enc_h.size(1)
        )

        #self._init_enc_h = nn.Parameter(torch.Tensor(state_layer, n_hidden))
        # unsqueeze后变成了(state_layer,1, n_hidden)
        # expand后变成了((state_layer,batch_size*n_sents, n_hidden))
        init_enc_states = (
            self._init_enc_h.unsqueeze(1).expand(*size),
            self._init_enc_c.unsqueeze(1).expand(*size)
        )
        
        with torch.no_grad():
            bert_out = self.bert_model(src)
        bert_hidden = bert_out[0]

        enc_word, final_states = lstm_encoder(bert_hidden, self.enc_lstm, src_len,
                                             init_enc_states, None, {}, {})
        enc_word = self.projection(enc_word)
        enc_word = enc_word.transpose(0, 1).contiguous()
        # print(src.size(), enc_word.size(), mask_local.size())
        para_vec = self.pooling(enc_word, enc_word, ~mask_local)
        para_vec = para_vec.view(-1, self.d_per_head)
        para_vec = self.layer_norm1(para_vec)
        para_vec = self.dropout1(para_vec).view(batch_size * n_sents, 1, -1)
        src_features = enc_word + para_vec
        src_features = self.feed_forward(src_features.view(-1, self.n_hidden))
        src_features = src_features.view(batch_size * n_sents, n_tokens, -1)
        para_vec = para_vec.view(batch_size, n_sents, -1)
        mask_hier = mask_local[:, :, None].float()
        src_features = src_features * mask_hier
        src_features = src_features.view(batch_size, n_sents, n_tokens, -1)

        return para_vec, src_features, None

def lstm_encoder(sequence, lstm, seq_lens=None, init_states=None,
                 embedding=None, feature_embeddings={}, feature_dict={}):
    batch_size = sequence.size(0)
    if not lstm.batch_first:
        sequence = sequence.transpose(0, 1).contiguous()
        emb_sequence = (embedding(sequence) if embedding is not None else sequence)

    if seq_lens is not None:
        assert batch_size == len(seq_lens)
        sort_ind = sorted(range(len(seq_lens)), key=lambda i: seq_lens[i], reverse=True)
        seq_lens = [seq_lens[i] for i in sort_ind]
        emb_sequence = reorder_sequence(emb_sequence, sort_ind, lstm.batch_first)

    if init_states is None:
        device = sequence.device
        init_states = init_lstm_states(lstm, batch_size, device)
    else:
        init_states = (init_states[0].contiguous(),
                       init_states[1].contiguous())

    if seq_lens is not None:
        packed_seq = nn.utils.rnn.pack_padded_sequence(emb_sequence, seq_lens)

        packed_out, final_states = lstm(packed_seq, init_states)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out)

        back_map = {ind: i for i, ind in enumerate(sort_ind)}
        reorder_ind = [back_map[i] for i in range(len(seq_lens))]
        lstm_out = reorder_sequence(lstm_out, reorder_ind, lstm.batch_first)
        final_states = reorder_lstm_states(final_states, reorder_ind)
    else:
        lstm_out, final_states = lstm(emb_sequence, init_states)

    return lstm_out, final_states



#################### LSTM helper #########################

def reorder_sequence(sequence_emb, order, batch_first=False):
    """
    sequence_emb: [T, B, D] if not batch_first
    order: list of sequence length
    """
    batch_dim = 0 if batch_first else 1
    assert len(order) == sequence_emb.size()[batch_dim]

    order = torch.LongTensor(order).to(sequence_emb.device)
    sorted_ = sequence_emb.index_select(index=order, dim=batch_dim)

    return sorted_

def reorder_lstm_states(lstm_states, order):
    """
    lstm_states: (H, C) of tensor [layer, batch, hidden]
    order: list of sequence length
    """
    assert isinstance(lstm_states, tuple)
    assert len(lstm_states) == 2
    assert lstm_states[0].size() == lstm_states[1].size()
    assert len(order) == lstm_states[0].size()[1]

    order = torch.LongTensor(order).to(lstm_states[0].device)
    sorted_states = (lstm_states[0].index_select(index=order, dim=1),
                     lstm_states[1].index_select(index=order, dim=1))

    return sorted_states

def init_lstm_states(lstm, batch_size, device):
    n_layer = lstm.num_layers*(2 if lstm.bidirectional else 1)
    n_hidden = lstm.hidden_size

    states = (torch.zeros(n_layer, batch_size, n_hidden).to(device),
              torch.zeros(n_layer, batch_size, n_hidden).to(device))
    return states

