"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np

from module.attention import MultiHeadedAttention
from module.neural import PositionwiseFeedForward
from module.transformer_encoder import PositionalEncoding

MAX_SIZE = 5000


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        """ Need to document this """
        self.hidden = tuple([_.detach() for _ in self.hidden])
        self.input_feed = self.input_feed.detach()

    def beam_update(self, idx, positions, beam_size):
        """ Need to document this """
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))

    def map_batch_fn(self, fn):
        raise NotImplementedError()



class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout, padding_idx):
        super(TransformerDecoderLayer, self).__init__()


        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.ent_attn = HierentAttention(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE).bool()
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)
        self.padding_idx = padding_idx
        self.d_model = d_model

    def forward(self, inputs, src, src_features, ent_feature, ent_mask, edge,
                tgt_pad_mask, layer_cache=None, step=None):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`      actually, 1 is tgt_len
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`
            mask_ent

        Returns:
            (`FloatTensor`, `FloatTensor`, `FloatTensor`):

            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`
            * all_input `[batch_size x current_step x model_dim]`

        """
        batch_size, tgt_len, emb_dim = inputs.size()
        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                                      :tgt_pad_mask.size(1)], 0).bool()
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm

        query = self.self_attn(all_input, all_input, input_norm,
                                     mask=dec_mask.bool(),
                                     layer_cache=layer_cache,
                                     type="self")

        query = self.drop(query) + inputs
        query_norm = self.layer_norm_2(query)           # batch_size x tgt_len x emb

        selected_features, selected_mask = self.ent_attn(ent_feature, ent_mask,
                                            edge, query_norm, src, src_features, self.padding_idx, max_para=5)
        # selected_features: batch_size*tgt_len, src_words, embed
        # selected_mask: batch_size*tgt_len, src_words

        query_norm = query_norm.view(-1, emb_dim).unsqueeze(1)     # batch_size*tgt_len x 1 x emb


        selected_mask = selected_mask.unsqueeze(1)      # batch_size*tgt_len x 1 x src_words
        mid = self.context_attn(selected_features, selected_features, query_norm,
                                      mask=~selected_mask,
                                      layer_cache=layer_cache,
                                      type="context")
        mid = mid.view(batch_size, tgt_len, -1)
        query_norm = mid.view(batch_size, tgt_len, -1)
        output = self.feed_forward(self.drop(mid) + query_norm)

        return output, all_input
        # return output

    def _get_attn_subsequent_mask(self, size):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('bool')
        subsequent_mask = torch.from_numpy(subsequent_mask).bool()
        return subsequent_mask


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings):
        super(TransformerDecoder, self).__init__()

        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.emb_dim = self.embeddings.weight.size(1)
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.linear_input = nn.Linear(self.emb_dim, d_model)
        self.linear_ent = nn.Linear(d_model, d_model)
        self.softmax_ent = nn.Softmax(dim=-1)
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout, padding_idx=1)
             for _ in range(num_layers)])

        # TransformerDecoder has its own attention mechanism.
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, tgt, memory_bank, state, edge, ent, ent_feature, memory_lengths=None,
                step=None, cache=None,memory_masks=None):
        """
        :param tgt: tgt_len x batch_size
        :param memory_bank:  batch_size x n_sent x n_tokens x embed_dim
        :param state:
        :param edge: batch_size x n_sent x n_ents
        :param ent: batch_size x n_ents x n_tokens
        :param ent_feature: batch_size x n_ents x hidden
        :param memory_lengths:
        :param step:
        :param cache:
        :param memory_masks:
        :return:
        """
        batch_size, n_sent, n_ent = edge.size()
        src = state.src  # n_blocks * n_tokens, batch_size
        src_words = src.transpose(0, 1).contiguous()
        src_words = src_words.view(batch_size, n_sent, -1)  # batch_size x n_blocks x n_tokens
        tgt_words = tgt.transpose(0, 1).contiguous()  # batch_size, max_len, because transposed in dataloader
        tgt_batch, tgt_len = tgt_words.size()
        padding_idx = 1
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len).bool()

        emb = self.embeddings(tgt)
        emb = self.linear_input(emb)
        assert emb.dim() == 3  # tgt_len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()  # batch_size x tgt_len x embedding_dim
        output = self.pos_emb(output, step)

        ent_mask = ~(ent.data.eq(padding_idx).bool())
        ent_mask = (torch.sum(ent_mask, -1) > 0).bool()

        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len).bool()

        for i in range(self.num_layers):
            output, all_input \
                = self.transformer_layers[i](output, src_words, memory_bank, ent_feature, ent_mask, edge,
                        tgt_pad_mask,layer_cache=state.cache["layer_{}".format(i)] if state.cache is not None else None, step=step)

        output = self.layer_norm(output)

        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()

        return outputs,state

    def init_decoder_state(self, src, memory_bank,
                           with_cache=False):
        """ Init decoder state """
        if(src.dim()==3):
            src = src.view(src.size(0),-1).transpose(0,1).contiguous()
            # nblocks * n_tokens x batch_size
        else:
            src = src.transpose(0,1).contiguous()
            # n_tokens x batch_size
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers)
        return state



class TransformerDecoderState(DecoderState):
    """ Transformer Decoder state base class """

    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        if (self.previous_input is not None
                and self.previous_layer_inputs is not None):
            return (self.previous_input,
                    self.previous_layer_inputs,
                    self.src)
        else:
            return (self.src,)

    def detach(self):
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()

    def update_state(self, new_input, previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def _init_cache(self, memory_bank, num_layers):
        self.cache = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for l in range(num_layers):
            layer_cache = {
                "memory_keys": None,
                "memory_values": None
            }
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.cache["layer_{}".format(l)] = layer_cache

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = self.src.data.repeat(1, beam_size, 1)

    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.src = fn(self.src, 1)
        if self.cache is not None:
            _recursive_map(self.cache)

# class HierDecoderLayerent(nn.Module):
#     def __init__(self, d_model, heads, d_ff, dropout):
#         super(HierDecoderLayerent, self).__init__()
#
#

class HierentAttention(nn.Module):
    def __init__(self, model_dim, dropout=0.1):
        self.model_dim = model_dim

        super(HierentAttention, self).__init__()
        self.linear_input = nn.Linear(model_dim, model_dim)
        self.linear_ent = nn.Linear(model_dim, model_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ent, ent_mask, edge, inputs, src, src_features, padding_idx, max_para=5):
        """
        :param ent: batch_size x n_ents x hidden
        :param ent_mask: index the pad ent nodes: batch_size x n_ents, 1 indicates the None nodes
        :param edge: batch_size x n_sent x n_ents
        :param input: the decoder tgt, batch_size x tgt_len x hidden
        :param src:
        :param src_features:
        :param padding_idx:
        :param max_para:
        :return:
        """
        key = self.linear_ent(ent)
        query = self.linear_input(inputs)
        '''
        if key.shape[0] != query.shape[0]:
            import pdb
            pdb.set_trace()

        print('key.shape=%s'%str(key.shape))
        print('query.shape=%s'%str(query.shape))
        '''
        scores = torch.matmul(query, key.transpose(1, 2))       # batch_size x tgt_len x n_ents

        if ent_mask is not None:
            ent_mask  = ent_mask.unsqueeze(1).expand_as(scores).bool()
            scores = scores.masked_fill(ent_mask,-1e18)

        attn_ent = self.softmax(scores)     # batch_size x tgt_len x n_ents
        attn_ent = self.dropout(attn_ent)
        attn_para = torch.matmul(attn_ent, edge.transpose(1, 2))        # 最后的结果是batch_size x tgt_len x n_sent
        # select the top-max_para score index of para

        ###这里返回按照降序排序好的句子
        attn_para = attn_para.argsort(descending=True)[:, :, :max_para]     # batch_size x tgt_len x max_para

        batch_size, tgt_len, max_n_para = attn_para.size()
        feature, word = [], []
        for i in range(attn_para.size(0)):
            _feature = []
            _word = []
            for j in range(attn_para.size(1)):
                __feature = torch.index_select(src_features[i], 0, attn_para[i][j])
                __word = torch.index_select(src[i], 0, attn_para[i][j])
                _feature.append(__feature)
                _word.append(__word)
            _feature = torch.stack(_feature, dim=0)
            _word = torch.stack(_word, dim=0)
            feature.append(_feature)
            word.append(_word)
        selected_features = torch.stack(feature, dim=0)  # batch_size x tgt_len x max_sent x n_tokens x embed
        selected_words = torch.stack(word, dim=0)  # batch_size x tgt_len x max_sent x n_tokens

        selected_features = selected_features.view(batch_size * tgt_len, -1, self.model_dim)
        selected_words = selected_words.view(batch_size * tgt_len, -1)
        selected_mask = ~(selected_words.data.eq(padding_idx).bool())

        return selected_features, selected_mask


