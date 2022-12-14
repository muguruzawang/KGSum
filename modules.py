import torch
import math
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from torch import nn
from module.utlis_dataloader import pad
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import pdb

NODE_TYPE = {'word':0, 'sentence': 1, 'doc':2, 'entity': 3, 'relation':4, 'type':5, 'root':6  }

class MSA(nn.Module):
    # multi-head self-attention, three modes
    # the first is the copy, determining which entity should be copied.
    # the second is the normal attention with two sequence inputs
    # the third is the attention but with one token and a sequence. (gather, attentive pooling)
    
    def __init__(self, args, mode='normal'):
        super(MSA, self).__init__()
        if mode=='copy':
            nhead, head_dim = 1, args.nhid
            qninp, kninp = args.dec_ninp, args.nhid
        if mode=='normal':
            nhead, head_dim = args.nhead, args.head_dim
            qninp, kninp = args.nhid, args.nhid
        self.attn_drop = nn.Dropout(0.1)
        self.WQ = nn.Linear(qninp, nhead*head_dim, bias=True if mode=='copy' else False)
        if mode!='copy':
            self.WK = nn.Linear(kninp, nhead*head_dim, bias=False)
            self.WV = nn.Linear(kninp, nhead*head_dim, bias=False)
        self.args, self.nhead, self.head_dim, self.mode = args, nhead, head_dim, mode

    def forward(self, inp1, inp2, mask=None):
        B, L2, H = inp2.shape
        NH, HD = self.nhead, self.head_dim
        if self.mode=='copy':
            q, k, v = self.WQ(inp1), inp2, inp2
        else:
            q, k, v = self.WQ(inp1), self.WK(inp2), self.WV(inp2)
        L1 = 1 if inp1.ndim==2 else inp1.shape[1]
        if self.mode!='copy':
            q = q / math.sqrt(H)
        q = q.view(B, L1, NH, HD).permute(0, 2, 1, 3) 
        k = k.view(B, L2, NH, HD).permute(0, 2, 3, 1)
        v = v.view(B, L2, NH, HD).permute(0, 2, 1, 3)
        pre_attn = torch.matmul(q,k)
        if mask is not None:
            pre_attn = pre_attn.masked_fill(mask[:,None,None,:], -1e8)
        if self.mode=='copy':
            return pre_attn.squeeze(1)
        else:
            alpha = self.attn_drop(torch.softmax(pre_attn, -1))
            attn = torch.matmul(alpha, v).permute(0, 2, 1, 3).contiguous().view(B,L1,NH*HD)
            ret = attn
            if inp1.ndim==2:
                return ret.squeeze(1)
            else:
                return ret

class BiLSTM(nn.Module):
    # for entity encoding or the title encoding
    def __init__(self, args, enc_type='ent'):
        super(BiLSTM, self).__init__()
        self.enc_type = enc_type
        self.drop = nn.Dropout(args.emb_drop)
        self.bilstm = nn.LSTM(args.nhid, args.nhid//2, bidirectional=True, \
                num_layers=args.enc_lstm_layers, batch_first=True)
 
    def forward(self, inp, mask, length=None):
        inp = self.drop(inp)
        # ?????????????????????????????????
        #pdb.set_trace()
        if self.enc_type == 'doc':
            lens = mask
            #??????pack???????????????packedlist ??????batch????????????????????????????????? packed_input.data.shape : (batch_sum_seq_len X embedding_dim) 
            pad_seq = pack_padded_sequence(inp, lens, batch_first=True, enforce_sorted=False)
            # ?????????y???output????????????_h???????????????????????????????????????
            y, (_h, _c) = self.bilstm(pad_seq)
            ###?????????title??????????????????lstm?????????????????????title??????

            _h = _h.transpose(0,1).contiguous()
            ###?????????lstm,??????h????????????????????????????????????????????????????????????????????????????????????????????????
            
            ###???????????????????????????????????????????????????????????????embedding ??????
            _h = _h[:,-2:].view(_h.size(0), -1) # two directions of the top-layer
            ret = pad(_h.split(length), out_type='tensor')
            return ret

        elif self.enc_type == 'sent':
            lens = (mask==0).sum(-1).long().tolist()
            #??????pack???????????????packedlist ??????batch????????????????????????????????? packed_input.data.shape : (batch_sum_seq_len X embedding_dim) 
            pad_seq = pack_padded_sequence(inp, lens, batch_first=True, enforce_sorted=False)
            # ?????????y???output????????????_h???????????????????????????????????????
            y, (_h, _c) = self.bilstm(pad_seq)
            ###?????????title??????????????????lstm?????????????????????title??????

            _h = _h.transpose(0,1).contiguous()
            ###?????????lstm,??????h????????????????????????????????????????????????????????????????????????????????????????????????
            
            ###???????????????????????????????????????????????????????????????embedding ??????
            _h = _h[:,-2:].view(_h.size(0), -1) # two directions of the top-layer
            ret = pad(_h.split(length[0]), out_type='tensor')

            ret_for_doc = pad(_h.split(length[1]), out_type='tensor')

            return ret, ret_for_doc
        else:
            lens = (mask==0).sum(-1).long().tolist()
            #??????pack???????????????packedlist ??????batch????????????????????????????????? packed_input.data.shape : (batch_sum_seq_len X embedding_dim) 
            pad_seq = pack_padded_sequence(inp, lens, batch_first=True, enforce_sorted=False)
            # ?????????y???output????????????_h???????????????????????????????????????
            y, (_h, _c) = self.bilstm(pad_seq)
            ###?????????title??????????????????lstm?????????????????????title??????

            _h = _h.transpose(0,1).contiguous()
            ###?????????lstm,??????h????????????????????????????????????????????????????????????????????????????????????????????????
            
            ###???????????????????????????????????????????????????????????????embedding ??????
            _h = _h[:,-2:].view(_h.size(0), -1) # two directions of the top-layer
            ret = pad(_h.split(length), out_type='tensor')
            return ret

class GATLayer_Hetersum(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    ### ????????????attention???????????????
    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)  
        wa = F.leaky_relu(self.attn_fc(z2))  # [edge_num, 1]
        return {'e': wa}

    ###message_func??????????????????????????????tensors,?????????????????????z,????????????????????????attention e
    def message_func(self, edges):
        # print("edge e ", edges.data['e'].size())
        return {'z': edges.src['z'], 'e': edges.data['e']}

    ###reduce_func??????softmax?????????attention?????????aggregate?????????????????????
    def reduce_func(self, nodes):
        ### mailbox???????????????
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'sh': h}

    def forward(self, g, h):
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)###dgl????????????apply_edges????????????????????????edge_attention???????????????
        ###  Pull messages from the node(s)??? predecessors and then update their features.
        #g.pull(snode_id, self.message_func, self.reduce_func) ###dgl????????????pull,???????????????message???reduce??????????????????
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata.pop('sh')
        return h  ###????????????????????????

class GATStackLayer_Hetersum(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop_out, merge='cat'):
        super(GATStackLayer_Hetersum, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads): #8 heads
            self.heads.append(GATLayer_Hetersum(in_dim, out_dim))  # [n_nodes, hidden_size]  
        self.merge = merge
        self.dropout = nn.Dropout(attn_drop_out) #????????????dropout

    def forward(self, g, h):
        head_outs = [attn_head(g, self.dropout(h)) for attn_head in self.heads]  # n_head * [n_nodes, hidden_size]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            result = torch.cat(head_outs, dim=1)  # [n_nodes, hidden_size * n_head]
        else:
            # merge using average
            result = torch.mean(torch.stack(head_outs))
        return result

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        assert not torch.any(torch.isnan(x)), "FFN input"
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        assert not torch.any(torch.isnan(output)), "FFN output"
        return output


class GAT_Hetersum(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.gat = nn.ModuleList([GATStackLayer_Hetersum(args.nhid, args.nhid//args.n_head, args.n_head, args.attn_drop) for _ in range(args.prop)]) #untested
        self.prop = args.prop

        self.ffn = PositionwiseFeedForward(args.nhid, args.ffn_inner_hidden_size, args.ffn_dropout_prob)

    def forward(self, word_enc, word_mask, sent_enc, text_num_mask, doc_enc, ent_enc, ent_num_mask, rel_emb, rel_mask, type_emb, type_mask, graphs, text_len, doc_num_mask):
        device = ent_enc.device
        graphs = graphs.to(device)
        ent_num_mask = (ent_num_mask==0) # reverse mask
        rel_mask = (rel_mask==0)
        word_mask = (word_mask == 0)
        text_num_mask = (text_num_mask == 0)
        type_mask = (type_mask == 0)
        doc_num_mask = (doc_num_mask == 0)

        init_h = []

        #pdb.set_trace()
        for i in range(graphs.batch_size):
            #pdb.set_trace()
            # add word embedding
            init_h.append(word_enc[i][word_mask[i]])
            # add sentence embedding
            init_h.append(sent_enc[i][text_num_mask[i]])
            # add doc embedding,
            # doc_enc ????????????????????????batch????????????????????????????????????
            init_h.append(doc_enc[i][doc_num_mask[i]])
            # add ent embedding
            init_h.append(ent_enc[i][ent_num_mask[i]])
            # add rel embedding
            init_h.append(rel_emb[i][rel_mask[i]])
            # add type embedding
            init_h.append(type_emb[i][type_mask[i]])

        init_h = torch.cat(init_h, 0)
        feats = init_h
        for i in range(self.prop):
            feats_pre = feats
            feats = F.elu(self.gat[i](graphs, feats))
            feats = feats + feats_pre
            feats = self.ffn(feats.unsqueeze(0)).squeeze(0)
        g_sent = feats.index_select(0, graphs.filter_nodes(lambda x: x.data['type']==NODE_TYPE['sentence']).to(device))#???????????????????????????
        return g_sent

class GAT(nn.Module):
    # a graph attention network with dot-product attention
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 ffn_drop=0.,
                 attn_drop=0.,
                 trans=True):
        super(GAT, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.q_proj = nn.Linear(in_feats, num_heads*out_feats, bias=False)
        self.k_proj = nn.Linear(in_feats, num_heads*out_feats, bias=False)
        self.v_proj = nn.Linear(in_feats, num_heads*out_feats, bias=False)
        self.attn_drop = nn.Dropout(0.1)
        self.ln1 = nn.LayerNorm(in_feats)
        self.ln2 = nn.LayerNorm(in_feats)
        if trans:
            self.FFN = nn.Sequential(
                nn.Linear(in_feats, 4*in_feats),
                nn.PReLU(4*in_feats),
                nn.Linear(4*in_feats, in_feats),
                nn.Dropout(0.1),
            )
            # a strange FFN, see the author's code
        self._trans = trans

    def forward(self, graph, feat):
        graph = graph.local_var()
        ###????????????feat
        feat_c = feat.clone().detach().requires_grad_(False)
        q, k, v = self.q_proj(feat), self.k_proj(feat_c), self.v_proj(feat_c)
        q = q.view(-1, self._num_heads, self._out_feats)
        k = k.view(-1, self._num_heads, self._out_feats)
        v = v.view(-1, self._num_heads, self._out_feats)
        graph.ndata.update({'ft': v, 'el': k, 'er': q}) # k,q instead of q,k, the edge_softmax is applied on incoming edges
        # compute edge attention
        graph.apply_edges(fn.u_dot_v('el', 'er', 'e'))
        e =  graph.edata.pop('e') / math.sqrt(self._out_feats * self._num_heads)
        graph.edata['a'] = edge_softmax(graph, e)
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft2'))
        rst = graph.ndata['ft2']
        # residual
        rst = rst.view(feat.shape) + feat
        if self._trans:
            rst = self.ln1(rst)
            rst = self.ln1(rst+self.FFN(rst))
            # use the same layer norm, see the author's code
        return rst


### graph  transformer
class GraphTrans(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        if args.graph_enc == "gat":
            # we only support gtrans, don't use this one
            ###???4???????????????????????????????????????????????????4
            self.gat = nn.ModuleList([GAT(args.enc_hidden_size, args.enc_hidden_size//args.n_head, args.n_head, attn_drop=args.attn_drop, trans=False) for _ in range(args.prop)]) #untested
        else:
            self.gat = nn.ModuleList([GAT(args.nhid, args.nhid//args.n_head, args.n_head, attn_drop=args.attn_drop, ffn_drop=args.drop, trans=True) for _ in range(args.prop)])
        self.prop = args.prop

    def forward(self, sent_enc, sent_num_mask, ent_enc, ent_num_mask, rel_emb, rel_mask, type_emb, graphs, text_len, ent_len):
        device = ent_enc.device
        graphs = graphs.to(device)
        ent_num_mask = (ent_num_mask==0) # reverse mask
        rel_mask = (rel_mask==0)
        sent_num_mask = (sent_num_mask == 0)

        #sent_enc   [19, 50, 256]
        #ent_enc [19,132,256]
        #rel_emb [19,200,256]
        init_h = []

        for i in range(graphs.batch_size):
            # add sentence embedding
            init_h.append(sent_enc[i][sent_num_mask[i]])
            # add ent embedding
            init_h.append(ent_enc[i][ent_num_mask[i]])
            # add rel embedding
            init_h.append(rel_emb[i][rel_mask[i]])
            init_h.append(type_emb[i])

        # init_h.shape  [1889,256]
        init_h = torch.cat(init_h, 0)
        feats = init_h
        for i in range(self.prop):
            feats = self.gat[i](graphs, feats)
        g_sent = pad(feats.index_select(0, graphs.filter_nodes(lambda x: x.data['type']==NODE_TYPE['sentence']).to(device)).split(text_len), out_type='tensor')
        g_ent = pad(feats.index_select(0, graphs.filter_nodes(lambda x: x.data['type']==NODE_TYPE['entity']).to(device)).split(ent_len), out_type='tensor')
        return g_sent, g_ent
