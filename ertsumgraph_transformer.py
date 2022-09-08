import torch
from modules import BiLSTM, GraphTrans, GAT_Hetersum, MSA 
from torch import nn
from transformers import RobertaTokenizer, RobertaModel
from torch.nn.init import xavier_uniform_

import dgl
from module.embedding import Word_Embedding
from module.transformer_decoder import Phase1_TransformerDecoder, Phase2_TransformerDecoder, TransformerDecoder
from module.roberta import RobertaEmbedding
from module.EMNewEncoder import EMEncoder
from module.optimizer import Optimizer
import pdb

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps_bert,
            model_size=args.enc_hidden_size)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert_model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps_dec,
            model_size=args.enc_hidden_size)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert_model')]
    optim.set_parameters(params)


    return optim

def get_generator(dec_hidden_size, vocab_size, emb_dim, device):
    gen_func = nn.Softmax(dim=-1)
    ### nn.Sequential内部实现了forward函数
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, emb_dim),
        nn.LeakyReLU(),
        nn.Linear(emb_dim, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

class Roberta_model(nn.Module):
    def __init__(self, roberta_path, finetune=False):
        super(Roberta_model, self).__init__()
        print('Roberta initialized')
        self.model = RobertaModel.from_pretrained(roberta_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
        self.pad_id = self.tokenizer.pad_token_id
        self._embedding = self.model.embeddings.word_embeddings

        self.finetune = finetune

    def forward(self, input_ids):
        attention_mask = (input_ids != self.pad_id).float()
        if(self.finetune):
            return self.model(input_ids, attention_mask=attention_mask)
        else:
            self.eval()
            with torch.no_grad():
                return self.model(input_ids, attention_mask=attention_mask)

class ERTSumGraph(nn.Module):
    def __init__(self, args, word_padding_idx, vocab_size, device, checkpoint=None):
        super(ERTSumGraph, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.padding_idx = word_padding_idx
        self.device = device
        # need to encode the following nodes:
        # word embedding : use glove embedding
        # sentence encoder: bilstm (word)
        # doc encoder: bilstm (sentence)
        # entity encoder: bilstm (word)
        # relation embedding: initial embedding
        # type embedding: initial embedding 

        # use roberta
        
        src_embeddings = torch.nn.Embedding(self.vocab_size, self.args.emb_size, padding_idx=word_padding_idx)
        tgt_embeddings = torch.nn.Embedding(self.vocab_size, self.args.emb_size, padding_idx=word_padding_idx)
        if self.args.share_embeddings:
            tgt_embeddings.weight = src_embeddings.weight
        self.encoder = EMEncoder(self.args, self.device, src_embeddings, self.padding_idx, None)
        emb_dim = tgt_embeddings.weight.size(1)
        
        self.generator = get_generator(self.args.dec_hidden_size, self.vocab_size, emb_dim, self.device)
        if self.args.share_decoder_embeddings:
            self.generator[2].weight = tgt_embeddings.weight

        self.generator_ent = get_generator(self.args.dec_hidden_size, self.vocab_size, emb_dim, self.device)
        if self.args.share_decoder_embeddings:
            self.generator_ent[2].weight = tgt_embeddings.weight

        self.type_emb = nn.Embedding(len(args.type_vocab), args.enc_hidden_size, padding_idx=self.padding_idx)
        nn.init.xavier_normal_(self.type_emb.weight)

        self.decoder = TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.heads,
            d_ff=self.args.ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings, 
            generator=self.generator)

        self.phase1_decoder = Phase1_TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.heads,
            d_ff=self.args.ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings, 
            generator=self.generator)

        self.phase2_decoder = Phase2_TransformerDecoder(
            self.args.dec_layers,
            self.args.dec_hidden_size, heads=self.args.heads,
            d_ff=self.args.ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings, 
            generator=self.generator, generator_ent=self.generator_ent, type_emb = self.type_emb)

        if checkpoint is not None:
            # checkpoint['model']
            keys = list(checkpoint['model'].keys())
            print('keys为:'+str(keys))
            for k in keys:
                if ('a_2' in k):
                    checkpoint['model'][k.replace('a_2', 'weight')] = checkpoint['model'][k]
                    del (checkpoint['model'][k])
                if ('b_2' in k):
                    checkpoint['model'][k.replace('b_2', 'bias')] = checkpoint['model'][k]
                    del (checkpoint['model'][k])

            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for n, p in self.named_parameters():
                if 'RobertaModel' not in n:
                    if p.dim() > 1:
                        xavier_uniform_(p)


        self.to(device)

        
    def forward(self,  batch):
        tgt = batch['tgt']
        tgt_template = batch['template_input']
        tgt_template_out = batch['template_target']
        #tgt_extend = batch['tgt_extend']
        #src = batch['text']
        src_extend = batch['text_extend']
        ent_extend = batch['ent_text_extend']
        edge = batch['edges']
        ent_score = batch['ent_score']
        extra_zeros = batch['extra_zeros']
        
        
        sent_state, sent_context, ent_state, ent_context = self.encoder(batch)
        
        dec_state = self.phase1_decoder.init_decoder_state(src_extend, sent_context)     # src: num_paras_in_one_batch x max_length
        
        phase1_decoder_outputs = self.phase1_decoder(tgt_template,sent_context, ent_extend, ent_context, ent_score, dec_state)

        
        tgt_len, batch_size = tgt.size()
        src_words = src_extend.view(src_extend.size(0),-1)
        src_words = src_words.unsqueeze(0).expand(tgt_len, batch_size,src_words.size(1)).contiguous()
        if extra_zeros is not None:
            extra_zeros = extra_zeros.unsqueeze(0).expand(tgt_len, batch_size,extra_zeros.shape[1]).contiguous()
        else:
            extra_zeros = None
        phase1_digits_probs = self.phase1_decoder.get_normalized_probs(src_words, extra_zeros, 
             phase1_decoder_outputs[0], phase1_decoder_outputs[1]['attn'], phase1_decoder_outputs[1]['copy_or_generate'],dim=2)

        #decoder output of phase 1
        phase1_digits = phase1_digits_probs.max(2)[1]   #.transpose(0,1).unsqueeze(1).contiguous()
        mask_tgt = tgt_template.data.eq(self.padding_idx).bool()
        phase1_digits[mask_tgt] = self.padding_idx
        phase1_digits = phase1_digits.transpose(0,1).unsqueeze(1).contiguous()

        unknown_ids = torch.zeros(phase1_digits.shape, device= phase1_digits.device, dtype=torch.long)
        phase1_digits = torch.where(phase1_digits <self.vocab_size, phase1_digits, unknown_ids)
        phase1_feature, phase1_context, _ = self.encoder.sent_encoder(phase1_digits)
        
        dec_state2 = self.phase2_decoder.init_decoder_state(src_extend, sent_context)
        phase2_decoder_outputs = self.phase2_decoder(tgt,sent_context, ent_extend, ent_context, ent_score, phase1_digits, phase1_context, dec_state2)

        dec_state = self.decoder.init_decoder_state(src_extend, sent_context)     # src: num_paras_in_one_batch x max_length
        decoder_outputs = self.decoder(tgt,sent_context, ent_extend, ent_context, ent_score, dec_state)
        return phase1_decoder_outputs, phase2_decoder_outputs, decoder_outputs