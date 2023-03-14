import torch
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def fill_config(args):
    # dirty work
    #args.device = torch.device(args.gpu)
    args.dec_ninp = args.nhid * 3  
    #args.fnames = [args.train_file, args.valid_file, args.test_file]
    return args


def vocab_config(args, rel_vocab, type_vocab, wordvocab):
    # dirty work
    args.rel_vocab = rel_vocab
    args.type_vocab = type_vocab
    args.wordvocab = wordvocab

    return args

def get_args():
    parser = argparse.ArgumentParser(description='Graph Writer in DGL')
    # Where to find data
    parser.add_argument('--data_dir', type=str, default='data/CNNDM',help='The dataset directory.')
    parser.add_argument('--cache_dir', type=str, default='cache/CNNDM',help='The processed dataset directory')
    # Important settings
    parser.add_argument('--restore_model', type=str, default='None', help='Restore model for further training. [bestmodel/bestFmodel/earlystop/None]')

    # Where to save output
    parser.add_argument('--save_root', type=str, default='save/', help='Root directory for all model.')
    parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')

    parser.add_argument('--test_model', type=str, default='evalbestmodel', help='choose different model to test [multi/evalbestmodel/trainbestmodel/earlystop]')

    # Hyperparameters
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. [default: 0]')
    parser.add_argument('--cuda', action='store_true', default=False, help='GPU or CPU [default: False]')
    parser.add_argument('--vocab_size', type=int, default=50000,help='Size of vocabulary. [default: 50000]')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs [default: 20]')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 8]')
    parser.add_argument('--n_iter', type=int, default=1, help='iteration hop [default: 1]')

    parser.add_argument('--word_embedding', action='store_true', default=False, help='whether to use Word embedding [default: True]')
    parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding size [default: 300]')
    parser.add_argument('--sent_enc_size', type=int, default=300,help='Size of LSTM hidden states')
    parser.add_argument('--doc_enc_size', type=int, default=300,help='Size of LSTM hidden states')
    parser.add_argument('--embed_train', action='store_true', default=False,help='whether to train Word embedding [default: False]')
    parser.add_argument('--feat_embed_size', type=int, default=50, help='feature embedding size [default: 50]')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of GAT layers [default: 1]')
    parser.add_argument('--lstm_hidden_state', type=int, default=128, help='size of lstm hidden state [default: 128]')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of lstm layers [default: 2]')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='whether to use bidirectional LSTM [default: True]')
    parser.add_argument('--n_feature_size', type=int, default=128, help='size of node feature [default: 128]')
    parser.add_argument('--ffn_inner_hidden_size', type=int, default=512,help='PositionwiseFeedForward inner hidden size [default: 512]')
    parser.add_argument('--n_head', type=int, default=8, help='multihead attention number [default: 8]')
    parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1,help='recurrent dropout prob [default: 0.1]')
    parser.add_argument('--atten_dropout_prob', type=float, default=0.1, help='attention dropout prob [default: 0.1]')
    parser.add_argument('--ffn_dropout_prob', type=float, default=0.1,help='PositionwiseFeedForward dropout prob [default: 0.1]')
    parser.add_argument('--use_orthnormal_init', action='store_true', default=True,help='use orthnormal init for lstm [default: True]')
    parser.add_argument('--sent_max_len', type=int, default=100,help='max length of sentences (max source text sentence tokens)')
    parser.add_argument('--doc_max_timesteps', type=int, default=50,help='max length of documents (max timesteps of documents)')
    parser.add_argument('-sent_dropout', type=float, default=0.3,help='Dropout probability; applied between LSTM stacks.')
    parser.add_argument('-doc_dropout', type=float, default=0.3,help='Dropout probability; applied between LSTM stacks.')

    # Training
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lr_descent', action='store_true', default=False, help='learning rate descent')
    parser.add_argument('--grad_clip', action='store_true', default=False, help='for gradient clipping')
    parser.add_argument('-m', type=int, default=3, help='decode summary length')
    
    ###################################################################################
    parser.add_argument('--nhid', default=256, type=int, help='hidden size')
    parser.add_argument('--nhead', default=4, type=int, help='number of heads')
    parser.add_argument('--head_dim', default=75, type=int, help='head dim')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument('--prop', default=6, type=int, help='number of layers of gnn')
    parser.add_argument('--test', action='store_true', help='inference mode')
    parser.add_argument('--title', action='store_true', help='use title input')
    parser.add_argument('--epoch', default=20, type=int, help='training epoch')
    parser.add_argument('--beam_max_len', default=200, type=int, help='max length of the generated text')
    parser.add_argument('--enc_lstm_layers', default=2, type=int, help='number of layers of lstm')
    parser.add_argument('--clip', default=1, type=float, help='gradient clip')
    parser.add_argument('--emb_drop', default=0.0, type=float, help='embedding dropout')
    parser.add_argument('--attn_drop', default=0.1, type=float, help='attention dropout')
    parser.add_argument('--drop', default=0.1, type=float, help='dropout')
    parser.add_argument('--lp', default=1.0, type=float, help='length penalty')
    parser.add_argument('--graph_enc', default='gtrans', type=str, help='gnn mode, we only support the graph transformer now')
    parser.add_argument('--save_dataset', default='', type=str, help='save path of dataset')
    parser.add_argument('--save_model', default='saved_model.pt', type=str, help='save path of model')

    parser.add_argument('--save_label', action='store_true', default=False, help='require multihead attention')
    parser.add_argument('--limited', action='store_true', default=False, help='limited hypo length')
    parser.add_argument('--blocking', action='store_true', default=False, help='ngram blocking')
    parser.add_argument('--use_pyrouge', action='store_true', default=False, help='use_pyrouge')

    ###########################################from openNMT###############################################
    parser.add_argument('--log_file', default='', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--visible_gpus', default='--1', type=str)
    parser.add_argument('--model_path', default='../../models', type=str)
    parser.add_argument('--train_from', default='', type=str)

    parser.add_argument('--trunc_src_ntoken', default=500, type=int)
    parser.add_argument('--trunc_tgt_ntoken', default=200, type=int)

    parser.add_argument('--emb_size', default=256, type=int)
    parser.add_argument('--enc_layers', default=4, type=int)
    parser.add_argument('--dec_layers', default=1, type=int)
    parser.add_argument('--enc_dropout', default=6, type=float)
    parser.add_argument('--dec_dropout', default=0, type=float)
    parser.add_argument('--enc_hidden_size', default=256, type=int)
    parser.add_argument('--dec_hidden_size', default=256, type=int)
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--ff_size', default=1024, type=int)
    parser.add_argument("--hier", type=str2bool, nargs='?',const=True,default=True)

    parser.add_argument('--valid_batch_size', default=16, type=int)
    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--max_grad_norm', default=0, type=float)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--train_steps', default=20, type=int)
    parser.add_argument('--save_checkpoint_steps', default=20, type=int)
    parser.add_argument('--report_every', default=100, type=int)


    # multi--gpu
    parser.add_argument('--accum_count', default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--gpu_ranks', default='0', type=str)

    # don't need to change flags
    parser.add_argument("--share_embeddings", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("--share_decoder_embeddings", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument('--max_generator_batches', default=32, type=int)

    # flags for  testing
    parser.add_argument("--test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument('--test_from', default='../../results', type=str)
    parser.add_argument('--result_path', default='../../results', type=str)
    parser.add_argument('--alpha', default=0, type=float)
    parser.add_argument('--length_penalty', default='wu', type=str)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--n_best', default=1, type=int)
    parser.add_argument('--max_length', default=250, type=int)
    parser.add_argument('--min_length1', default=110, type=int)
    parser.add_argument('--min_length2', default=110, type=int)
    parser.add_argument("--report_rouge", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--max_wiki', default=5, type=int)

    # flags for  hier
    # flags.DEFINE_boolean('old_inter_att', False, 'old_inter_att')
    parser.add_argument('--inter_layers', default='0', type=str)
    parser.add_argument('--gat_iter', default=2, type=str)

    parser.add_argument('--inter_heads', default=8, type=int)
    parser.add_argument('--trunc_src_nblock', default=24, type=int)

    # flags for  graph
    parser.add_argument('--use_bert', default=True, type=str2bool)


    # flags for  learning
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.998, type=float)
    parser.add_argument('--warmup_steps', default=8000, type=int)
    parser.add_argument('--decay_method', default='noam', type=str)
    parser.add_argument('--label_smoothing', default=0.0, type=float)

    # fine-tune roberta and two separate warmup steps
    parser.add_argument("--finetune_bert", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--warmup_steps_bert", default=20000, type=int)
    parser.add_argument("--warmup_steps_dec", default=10000, type=int)
    parser.add_argument("--sep_optim", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("--lr_bert", default=2e-3, type=float)
    parser.add_argument("--lr_dec", default=0.2, type=float)
    parser.add_argument('--roberta_path', type=str, default='/data/home/scv0028/run/wpc/huggingface/roberta-base',help='The roberta model path.')
    parser.add_argument("--num_workers", default=8, type=int)
    ###是否设置层次transformer decoder
    parser.add_argument("--hier_decoder", type=str2bool, nargs='?',const=False,default=False)
    ###是否使用nucleus-sampling代替beam search 
    parser.add_argument("--use_nucleus_sampling", type=str2bool, nargs='?',const=False,default=False)
    parser.add_argument("--no_repeat_ngram_size2", default=0, type=int, help='ngram blocking parameter, to avoid repeat')
    parser.add_argument("--no_repeat_ngram_size1", default=3, type=int, help='ngram blocking parameter, to avoid repeat')

    ###是否使用copy mechanism
    parser.add_argument("--generate_and_copy", type=str2bool, nargs='?',const=False,default=False)
    parser.add_argument("--block_bigram_range", default=0, type=int, help='bigram blocking in range n, to avoid repeat')
    parser.add_argument('--unk_penalty', default=-1.0, type=float)
    parser.add_argument('--entloss_weight', default=1.0, type=float)

    args = parser.parse_args()
    args = fill_config(args)
    return args
