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

import argparse
import datetime
import os
import time
import json

import torch
import torch.nn as nn
from rouge import Rouge

from Tester import SLTester
from module.dataloader import ExampleSet, MultiExampleSet, graph_collate_fn
from module.embedding import Word_Embedding
from module.vocabulary import WordVocab
from tools import utils
from tools.logger import *

from module.ERTvocabulary import Vocab

#from module.utils import get_datasets
from module.opts import vocab_config
from ertsumgraph import ERTSumGraph

from tools.logger import *

from module.utlis_dataloader import *


def load_test_model(model, model_name, eval_dir, save_root):
    """ choose which model will be loaded for evaluation """
    if model_name.startswith('eval'):
        bestmodel_load_path = os.path.join(eval_dir, model_name[4:])
    elif model_name.startswith('train'):
        train_dir = os.path.join(save_root, "train")
        bestmodel_load_path = os.path.join(train_dir, model_name[5:])
    elif model_name == "earlystop":
        train_dir = os.path.join(save_root, "train")
        bestmodel_load_path = os.path.join(train_dir, 'earlystop')
    else:
        logger.error("None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
        raise ValueError("None of such model! Must be one of evalbestmodel/trainbestmodel/earlystop")
    if not os.path.exists(bestmodel_load_path):
        logger.error("[ERROR] Restoring %s for testing...The path %s does not exist!", model_name, bestmodel_load_path)
        return None
    logger.info("[INFO] Restoring %s for testing...The path is %s", model_name, bestmodel_load_path)

    model.load_state_dict(torch.load(bestmodel_load_path))

    return model

def run_test(model, dataset, loader, model_name, hps):
    test_dir = os.path.join(hps.save_root, "test") # make a subdir of the root dir for eval data
    eval_dir = os.path.join(hps.save_root, "eval")
    if not os.path.exists(test_dir) : os.makedirs(test_dir)
    if not os.path.exists(eval_dir) :
        logger.exception("[Error] eval_dir %s doesn't exist. Run in train mode to create it.", eval_dir)
        raise Exception("[Error] eval_dir %s doesn't exist. Run in train mode to create it." % (eval_dir))

    resfile = None
    if hps.save_label:
        log_dir = os.path.join(test_dir, hps.cache_dir.split("/")[-1])
        resfile = open(log_dir, "w")
        logger.info("[INFO] Write the Evaluation into %s", log_dir)

    model = load_test_model(model, model_name, eval_dir, hps.save_root)
    model.eval()

    iter_start_time=time.time()
    with torch.no_grad():
        logger.info("[Model] Sequence Labeling!")
        tester = SLTester(model, hps.m, limited=hps.limited, test_dir=test_dir)

        for i, batch in enumerate(loader):
            G = batch['graph']
            if hps.cuda:
                G = G.to(torch.device("cuda"))
            tester.evaluation(batch, dataset, blocking=hps.blocking)

    running_avg_loss = tester.running_avg_loss

    if hps.save_label:
        # save label and do not calculate rouge
        json.dump(tester.extractLabel, resfile)
        tester.SaveDecodeFile()
        logger.info('   | end of test | time: {:5.2f}s | '.format((time.time() - iter_start_time)))
        return

    logger.info("The number of pairs is %d", tester.rougePairNum)
    if not tester.rougePairNum:
        logger.error("During testing, no hyps is selected!")
        sys.exit(1)

    if hps.use_pyrouge:
        if isinstance(tester.refer[0], list):
            logger.info("Multi Reference summaries!")
            scores_all = utils.pyrouge_score_all_multi(tester.hyps, tester.refer)
        else:
            scores_all = utils.pyrouge_score_all(tester.hyps, tester.refer)
    else:
        rouge = Rouge()
        scores_all = rouge.get_scores(tester.hyps, tester.refer, avg=True)

    res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
            + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
                + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
    logger.info(res)

    tester.getMetric()
    tester.SaveDecodeFile()
    logger.info('[INFO] End of test | time: {:5.2f}s | test loss {:5.4f} | '.format((time.time() - iter_start_time),float(running_avg_loss)))

def main():
    parser = argparse.ArgumentParser(description='HeterSumGraph Model')

    # Where to find data
    parser.add_argument('--data_dir', type=str, default='data/CNNDM', help='The dataset directory.')
    parser.add_argument('--cache_dir', type=str, default='cache/CNNDM', help='The processed dataset directory')
    parser.add_argument('--embedding_path', type=str, default='/data/home/scv0028/run/wpc/survey_generation/NeuSum/neusum_pt/data/cnndm/glove/glove.6B.300d.txt', help='Path expression to external word embedding.')

    # Important settings
    parser.add_argument('--model', type=str, default="HSumGraph", help="model structure[HSG|HDSG]")
    parser.add_argument('--test_model', type=str, default='evalbestmodel', help='choose different model to test [multi/evalbestmodel/trainbestmodel/earlystop]')
    parser.add_argument('--use_pyrouge', action='store_true', default=False, help='use_pyrouge')

    # Where to save output
    parser.add_argument('--save_root', type=str, default='save/', help='Root directory for all model.')
    parser.add_argument('--log_root', type=str, default='log/', help='Root directory for all logging.')

    # Hyperparameters
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    parser.add_argument('--cuda', action='store_true', default=False, help='use cuda')
    parser.add_argument('--vocab_size', type=int, default=50000, help='Size of vocabulary.')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
    parser.add_argument('--n_iter', type=int, default=1, help='iteration ')
    parser.add_argument('--nhid', default=300, type=int, help='hidden size')
    parser.add_argument('--emb_drop', default=0.0, type=float, help='embedding dropout')
    parser.add_argument('--enc_lstm_layers', default=2, type=int, help='number of layers of lstm')
    parser.add_argument('--graph_enc', default='gtrans', type=str, help='gnn mode, we only support the graph transformer now')
    parser.add_argument('--prop', default=6, type=int, help='number of layers of gnn')
    parser.add_argument('--attn_drop', default=0.1, type=float, help='attention dropout')
    parser.add_argument('--drop', default=0.1, type=float, help='dropout')
    parser.add_argument('--sent_enc_size', type=int, default=300,help='Size of LSTM hidden states')
    parser.add_argument('--doc_enc_size', type=int, default=300,help='Size of LSTM hidden states')

    parser.add_argument('--word_embedding', action='store_true', default=True, help='whether to use Word embedding')
    parser.add_argument('--word_emb_dim', type=int, default=300, help='Word embedding size [default: 300]')
    parser.add_argument('--embed_train', action='store_true', default=False, help='whether to train Word embedding [default: False]')
    parser.add_argument('--feat_embed_size', type=int, default=50, help='feature embedding size [default: 50]')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of GAT layers [default: 1]')
    parser.add_argument('--lstm_hidden_state', type=int, default=128, help='size of lstm hidden state')
    parser.add_argument('--lstm_layers', type=int, default=2, help='lstm layers')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='use bidirectional LSTM')
    parser.add_argument('--n_feature_size', type=int, default=128, help='size of node feature')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size [default: 64]')
    parser.add_argument('--gcn_hidden_size', type=int, default=128, help='hidden size [default: 64]')
    parser.add_argument('--ffn_inner_hidden_size', type=int, default=512, help='PositionwiseFeedForward inner hidden size [default: 512]')
    parser.add_argument('--n_head', type=int, default=8, help='multihead attention number [default: 8]')
    parser.add_argument('--recurrent_dropout_prob', type=float, default=0.1, help='recurrent dropout prob [default: 0.1]')
    parser.add_argument('--atten_dropout_prob', type=float, default=0.1,help='attention dropout prob [default: 0.1]')
    parser.add_argument('--ffn_dropout_prob', type=float, default=0.1, help='PositionwiseFeedForward dropout prob [default: 0.1]')
    parser.add_argument('--use_orthnormal_init', action='store_true', default=True, help='use orthnormal init for lstm [default: true]')
    parser.add_argument('--sent_max_len', type=int, default=100, help='max length of sentences (max source text sentence tokens)')
    parser.add_argument('--doc_max_timesteps', type=int, default=50, help='max length of documents (max timesteps of documents)')
    parser.add_argument('--save_label', action='store_true', default=False, help='require multihead attention')
    parser.add_argument('--limited', action='store_true', default=False, help='limited hypo length')
    parser.add_argument('--blocking', action='store_true', default=False, help='ngram blocking')

    parser.add_argument('-m', type=int, default=3, help='decode summary length')


    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_printoptions(threshold=50000)

    LOG_PATH = args.log_root

    # train_log setting
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_PATH, "train_" + nowTime)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Pytorch %s", torch.__version__)

    # create word rel type vocab
    VOCAL_FILE = os.path.join(args.cache_dir, "vocab")
    logger.info("[INFO] Create Vocab, vocab path is %s", VOCAL_FILE)
    wordvocab = WordVocab(VOCAL_FILE, args.vocab_size)
    filter_word_path = os.path.join(args.cache_dir, "filter_word.txt")

    logger.info("[INFO] Create Relation Vocab.......")
    typelist = ['Task', 'Method', 'Metric', 'Material', 'OtherScientificTerm', 'Generic']
    rellist = ['USED-FOR','CONJUNCTION','HYPONYM-OF','COMPARE','FEATURE-OF','EVALUATE-FOR','PART-OF', 'Coreference']
    rellist = sum([[x,x+'_INV'] for x in rellist], [])

    rel_vocab =  Vocab(sp=['<PAD>', '<UNK>'])
    type_vocab = Vocab(sp=['<UNK>'])
    rel_vocab.update(rellist)
    rel_vocab.build()

    logger.info("[INFO] Create Type Vocab.......")
    type_vocab.update(typelist)
    type_vocab.build()
    
    
    test_text_file = os.path.join(args.data_dir, "test.label.jsonl")
    test_ent_file = os.path.join(args.cache_dir, "test.ent_type_relation.jsonl")
    '''
    test_text_file = os.path.join(args.data_dir, "train.label19.jsonl")
    test_ent_file = os.path.join(args.cache_dir, "train.ent_type_relation19.jsonl")
    '''
    if args.cuda:
        args.device = torch.device('cuda:0')
    args = vocab_config(args, rel_vocab, type_vocab, wordvocab)

    test_dataset = ExampleSet(test_text_file, test_ent_file, wordvocab, rel_vocab, type_vocab, args.sent_max_len, args.doc_max_timesteps, filter_word_path, args.device)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=test_dataset.batch_fn)
      
    model = ERTSumGraph(args)
    if args.cuda:
        model.to(torch.device("cuda:0"))
        logger.info("[INFO] Use cuda")
    
    logger.info("[INFO] Decoding...")
    if args.test_model == "multi":
        for i in range(3):
            model_name = "evalbestmodel_%d" % i
            run_test(model, test_dataset, test_dataloader, model_name, args)
    else:
        run_test(model, test_dataset, test_dataloader, args.test_model, args)

if __name__ == '__main__':
    main()
