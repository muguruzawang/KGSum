#!/usr/bin/python
# -*- coding: utf-8 -*-

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

###这个版本的代码利用roberta的BPE编码方式来tokenize句子

import argparse
import datetime
import os
import shutil
import time
import pandas as pd
import dgl
import numpy as np
import torch
from rouge import Rouge
import pdb
import signal
import glob

from Tester import SLTester

from module.embedding import Word_Embedding
import torch.nn.functional as F
from module.vocabulary import WordVocab
from module.ERTvocabulary import Vocab
import torch.nn as nn

import time
from tqdm import tqdm

#from module.utils import get_datasets
from module.opts import vocab_config, fill_config, get_args
from ertsumgraph_transformer import ERTSumGraph,build_optim_bert,build_optim_dec

from tools.logger import *
from tools import utils

from module.utlis_dataloader import *
from utils import distributed

from transformers import RobertaTokenizer, RobertaModel
from module.trainer_builder import build_trainer
from module.predictor_builder import build_predictor
from utils.logging import init_logger, logger
from module.optimizer import Optimizer

_DEBUG_FLAG_ = False
global val_loss
val_loss = 2**31


model_flags = [ 'emb_size', 'enc_hidden_size', 'dec_hidden_size', 'enc_layers', 'dec_layers', 'block_size',  'heads', 'ff_size', 'hier',
               'inter_layers', 'inter_heads', 'block_size']

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def build_optim(args, model, checkpoint):
    """ Build optimizer """
    if checkpoint is not None:
        optim = checkpoint['optim'][0]
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
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps, model_size=args.enc_hidden_size)
    '''
    optim.set_parameters(list(model.named_parameters()))
    if args.train_from != '':
        optim.optimizer.load_state_dict(checkpoint['optim'])
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")
    '''
    optim.set_parameters(list(model.named_parameters()))
    return optim

def multi_main(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i

        procs.append(mp.Process(target=run, args=(args,
            device_id, error_queue), daemon=False))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()

def main(args):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1
    init_logger(args.log_file)
    if (args.mode == 'train'):
        train(args, device_id,)
    elif (args.mode == 'test'):
        step = int(args.test_from.split('.')[-2].split('_')[-1])
        # validate(args, device_id, args.test_from, step)
        test(args, args.test_from, step)
    elif (args.mode == 'validate'):
        wait_and_validate(args, device_id)
    # elif (args.mode == 'baseline'):
    #     baseline()
    elif (args.mode == 'print_flags'):
        print_flags()
    # elif (args.mode == 'stats'):
    #     stats()


def train(args,device_id):
    init_logger(args.log_file)
    logger.info(str(args))
    logger.info('################now the device_id is: %d'%device_id)

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])

    else:
        checkpoint = None

    '''
    train_dataset = ExampleSet(args.fnames[0], args.fnames[3], tokenizer, args.rel_vocab, args.sent_max_len, args.doc_max_timesteps, device)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, \
                        sampler = train_sampler, pin_memory = True, collate_fn=train_dataset.batch_fn)
    '''
    model = ERTSumGraph(args, args.word_padding_idx, args.vocab_size, device, checkpoint)

    if (args.sep_optim):
        optim_bert = build_optim_bert(args, model, checkpoint)
        optim_dec = build_optim_dec(args, model, checkpoint)
        optim = [optim_bert, optim_dec]
    else:
        optim = [build_optim(args, model, checkpoint)]

    symbols = {'BOS':2,'EOS':3,'PAD':1,'UNK':0}
    #optim = build_optim(args, model, checkpoint)
    logger.info(model)
    trainer = build_trainer(args, device_id, model, symbols, args.vocab_size, optim, device)
    
    trainer.train(args.train_dataloader, args.train_steps)


def wait_and_validate(args, device_id):
    timestep = 0
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime, reverse=True)
        ppl_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            ppl = validate(args, device_id, cp, step)
            ppl_lst.append((ppl, cp))
            max_step = ppl_lst.index(min(ppl_lst))
            '''
            if (i - max_step > 5):
                break
            ''' 
        ppl_lst = sorted(ppl_lst, key=lambda x: x[0])
        logger.info('Xent %s' % str(ppl_lst))
        for pp, cp in ppl_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            #test(args, cp, step)
    else:
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args,device_id, cp, step)
                    test(args,cp, step)
                    time.sleep(1200)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)

def validate(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)


    opt = vars(checkpoint['opt'])

    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])

    model = ERTSumGraph(args, args.word_padding_idx, args.vocab_size, device, checkpoint)
    model.to(device)

    model.eval()

    symbols = {'BOS':2,'EOS':3,'PAD':1,'UNK':0}
    trainer = build_trainer(args, device_id, model, symbols, args.vocab_size, None, device)
    stats = trainer.validate(args.valid_dataloader)
    trainer._report_step(0, step, valid_stats=stats)
    return stats.xent()


def test(args, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"

    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])

    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    # vocab = spm
    model = ERTSumGraph(args, args.word_padding_idx, args.vocab_size, device, checkpoint)
    model.to(device)

    model.eval()
    symbols = {'BOS':2,'EOS':3,'PAD':1,'UNK':0}
    predictor = build_predictor(args, args.wordvocab, symbols, model, device, logger=logger)
    predictor.translate(args.test_dataloader, step)

    # trainer.train(train_iter_fct, valid_iter_fct, FLAGS.train_steps, FLAGS.valid_steps)


def print_flags(args):
    checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
    print(checkpoint['opt'])



def run(args, device_id, error_queue):
    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' %gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")
        train(args,device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)

if __name__ == '__main__':
    args = get_args()
    args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(',')]
    args.inter_layers = [int(i) for i in args.inter_layers.split(',')]

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

    VOCAL_FILE = os.path.join(args.cache_dir, "vocab_prompt2")
    logger.info("[INFO] Create Context Vocab, vocab path is %s", VOCAL_FILE)
    wordvocab = WordVocab(VOCAL_FILE, args.vocab_size)
    filter_word_path = os.path.join(args.cache_dir, "filter_word.txt")

    ENTVOCAL_FILE = os.path.join(args.cache_dir, "entityVocab2")
    logger.info("[INFO] Create Entity Vocab, vocab path is %s", ENTVOCAL_FILE)
    entityvocab = WordVocab(ENTVOCAL_FILE, args.vocab_size)

    word_padding_idx = wordvocab.word2id('<PAD>')

    logger.info("[INFO] Create Relation Vocab.......")
    typelist = ['task', 'method', 'metric', 'material', 'otherscientificterm', 'generic']
    rellist = ['USED-FOR','CONJUNCTION','HYPONYM-OF','COMPARE','FEATURE-OF','EVALUATE-FOR','PART-OF', 'Coreference']
    rellist = ['--root--'] + sum([[x,x+'_INV'] for x in rellist], [])

    rel_vocab =  Vocab(sp=['<UNK>','<PAD>'])
    type_vocab = Vocab(sp=['<UNK>','<PAD>'])
    rel_vocab.update(rellist)
    rel_vocab.build()

    logger.info("[INFO] Create Type Vocab.......")
    type_vocab.update(typelist)
    type_vocab.build()
    
    train_text_file = os.path.join(args.data_dir, "train.label.jsonl")
    train_ent_file = os.path.join(args.cache_dir, "train.ent_type_relation.jsonl")
    
    val_text_file = os.path.join(args.data_dir, "val.label.jsonl")
    val_ent_file = os.path.join(args.cache_dir, "val.ent_type_relation.jsonl")

    '''
    test_text_file = os.path.join(args.data_dir, "val.label.jsonl")
    test_ent_file = os.path.join(args.cache_dir, "val.ent_type_relation.jsonl")
    
    '''
    test_text_file = os.path.join(args.data_dir, "test.label.jsonl")
    test_ent_file = os.path.join(args.cache_dir, "test.ent_type_relation.jsonl")
    
    train_template_file = os.path.join(args.data_dir, "train.ent_promptsummary.jsonl")
    
    val_template_file = os.path.join(args.data_dir, "val.ent_promptsummary.jsonl")

    test_template_file = os.path.join(args.data_dir, "test.ent_promptsummary.jsonl")
    
    train_entscore_file = os.path.join(args.cache_dir, "train.ent_importance_score.jsonl")
    
    val_entscore_file = os.path.join(args.cache_dir, "val.ent_importance_score.jsonl")
    
    test_entscore_file = os.path.join(args.cache_dir, "test.ent_importance_score.jsonl")

    train_entoracle_file = os.path.join(args.cache_dir, "train.ent_oracle.jsonl")

    test_entoracle_file = os.path.join(args.cache_dir, "test.ent_oracle.jsonl")

    val_entoracle_file = os.path.join(args.cache_dir, "val.ent_oracle.jsonl")

    train_sourcetype_file = os.path.join(args.cache_dir, "train.source_word_type.jsonl")

    val_sourcetype_file = os.path.join(args.cache_dir, "val.source_word_type.jsonl")
    
    test_sourcetype_file = os.path.join(args.cache_dir, "test.source_word_type.jsonl")

    '''
    val_text_file = os.path.join(args.data_dir, "train.label19.jsonl")
    val_ent_file = os.path.join(args.cache_dir, "train.ent_type_relation19.jsonl")
    
    test_text_file = os.path.join(args.data_dir, "train.label19.jsonl")
    test_ent_file = os.path.join(args.cache_dir, "train.ent_type_relation19.jsonl")
    '''
    args.fnames = [train_text_file, val_text_file, test_text_file, train_ent_file, val_ent_file, test_ent_file]
    
    
    args = vocab_config(args, rel_vocab, type_vocab, wordvocab, entityvocab)
    args.word_padding_idx = word_padding_idx
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    args.device =  device
    
    if (args.mode == 'train'):
        train_dataset = ExampleSet(train_text_file, train_ent_file, train_sourcetype_file, train_template_file, train_entscore_file, train_entoracle_file, wordvocab, entityvocab,  rel_vocab,type_vocab, args.sent_max_len, args.doc_max_timesteps, args.device)
        #train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, \
                            pin_memory = True, collate_fn=train_dataset.batch_fn)
        args.train_dataloader = train_dataloader

    elif (args.mode == 'validate'):
        valid_dataset = ExampleSet(val_text_file, val_ent_file,val_sourcetype_file, val_template_file, val_entscore_file,val_entoracle_file, wordvocab , entityvocab, rel_vocab,type_vocab, args.sent_max_len, args.doc_max_timesteps, args.device)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, \
                            pin_memory = True, collate_fn=valid_dataset.batch_fn)
        args.valid_dataloader = valid_dataloader

        test_dataset = ExampleSet(test_text_file, test_ent_file, test_sourcetype_file, test_template_file, test_entscore_file,test_entoracle_file, wordvocab,entityvocab, rel_vocab,type_vocab, args.sent_max_len, args.doc_max_timesteps, args.device)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.num_workers, \
                            pin_memory = True, collate_fn=test_dataset.batch_fn)
        args.test_dataloader = test_dataloader
    
    elif (args.mode == 'test'):
        test_dataset = ExampleSet(test_text_file, test_ent_file,test_sourcetype_file, test_template_file, test_entscore_file, test_entoracle_file, wordvocab,entityvocab, rel_vocab, type_vocab,args.sent_max_len, args.doc_max_timesteps, args.device)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.num_workers, \
                            pin_memory = True, collate_fn=test_dataset.batch_fn)
        args.test_dataloader = test_dataloader
    # pass ERT_Representation is not the model, we still need a main function

    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpus
    if(args.world_size>1):
        multi_main(args)
    else:
        main(args)
