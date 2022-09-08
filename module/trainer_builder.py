from datetime import datetime

import torch
import os

from module.loss_two_phase import build_loss_compute
from tensorboardX import SummaryWriter

from utils import distributed
from utils.logging import logger
from utils.report_manager import ReportMgr
from utils.statistics import Statistics
from module.utlis_dataloader import load_to_cuda
import pdb

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'RobertaModel' not in name and (param.requires_grad == True):
            if 'encoder' in name:
                enc += param.nelement()
            elif 'decoder' or 'generator' in name:
                dec += param.nelement()
    return n_params, enc, dec


def build_trainer(args, device_id, model, symbols, vocab_size,
                  optim, device):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    device = "cpu" if args.visible_gpus == '-1' else "cuda"

    train_loss = build_loss_compute(
        model.phase1_decoder, model.type_emb, model.phase2_decoder,model.decoder,args.entloss_weight, symbols, vocab_size, device, train=True, label_smoothing=args.label_smoothing)
    valid_loss = build_loss_compute(
        model.phase1_decoder, model.type_emb, model.phase2_decoder,model.decoder, args.entloss_weight, symbols, vocab_size, train=False, device=device)

    shard_size = args.max_generator_batches
    grad_accum_count = args.accum_count
    n_gpu = args.world_size
    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    # if(gpu_rank==0):
    #     report_manager = ReportMgr(FLAGS.report_every, start_time=-1, tensorboard_writer=writer)
    # else:
    #     report_manager = None
    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, model, train_loss, valid_loss, optim, device,
                      shard_size,
                      grad_accum_count, n_gpu, gpu_rank, report_manager)

    # print(tr)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(self, args, model, train_loss, valid_loss, optim, device,
                 shard_size=32, grad_accum_count=1, n_gpu=1, gpu_rank=1,report_manager=None):
        # Basic attributes.
        self.args = args
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.shard_size = shard_size
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.device = device

        assert grad_accum_count > 0
        self.model.train()

    def train(self, train_iter, train_steps):
        logger.info('Start training...')

        step = self.optim[0]._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        normalization_ent = 0

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:

            reduce_counter = 0
            print(train_iter)
            for i, batch in enumerate(train_iter):
                batch = load_to_cuda(batch,self.device)
                #print('##########batchçš„deviceä¸ºï¼š%s'%batch['text'].device)
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):

                    true_batchs.append(batch)
                    
                    # num_tokens = batch.tgt[1:].ne(
                    #     self.train_loss.padding_idx).sum()
                    num_tokens = batch['tgt_extend'].ne(
                        self.train_loss.padding_idx).sum()
                    normalization += num_tokens.item()
                    num_enttokens = batch['template_target'].ne(
                        self.train_loss.padding_idx).sum()
                    normalization_ent += num_enttokens.item()
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed
                                                .all_gather_list
                                                (normalization))
                            normalization_ent = sum(distributed
                                                .all_gather_list
                                                (normalization_ent))


                        self._gradient_accumulation(
                            true_batchs, normalization,normalization_ent, total_stats,
                            report_stats)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim[0].learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        normalization_ent = 0
                        if (step % self.args.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)

                        step += 1
                        if step > train_steps:
                            break

        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        self.model.eval()

        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:
                batch = load_to_cuda(batch,self.device)
                src = batch['text_extend']
                ent = batch['ent_text']
                edge = batch['edges']
                tgt = batch['tgt_extend']
                outputs1, outputs2,outputs = self.model(batch)

                batch_stats = self.valid_loss.monolithic_compute_loss(
                    batch, outputs1, outputs2,outputs)
                stats.update(batch_stats)
            return stats

    def _gradient_accumulation(self, true_batchs, normalization,normalization_ent, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:

            src = batch['text']
            ent = batch['ent_text']
            edge = batch['edges']
            tgt = batch['tgt_extend']

            if self.grad_accum_count == 1:
                self.model.zero_grad()
            
            outputs1, outputs2, outputs = self.model(batch)
            
            # 3. Compute loss in shards for memory efficiency.
            batch_stats = self.train_loss.sharded_compute_loss(
                batch, outputs1, outputs2, outputs, self.shard_size, normalization, normalization_ent)

            report_stats.n_src_words += batch['text'].nelement()

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                
                for o in self.optim:
                    o.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            for o in self.optim:
                o.step()

    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate1,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate1, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
