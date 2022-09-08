"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from utils.statistics import Statistics

'''
def build_loss_compute(decoder,symbols, vocab_size, device, train=True,label_smoothing = 0.0):
    compute = NMTLossCompute(
        decoder, symbols, vocab_size,
        label_smoothing=label_smoothing if train else 0.1)
    compute.to(device)

    return compute
'''
def build_loss_compute(decoder1, type_emb, decoder2,decoder, entloss_weight, symbols, vocab_size, device, train=True,label_smoothing = 0.0):
    compute = NMTLossCompute(
        decoder1, type_emb, decoder2,decoder, entloss_weight, symbols, vocab_size,
        label_smoothing=label_smoothing if train else 0.1)
    compute.to(device)

    return compute

class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, decoder1,decoder2, decoder,pad_id,entloss_weight):
        super(LossComputeBase, self).__init__()
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.decoder = decoder
        self.padding_idx = pad_id
        self.entloss_weight = entloss_weight



    def _make_shard_state(self, batch, output,  attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, outputs1, outputs2,outputs,
                            ):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        shard_state = self._make_shard_state(batch, outputs1, outputs2,outputs)
        _,_,_,_, batch_stats = self._compute_loss(batch, outputs1[1]['src'], outputs2[1]['src'], **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, outputs1, outputs2,outputs,
                              shard_size,
                             normalization,
                             normalization_ent):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics

        """
        batch_stats = Statistics()
        # output.shape = [211, 4, 256]
        # attn.shape = [211,4,3600]
        # copy_or_generate = [211,4,1]
        # src.shape = [4,3600]

        shard_state = self._make_shard_state(batch, outputs1, outputs2, outputs)
        for shard in shards(shard_state, shard_size):
            loss1, loss2,loss,loss_cross, stats = self._compute_loss(batch, outputs1[1]['src'], outputs2[1]['src'], **shard)
            loss1 = loss1.div(float(normalization_ent))
            loss2 = self.entloss_weight*loss2.div(float(normalization))
            loss = loss.div(float(normalization))
            loss_cross = loss_cross.div(float(normalization))
            loss_total = loss1 + loss2 + loss + loss_cross
            loss_total.backward()
            batch_stats.update(stats)

        return batch_stats

    def _stats(self,loss_temp,loss_ent, loss, scores_template, target_template, scores_ent, target_ent, scores_total, target_total):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred_temp = scores_template.max(1)[1]
        non_padding_temp = target_template.ne(self.padding_idx)
        num_correct_temp = pred_temp.eq(target_template) \
                          .masked_select(non_padding_temp) \
                          .sum() \
                          .item()
        num_non_padding_temp = non_padding_temp.sum().item()

        pred_ent = scores_ent.max(1)[1]
        non_padding_ent = target_ent.ne(self.padding_idx)
        num_correct_ent = pred_ent.eq(target_ent) \
                          .masked_select(non_padding_ent) \
                          .sum() \
                          .item()
        num_non_padding_ent = non_padding_ent.sum().item()

        pred = scores_total.max(1)[1]
        non_padding = target_total.ne(self.padding_idx)
        num_correct = pred.eq(target_total) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        return Statistics(loss_temp.item(),loss_ent.item(),loss.item(), num_non_padding_temp, num_correct_temp, \
                num_non_padding_ent,num_correct_ent,num_non_padding,num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
    
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')

class LabelSmoothingLoss2(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss2, self).__init__()

        self.label_smoothing = label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target, extra_len):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        smoothing_value = self.label_smoothing / (self.tgt_vocab_size + extra_len - 2)
        one_hot = torch.full((self.tgt_vocab_size + extra_len,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        one_hot = one_hot.to(output.device)

        model_prob = one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')

class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, decoder1, type_emb, decoder2, decoder, entloss_weight, symbols, vocab_size,
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(decoder1,decoder2,decoder, symbols['PAD'],entloss_weight)
        self.sparse = not isinstance(decoder1.generator[-1], nn.Softmax)
        self.type_emb = type_emb
        self.embeddings = decoder1.embeddings
        self.emb_dim = self.type_emb.weight.size(1)
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss2(
                label_smoothing, vocab_size, ignore_index=self.padding_idx
            )
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )
        self.criterion_kl_div =nn.KLDivLoss(reduction='sum')

    def _make_shard_state(self, batch, output1,output2,outputs):

       return {"output1": output1[0],
            "output2": output2[0],
            "target":batch['tgt_extend'],
            "target_template": batch['template_target'],
            "copy_attn1": output1[1]['attn'],
            "copy_or_generate1": output1[1]['copy_or_generate'],
            "copy_attn2": output2[1]['attn'],
            "copy_or_generate2": output2[1]['copy_or_generate'],
            "output": outputs[0],
            "copy_attn": outputs[1]['attn'],
            "copy_or_generate": outputs[1]['copy_or_generate']
        }

    def _compute_loss(self, batch, src_words1,src_words2, output1, output2, target, target_template, copy_attn1, copy_or_generate1,
                         copy_attn2, copy_or_generate2, output, copy_attn,copy_or_generate):
        ###phase 1 loss
        split_size = output1.shape[0]
        bottled_output1 = self._bottle(output1) #[32,4,256]->[128,256]
        bottled_copyattn1 = self._bottle(copy_attn1.contiguous()) #[32,4,217]
        bottled_cog1 = self._bottle(copy_or_generate1.contiguous()) #[32,4,1]
        bottled_output2 = self._bottle(output2) #[32,4,256]->[128,256]
        bottled_copyattn2 = self._bottle(copy_attn2.contiguous()) #[32,4,217]
        bottled_cog2 = self._bottle(copy_or_generate2.contiguous()) #[32,4,1]
        batch_size, src_len = src_words1.size()  #[4,2065]
        src_words1 = src_words1.unsqueeze(0).expand(split_size, batch_size ,src_len).contiguous() #[32,4,2065]
        bottled_src1 = self._bottle(src_words1)

        batch_size, src_len = src_words2.size()  #[4,2065]
        src_words2 = src_words2.unsqueeze(0).expand(split_size, batch_size ,src_len).contiguous() #[32,4,2065]
        bottled_src2 = self._bottle(src_words2)

        extra_zeros = batch['extra_zeros'] #[4,18]
        if extra_zeros is not None:
            batch_extra, extra_len = extra_zeros.size()
            ###需要扩展extra_len
            extra_zeros = extra_zeros.unsqueeze(0).expand(split_size, batch_extra ,extra_len).contiguous() #[32,4,18]
            bottled_extra_zeros = self._bottle(extra_zeros)
        else:
            bottled_extra_zeros = None
            extra_len = 0
        '''
        if self.sparse:
            # for sparsemax loss, the loss function operates on the raw output
            # vector, not a probability vector. Hence it's only necessary to
            # apply the first part of the generator here.
            scores = self.decoder.generator[0](bottled_output)
        else:
            scores = self.decoder.generator(bottled_output)
        '''
        scores1 = self.decoder1.get_normalized_probs(bottled_src1, bottled_extra_zeros, bottled_output1, bottled_copyattn1, bottled_cog1) #[128,50018]

        gtruth1 =target_template.contiguous().view(-1) #[128]  target [32,4]
        '''
        typeid_range = [4,5,6,7,8,9,10]
        is_type = torch.zeros(gtruth1.shape, device = gtruth1.device)
        for t in typeid_range:
            is_type += gtruth1.eq(t)

        is_type = ~is_type.bool()
        gtruth1[is_type] = 1
        '''
        loss1 = self.criterion(scores1, gtruth1, extra_len)

        scores2 = self.decoder2.get_normalized_probs(bottled_src2, bottled_extra_zeros, bottled_output2, bottled_copyattn2, bottled_cog2) #[128,50018]
        gtruth2 =target.contiguous().view(-1) #[128]  target [32,4]
        
        loss2 = self.criterion(scores2, gtruth2, extra_len)

        bottled_output = self._bottle(output)
        bottled_copyattn = self._bottle(copy_attn.contiguous())
        bottled_cog = self._bottle(copy_or_generate.contiguous())
        scores = self.decoder.get_normalized_probs(bottled_src2, bottled_extra_zeros, bottled_output, bottled_copyattn, bottled_cog)
        gtruth =target.contiguous().view(-1)
        
        loss = self.criterion(scores, gtruth, extra_len)

        
        selected_ind = gtruth.ne(self.padding_idx)
        scores2_temp = scores2[selected_ind]
        scores_temp = scores[selected_ind]
        loss_cross1 = self.criterion_kl_div(scores2_temp, scores_temp.exp())
        loss_cross2 = self.criterion_kl_div(scores_temp, scores2_temp.exp())
        loss_cross = 0.5*loss_cross2 + 0.5*loss_cross1
        #loss_cross = self.criterion(scores2, gtruth2, extra_len)
        
        stats = self._stats(loss1.clone(), loss2.clone(),loss.clone(), scores1,gtruth1,scores2,gtruth2,scores,gtruth)

        return loss1,loss2,loss,loss_cross, stats


def filter_shard_state(state, shard_size=None):
    """ ? """
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.

        ###将output和target按照size=32进行划分,

        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []

        for k, (v, v_split) in non_none.items():
            #if isinstance(v, torch.Tensor) and state[k].requires_grad and k!='copy_attn' and k!='copy_or_generate':
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)

        torch.autograd.backward(inputs, grads, retain_graph=True)
