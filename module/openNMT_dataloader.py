import gc
import glob
import random
import torch
import numpy as np

from utils.logging import logger


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class GraphIterator(object):
    def __init__(self, args, dataset, symbols, batch_size, device=None, shuffle=True, is_test=False):
        self.args = args
        self.dataset = dataset
        self.symbols = symbols
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test

        self.iterations = 0
        self._iterations_this_epoch = 0

        self.secondary_sort_key = lambda x: sum([len(xi) for xi in x['src']])
        self.prime_sort_key = lambda x: len(x['tgt'])

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            c_len = 0
            for i in range(len(ex['clusters'])):
                c_len += len(ex['clusters'][i])
            if c_len == 0:
                continue
            minibatch.append(ex)
            size_so_far = self.simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def simple_batch_size_fn(self, new, count):
        global max_src_in_batch, max_tgt_in_batch
        if count == 1:
            max_src_in_batch = 0
        if (self.args.hier):
            max_src_in_batch = max(max_src_in_batch, sum([len(p) for p in new['src']]))
        else:
            max_src_in_batch = max(max_src_in_batch, len(new['src']))
        src_elements = count * max_src_in_batch
        return src_elements

    def get_batch(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        data = self.data()
        for minibatch_buffer in self.batch_buffer(data, self.batch_size * 100):
            if self.args.mode != 'train':
                p_batch = self.get_batch(
                    sorted(sorted(minibatch_buffer, key=self.prime_sort_key), key=self.secondary_sort_key),
                    self.batch_size
                )
            else:
                p_batch = self.get_batch(
                    sorted(sorted(minibatch_buffer, key=self.secondary_sort_key), key=self.prime_sort_key),
                    self.batch_size
                )
            p_batch = list(p_batch)

            if self.shuffle:
                random.shuffle(p_batch)
            for b in p_batch:
                if len(b) == 0:
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = GraphBatch(minibatch, self.args.hier, self.symbols['PAD'], self.device, self.is_test,
                                   self.symbols['BOS'], self.symbols['EOS'])

                yield batch

            return


class GraphBatch(object):
    def __init__(self, data=None, hier=False, pad_id=None, device=None, is_test=False, bos_id=None, eos_id=None):
        if data is not None:
            self.batch_size = len(data)
            self.pad_id = pad_id
            self.bos_id = bos_id
            self.eos_id = eos_id
            src = [ex['src'] for ex in data]
            tgt = [ex['tgt'] for ex in data]
            clusters = [ex['clusters'] for ex in data]

            self.max_npara = max([len(ex) for ex in src])  # generally, it's 20
            self.max_ntoken = max([max([len(p) for p in e]) for e in src])  # generally, it's 100
            self.max_ncluster = max([len(cluster) for cluster in clusters])
            self.max_cluster_ntoken = 0
            c_len = []
            for cs in clusters:
                if cs is not None:
                    cs_len = [len(cluster) for cluster in cs]
                else:
                    cs_len = 0
                c_len.extend(cs_len)
            if len(c_len) != 0:
                self.max_cluster_ntoken = max(c_len)
            else:
                self.max_cluster_ntoken = 0
            # self.max_cluster_ntoken = max([max([len(c) for c in cs]) if cs is not None else 0 for cs in clusters])
            if self.max_cluster_ntoken > 50:
                self.max_cluster_ntoken = 50
            if self.max_ntoken > 100:
                self.max_ntoken = 100
            if self.max_npara > 20:
                self.max_npara =20
            _src = [self._pad_para(ex[:self.max_npara], self.max_npara, self.max_ntoken, self.pad_id, self.bos_id, self.eos_id) for ex in src]
            _cluster = [self._pad_para(ex[:self.max_ncluster], self.max_ncluster, self.max_cluster_ntoken, self.pad_id, self.bos_id, self.eos_id) for ex in clusters]
            src = torch.stack([torch.tensor(e[0]) for e in _src])  # batch_size * max_npara * max_ntoken
            cluster = torch.stack([torch.tensor(e[0]) for e in _cluster])

            batch_E = []
            for ex in data:
                Eij = []
                c2p = ex['tfidf']
                for i in range(self.max_npara):
                    Ei = []
                    if not c2p.__contains__(str(i)):
                        c2p[str(i)] = {}
                    for j in range(self.max_ncluster):
                        if not c2p[str(i)].__contains__(str(j)):
                            c2p[str(i)][str(j)] = 0.0
                        Ei.append(c2p[str(i)][str(j)])
                    Eij.append(Ei)
                batch_E.append(Eij)
            batch_E = torch.FloatTensor(batch_E).to(device)

            setattr(self, 'edge', batch_E)

            # graphs = []
            # for ex in data:
            #     G = self.createGraph(ex)
            #     graphs.append(G)
            #
            # batched_graph = dgl.batch(graphs)
            # setattr(self, 'graph', batched_graph.to(device))

            setattr(self, 'src', src.to(device))
            setattr(self, 'cluster', cluster.to(device))

            _tgt = self._pad_para(tgt, width=max([len(d) for d in tgt]), height=len(tgt), pad_id=pad_id, bos_id=self.bos_id, eos_id=self.eos_id)
            tgt = torch.tensor(_tgt[0]).transpose(0, 1)
            setattr(self, 'tgt', tgt.to(device))

            if (is_test):
                tgt_str = [ex['tgt_str'] for ex in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size

    def _pad_cluster(self, clusters, pad_id, _max_cluster_token=50):
        """
        :param clusters:  [ [] ], src clusters in one example
        :return:
        """
        # there exits instance where clusters = []
        clusters_pad = []
        if len(clusters) == 0:
            clusters.append([])  # transform [] to [[]]

        if _max_cluster_token > self.max_cluster_ntoken:
            _max_cluster_token = self.max_cluster_ntoken

        trunc_cluster = [cluster[: _max_cluster_token] for cluster in clusters]
        # max_cluster_token_ex = max([len(cluster) for cluster in trunc_cluster])
        for i in range(len(clusters)):
            _cluster = clusters[i].copy()
            if len(_cluster) > _max_cluster_token:
                _cluster = _cluster[:_max_cluster_token]
            if len(_cluster) < _max_cluster_token:
                _cluster.extend([pad_id] * (_max_cluster_token - len(_cluster)))
            clusters_pad.append(_cluster)
        return clusters_pad

    def _pad_para(self, data, height, width, pad_id, bos_id, eos_id):
        """
        :param data:  [ [] ], src paras in one example
        :param height: num of paras in one example, generally, it's 20
        :param width:  num of max_tokens
        :param pad_id:
        :return:
        """
        # rtn_data = [para + [pad_id] * (width - len(para)) for para in data]
        rtn_data = []
        for para in data:
            if len(para) > width:
                para = para[:width]
            else:
                para += [pad_id] * (width - len(para))
            rtn_data.append(para)
        rtn_length = [len(para) for para in data]
        x = []
        x.append(bos_id)
        x.append(eos_id)
        x.extend([pad_id] * (width-2))
        rtn_data = rtn_data + [x] * (height - len(data))
        # rtn_data = rtn_data + [[pad_id] * width] * (height - len(data))
        rtn_length = rtn_length + [0] * (height - len(data))
        if len(rtn_data) == 0:
            rtn_data.append([])
        return rtn_data, rtn_length