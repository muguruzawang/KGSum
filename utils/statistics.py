import math
import sys
import time


from utils.distributed import all_gather_list
from utils.logging import logger


class Statistics(object):
    """
    Accumulator for loss statistics.
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss_temp=0,loss_ent=0, loss=0, n_words_temp=0, n_correct_temp=0,n_words_ent=0, \
                      n_correct_ent=0,n_words=0, n_correct=0):
        self.loss = loss
        self.loss_temp = loss_temp
        self.loss_ent = loss_ent
        self.n_words_temp = n_words_temp
        self.n_correct_temp = n_correct_temp
        self.n_words_ent = n_words_ent
        self.n_correct_ent = n_correct_ent
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            `Statistics`, the update stats object
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        from torch.distributed import get_rank

        """
        Gather a `Statistics` list accross all processes/nodes

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        # Get a list of world_size lists with len(stat_list) Statistics objects
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct
        self.loss_temp += stat.loss_temp
        self.n_words_temp += stat.n_words_temp
        self.n_correct_temp += stat.n_correct_temp
        self.loss_ent += stat.loss_ent
        self.n_words_ent += stat.n_words_ent
        self.n_correct_ent += stat.n_correct_ent

        if update_n_src_words:
            self.n_src_words += stat.n_src_words

    def accuracy(self):
        """ compute accuracy """
        return 100 * (self.n_correct / self.n_words)

    def accuracy_temp(self):
        """ compute accuracy """
        return 100 * (self.n_correct_temp / self.n_words_temp)

    def accuracy_ent(self):
        """ compute accuracy """
        return 100 * (self.n_correct_ent / self.n_words_ent)

    def xent(self):
        """ compute cross entropy """
        return self.loss_ent / self.n_words_ent

    def ppl(self):
        """ compute perplexity """
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        """ compute elapsed time """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate1, start):
        """Write out statistics to stdout.

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        logger.info(
            ("Step %2d/%5d; #loss: %6.3f; #loss_temp: %6.3f; #loss_ent: %6.3f; *acc: %6.2f; *acc_temp: %6.2f; *acc_ent: %6.2f;" +
                "  ppl: %5.2f; xent: %4.2f;" + "learning_rate: %7.5f; %3.0f/%3.0f tok/s; %6.0f sec")
            % (step, num_steps,
               self.loss,
               self.loss_temp,
               self.loss_ent,
               self.accuracy(),
               self.accuracy_temp(),
               self.accuracy_ent(),
               self.ppl(),
               self.xent(),
               learning_rate1,
               self.n_src_words / (t + 1e-5),
               self.n_words / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/loss", self.loss, step)
        writer.add_scalar(prefix + "/ppl", self.ppl(), step)
        writer.add_scalar(prefix + "/accuracy", self.accuracy(), step)
        writer.add_scalar(prefix + "/tgtper", self.n_words / t, step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
