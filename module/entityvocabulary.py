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

from tools.logger import *

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

# Note: none of [PAD], [UNK], [START], [STOP] should appear in the vocab file.


class EntityVocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, vocab_file, gloveembed):
        """
        Creates a vocab of up to max_size words, reading from the vocab_file. If max_size is 0, reads the entire vocab file.
        :param vocab_file: string; path to the vocab file, which is assumed to contain "<word> <frequency>" on each line, sorted with most frequent word first. This code doesn't actually use the frequencies, though.
        :param max_size: int; The maximum size of the resulting Vocabulary.
        """
        self._entity_to_id = {}
        self._id_to_entity = {}
        self._count = 0 # keeps track of total number of words in the Vocab

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r', encoding='utf8') as vocab_f: #New : add the utf8 encoding to prevent error
            cnt = 0
            for line in vocab_f:
                cnt += 1
                pieces = line.split("\t")
                # pieces = line.split()
                w = pieces[0]
                wordlist = w.split()
                for word in wordlist:
                    ### 如果单词在wordvocab里出现了，那么就可以拿来计算实体embedding，否则就放弃
                    if word in gloveembed:
                        self._entity_to_id[w] = self._count
                        self._id_to_entity[self._count] = w
                        self._count += 1
                        ###如果在glove的词向量中能够找到word，就break
                        break
                    
        logger.info("[INFO] Finished constructing vocabulary of %i total entities. Last entity added: %s", self._count, self._id_to_entity[self._count-1])

    def entity2id(self, entity):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        return self._entity_to_id[entity]

    def id2entity(self, entity_id):
        """Returns the word (string) corresponding to an id (integer)."""
        return self._id_to_entity[entity_id]

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def entity_list(self):
        """Return the word list of the vocabulary"""
        return self._entity_to_id.keys()