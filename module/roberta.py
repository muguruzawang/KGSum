import torch
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel
from utils.logging import logger


class RobertaEmbedding(object):
    def __init__(self, device, model='/data/home/scv0028/run/wpc/huggingface/roberta-base'):
        super(RobertaEmbedding, self).__init__()
        self._model = RobertaModel.from_pretrained(model)
        self._tokenizer = RobertaTokenizer.from_pretrained(model)
        #self._model = BertModel.from_pretrained(model)
        #self._tokenizer = BertTokenizer.from_pretrained(model)
        self._model.cuda()
        #print('################the model is:')
        #for name, parameters in self._model.named_parameters():
        #    print(name, ':', parameters.device)
        #    break
        self._model.eval()
        #print('Roberta initialized')
        print('Bert initialized')
        self._pad_id = self._tokenizer.pad_token_id
        self._cls_token = self._tokenizer.cls_token
        self._sep_token = self._tokenizer.sep_token
        self._embedding = self._model.embeddings.word_embeddings
        self._embedding.weight.requires_grad = False
        self._eos = self._tokenizer.encoder[self._tokenizer.eos_token]

    def __call__(self, input_ids):
        attention_mask = (input_ids != self._pad_id).float()
        return self._model(input_ids, attention_mask=attention_mask)
