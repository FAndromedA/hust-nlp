import torch
import torch.nn as nn
from torchcrf import CRF

from transformers import BertModel, BertTokenizer

BERT_PATH = ".\\bert"

class MyModel(nn.Module):
    def __init__(self, tag2id):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        self.embedding_dim = 768
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)
        self.hidden2tag = nn.Linear(self.embedding_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)

    def _get_bert_feat(self, sentence, mask):
        # encoded_input = self.tokenizer(sentence,
        #                           padding=True,
        #                           truncation=True,
        #                           return_tensors='pt')
        # with torch.no_grad():
        #     outputs = self.bert(input_ids=encoded_input['input_ids'],
        #                         attention_mask=encoded_input['attention_mask'])
        outputs = self.bert(sentence, attention_mask=mask)
        last_hidden_states = outputs.last_hidden_state
        bert_feat = self.hidden2tag(last_hidden_states)
        return bert_feat

    def forward(self, sentence, tags, mask):
        # print(sentence.shape)
        emissions = self._get_bert_feat(sentence, mask)
        loss = -self.crf(emissions, tags, mask, reduction='mean')
        return loss

    def infer(self, sentence, mask):
        emissions = self._get_bert_feat(sentence, mask)
        return self.crf.decode(emissions, mask=mask)