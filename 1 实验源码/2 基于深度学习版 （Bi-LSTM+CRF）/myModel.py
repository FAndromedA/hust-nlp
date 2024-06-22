import torch
import torch.nn as nn
from torchcrf import CRF

from transformers import BertModel, BertTokenizer

BERT_PATH = "./myBert"

class MyModel(nn.Module):
    def __init__(self, tag2id):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        self.embedding_dim = 768
        self.hidden_dim = 256
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)
        # 冻结BERT的底层参数，只训练顶部层
        for name, param in self.bert.named_parameters():
                param.requires_grad = False

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
        lstm_out, _ = self.lstm(last_hidden_states)
        feats = self.hidden2tag(lstm_out)
        return feats

    def forward(self, sentence, tags, mask):
        # print(sentence.shape)
        emissions = self._get_bert_feat(sentence, mask)
        loss = -self.crf(emissions, tags, mask, reduction='mean')
        return loss

    def infer(self, sentence, mask):
        emissions = self._get_bert_feat(sentence, mask)
        return self.crf.decode(emissions, mask=mask)