import torch
import torch.nn as nn
from torchcrf import CRF

from transformers import BertModel, BertTokenizer

BERT_PATH = "./myBert"

class BertNer(nn.Module):
    def __init__(self, tag2id):
        super(BertNer, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        self.embedding_dim = 768
        self.hidden_dim = 256
        self.tag2id = tag2id
        self.tagset_size = len(tag2id)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        
        #self.residual_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        #self.BN = nn.BatchNorm1d(self.hidden_dim)
        #self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.layerNorm = nn.LayerNorm(self.hidden_dim)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        #self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size, batch_first=True)
        # 冻结BERT的底层参数，只训练顶部层
        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11") or name.startswith("pooler") or name.startswith("classifier"):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _get_bert_feat(self, sentence, mask):
        outputs = self.bert(sentence, attention_mask=mask)
        last_hidden_states = outputs.last_hidden_state
        #/beg
        last_hidden_states = self.dropout(last_hidden_states)
        #/end
        lstm_out, _ = self.lstm(last_hidden_states)
        #/beg
        lstm_out = self.layerNorm(lstm_out)
        #/end
        feats = self.hidden2tag(lstm_out)
        return feats
        # outputs = self.bert(sentence, attention_mask=mask.bool())
        # last_hidden_states = outputs.last_hidden_state
        # lstm_out, _ = self.lstm(last_hidden_states)
        
        # residual = self.residual_projection(lstm_out)
        # residual = self.BN(residual.transpose(1, 2)).transpose(1, 2)
        # residual = self.relu(residual)

        # final_out = self.relu(lstm_out + residual)
        # feats = self.hidden2tag(final_out)
        # return feats

    def forward(self, sentence, tags, mask):
        # print(sentence.shape)
        emissions = self._get_bert_feat(sentence, mask)
        loss = -self.crf(emissions, tags, mask, reduction='mean')
        return loss

    def infer(self, sentence, mask):
        emissions = self._get_bert_feat(sentence, mask)
        return self.crf.decode(emissions, mask=mask)