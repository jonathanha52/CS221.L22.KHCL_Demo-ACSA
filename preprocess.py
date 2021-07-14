import pickle
import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#Tao Bert token va Bert segment lam dau vao cho mo hinh Bert Capsnet
#@Param text   Cau van ban dau vao
#@Param aspect Khia canh cua van ban
#@return 2 tensor la bert token va bert segment
def data_preprocess(text, aspect):
    bert_sentence = bert_tokenizer.tokenize(text)
    bert_aspect = bert_tokenizer.tokenize(aspect)
    bert_token = bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + bert_sentence + ['[SEP]'] + bert_aspect + ['[SEP]'])
    bert_segment = ([0] * (len(bert_sentence) +2)) + ([1] * (len(bert_aspect) + 1))
    bert_token.extend([0] * (76 - len(bert_token)))
    bert_segment.extend([0] * (76 - len(bert_segment)))
    token = torch.tensor(np.asarray([bert_token], dtype=np.int32)).long()
    segment = torch.tensor(np.asarray([bert_segment], dtype=np.int32)).long()
    return token, segment