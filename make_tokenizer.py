from pytorch_pretrained_bert import BertTokenizer
import pickle

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

TOKENIZER_FILE = "BertTokenizer.pickle"

with open(TOKENIZER_FILE, "wb") as f:
    pickle.dump(bert_tokenizer, f)