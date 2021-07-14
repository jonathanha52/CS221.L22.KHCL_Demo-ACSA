from joblib import load
from pytorch_pretrained_bert import BertModel
from src.aspect_category_model.bert_capsnet import BertCapsuleNetwork

#Load model da duoc load state dict va ma tran cam xuc
#@Param  path duong dan den file pickle cua model
#@return model da duoc load
def load_model():
    bert = BertModel.from_pretrained('bert-base-uncased')
    model = BertCapsuleNetwork(
        bert = bert,
        bert_size=768,
        capsule_size=300,
        dropout=0.1,
        num_categories=3
    )
    model.load_sentiment("sentiment_matrix.npy")
    return model

def load_vectorizer():
    vectorizer = load("tfidfVectorizer.joblib")
    return vectorizer

def load_aspect_classifier():
    classifier = load("aspect-classification.joblib")
    return classifier