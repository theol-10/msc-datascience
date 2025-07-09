import pandas as pd
import numpy as np
import re
import os
import spacy
import Levenshtein

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load SpaCy and cache (cache used later to fasten things a bit)
nlp = spacy.load("en_core_web_md")
_spacy_cache = {}


# Text Preprocessing

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Feature Functions

def jaccard_similarity(q1, q2):
    w1 = set(q1.split())
    w2 = set(q2.split())
    return len(w1 & w2) / len(w1 | w2) if w1 and w2 else 0

def tfidf_cosine_similarity(q1, q2, vectorizer):
    tfidf_matrix = vectorizer.transform([q1, q2])
    sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return sim[0][0]

def levenshtein_ratio(q1, q2):
    return Levenshtein.ratio(q1, q2)

def shared_bigrams_ratio(q1, q2):
    def get_bigrams(text):
        tokens = text.split()
        return set(zip(tokens, tokens[1:])) if len(tokens) > 1 else set()
    b1 = get_bigrams(q1)
    b2 = get_bigrams(q2)
    return len(b1 & b2) / len(b1 | b2) if b1 and b2 else 0

def avg_word_length_diff(q1, q2):
    def avg_len(text):
        words = text.split()
        return sum(len(w) for w in words) / len(words) if words else 0
    return abs(avg_len(q1) - avg_len(q2))

def get_spacy_vector(text):
    if text in _spacy_cache:
        return _spacy_cache[text]
    doc = nlp(text)
    vec = doc.vector if doc.has_vector else np.zeros(nlp.vocab.vectors_length)
    _spacy_cache[text] = vec
    return vec

def compute_spacy_cosine(q1, q2):
    v1 = get_spacy_vector(q1).reshape(1, -1)
    v2 = get_spacy_vector(q2).reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]


# Feature Extraction

def extract_basic_features(df):
    df = df.copy()
    df['question1'] = df['question1'].fillna("").apply(clean_text)
    df['question2'] = df['question2'].fillna("").apply(clean_text)
    df['jaccard'] = df.apply(lambda row: jaccard_similarity(row['question1'], row['question2']), axis=1)
    return df[['jaccard']]


def extract_improved_features(df, tfidf_vectorizer=None):
    df = df.copy()
    df['question1'] = df['question1'].fillna("").apply(clean_text)
    df['question2'] = df['question2'].fillna("").apply(clean_text)

    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(pd.concat([df['question1'], df['question2']]))

    df['jaccard'] = df.apply(lambda row: jaccard_similarity(row['question1'], row['question2']), axis=1)
    df['len_diff'] = df.apply(lambda row: abs(len(row['question1'].split()) - len(row['question2'].split())), axis=1)
    df['tfidf_cosine'] = df.apply(lambda row: tfidf_cosine_similarity(row['question1'], row['question2'], tfidf_vectorizer), axis=1)
    df['levenshtein'] = df.apply(lambda row: levenshtein_ratio(row['question1'], row['question2']), axis=1)
    df['shared_bigrams'] = df.apply(lambda row: shared_bigrams_ratio(row['question1'], row['question2']), axis=1)
    df['avg_word_len_diff'] = df.apply(lambda row: avg_word_length_diff(row['question1'], row['question2']), axis=1)
    df['spacy_cosine'] = df.apply(lambda row: compute_spacy_cosine(row['question1'], row['question2']), axis=1)

    feature_cols = ['jaccard', 'len_diff', 'tfidf_cosine', 'levenshtein', 'shared_bigrams', 'avg_word_len_diff', 'spacy_cosine']
    return df[feature_cols], tfidf_vectorizer


# Model & Evaluation

def train_logistic_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y_true):
    y_probs = model.predict_proba(X)[:, 1]
    y_preds = model.predict(X)
    return {
        'roc_auc': roc_auc_score(y_true, y_probs),
        'precision': precision_score(y_true, y_preds),
        'recall': recall_score(y_true, y_preds)
    }


# Data loading/split - code given in the assignment guide

#def load_and_split_data(path="./quora_data.csv"):
#    df = pd.read_csv(path)
#    train_df, val_df = train_test_split(A_df, test_size=0.05, random_state=123)
#    A_df, test_df = train_test_split(df, test_size=0.05, random_state=123)
#    return train_df, val_df, test_df

def load_and_split_data(path=None):
    if path is None:
        path = os.path.expanduser("~/Datasets/QuoraQuestionPairs/quora_data.csv")
    df = pd.read_csv(path)
    A_df, test_df = train_test_split(df, test_size=0.05, random_state=123)
    train_df, val_df = train_test_split(A_df, test_size=0.05, random_state=123)
    return train_df, val_df, test_df

