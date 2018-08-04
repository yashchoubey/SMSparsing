 # -*- coding: utf-8 -*-
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
import numpy as np
import datetime, csv, math
import MySQLdb
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from gensim.models import KeyedVectors
filename = 'dataset/glove/glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)


with open('dataset/stopwords.txt') as f:
    content = f.readlines()
stop_words = [x.strip() for x in content]
print len(stop_words)

db = MySQLdb.connect(host="localhost",user="root",passwd="anant",db="vdo_db")
cur = db.cursor()

video_name = []
cur.execute("SELECT video_name FROM discovery_video_details;")
res = cur.fetchall()
for row in res:
    video_name.append(row[0])

dset = []
cur.execute("SELECT sentencetext FROM discovery_sentence;")
res = cur.fetchall()
for row in res:
    dset.append(row[0])

sentence = []
for x in dset:
    words = x.split()
    tmp = [word for word in words if word.lower() not in stop_words]
    res = ' '.join(tmp)
    sentence.append(res)

print len(sentence)

def train_tfidf(dset, wnt):
    doc = []
    for x in dset:
        try:
            x = x.decode('utf-8')
            x.encode('ascii', 'ignore')
            tokens = word_tokenize(x.lower())
            stem = [wnt.lemmatize(t) for t in tokens]
            doc.append(' '.join(stem))
        except:
            pass
    vectorizer = TfidfVectorizer(ngram_range=(1,3))
    vectorizer.fit(doc)
    print len(vectorizer.vocabulary_)
    return vectorizer

vectorizer = train_tfidf(sentence, WordNetLemmatizer())

def avg_feature_vector(sentence, model, num_features=1):
    sentence = sentence.lower()
    words = sentence.split()
    feature_vec = np.zeros((1, num_features), dtype='float32')
    n_words = 0
    for word in words:
        if word in model:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if(n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def get_video_from_glovesimilarity(inp, sent, model, stop_words):
    vec2 = avg_feature_vector(sent, model)
    cont = []
    for x in range(0, len(inp)):
        token = word_tokenize(inp[x])
        words = [w.lower() for w in token if w.lower() not in stop_words]
        vec = avg_feature_vector(' '.join(words), model)
        tmp = cosine_similarity(vec, vec2)
        s1 = inp[x]
        cont.append((tmp[0][0], s1))
    cont = sorted(cont, reverse=True)
    if len(cont)>0:
        return cont[0]

def get_video_from_similarity(inp, vectorizer, sent, wnt):
    cont = []
    tk = word_tokenize(sent)
    stem = [wnt.lemmatize(t) for t in tk]
    vec2 = vectorizer.transform([' '.join(stem)])
    s1 = "NOT FOUND"
    for x in range(0, len(inp)):
        token = word_tokenize(inp[x])
        words = [w.lower() for w in token if w.lower() not in stop_words]
        stem_token = [wnt.lemmatize(t) for t in words]
        vec = vectorizer.transform([' '.join(stem_token)])
        tmp = cosine_similarity(vec, vec2)
#         print tmp[0][0]
        s1 = inp[x]
        cont.append((tmp[0][0], s1))
    
    cont = sorted(cont, reverse=True)
    return cont[0]

split_words = {
    "a/c":"ac",
    "/":" "
}
def preprocess(text, wnt):
    li = []
    token = word_tokenize(text)
    words = [w.lower() for w in token if w.lower() not in stop_words]
    for word in words:
        tmp = word
        if word in split_words.keys():
            tmp = split_words[word]
        tp = wnt.lemmatize(tmp)
#         print word, tmp, tp
        li.append(tp)
    return ' '.join(li)

def wmd_distance_sim(sent1, inp):
    sim = []
    t1 = [w.lower() for w in sent1 if w.lower() not in stop_words]
    tok1 = [t for t in t1 if t in model]
    cont = []
    for x in inp:
        t2 = [w.lower() for w in x if w.lower() not in stop_words]
        tok2 = [t for t in t2 if t in model]
        dis = model.wmdistance(t1, t2)
        cont.append((dis, x))
    cont = sorted(cont)
#     print cont[0]
    if cont!=None:
        return cont[0]
    return (float('int'), "NA")


similar_video_glove = []
similar_video_tfidf = []
score_glove = []
score_tfidf = []
similar_wmd = []
score_wmd = []
cause_text = []
wnt = WordNetLemmatizer()
# ps = PorterStemmer()
with open('dataset/cause_text.csv') as csvfile:
    lines = csv.reader(csvfile)
    next(lines, None)
    for i, line in enumerate(lines):
    	cause_text.append(line[2])
    	sent = preprocess(line[2], wnt)
    	v1 = get_video_from_glovesimilarity(video_name, sent, model, stop_words)
    	v2 = get_video_from_similarity(video_name, vectorizer, sent, wnt)
    	score_glove.append(v1[0])
    	score_tfidf.append(v2[0])
    	similar_video_glove.append(v1[1])
    	similar_video_tfidf.append(v2[1])
    	v3 = wmd_distance_sim(sent, video_name)
    	score_wmd.append(v3[0])
    	similar_wmd.append(v3[1])


df = pd.DataFrame({
    'CauseText': cause_text, 
    'Glove_similarity_score':score_glove,
    'Glove_Recommendation': similar_video_glove,
    'Tfidf_similarity_score':score_tfidf,
    'Tfidf':similar_video_tfidf,
    'wmd_sim_score':score_wmd,
    'wmd_similar_vdo':similar_wmd
    })

print df.head()
