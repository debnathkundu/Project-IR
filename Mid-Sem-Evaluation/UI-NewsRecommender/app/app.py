import math
from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.metrics import pairwise_distances
import numpy as np
import random

news_articles = pd.read_json("News_Category_Dataset_v2.json", lines = True)
df=news_articles

y=news_articles["category"].unique()
inp = {}
cat_ls = []
for i in range(len(y)):
  inp[i] = y[i]
category=inp

filename = 'tf-idf_vectorizer features.sav'
vectorizer_features = pickle.load(open(filename,'rb'))

def tfidf_based_model(row_index, num_similar_items):
    couple_dist = pairwise_distances(vectorizer_features,vectorizer_features[row_index])
    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
    df = pd.DataFrame({'publish_date': news_articles['date'][indices].values,
               'headline':news_articles['headline'][indices].values,'link':news_articles['link'][indices].values,'index':indices,
                'Euclidean similarity with the queried article': couple_dist[indices].ravel()})
    print("="*30,"Queried article details","="*30)
    print('headline : ',news_articles['headline'][indices[0]])
    print("\n","="*25,"Recommended articles : ","="*23)
    
    
    return df.iloc[1:,]
    
filename = 'Count-Vectorizer features.sav'
BoW_vectorizer_features = pickle.load(open(filename,'rb'))

def bag_of_words_based_model(row_index, num_similar_items):
    couple_dist = pairwise_distances(BoW_vectorizer_features,BoW_vectorizer_features[row_index])
    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
    df = pd.DataFrame({'publish_date': news_articles['date'][indices].values,
               'headline':news_articles['headline'][indices].values,'link':news_articles['link'][indices].values,'index':indices,
                'Euclidean similarity with the queried article': couple_dist[indices].ravel()})
    print("="*30,"Queried article details","="*30)
    print('headline : ',news_articles['headline'][indices[0]])
    print("\n","="*25,"Recommended articles : ","="*23)
    #return df.iloc[1:,1]
    return df.iloc[1:,]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', category = category) 

@app.route('/about',methods=['POST'])
def getvalue():
    global df
    cat_input=request.form.getlist('category')[0]
    cat_ls = df.index[df['category'] == inp[int(cat_input)]][:25]
    cat_index = random.choice(cat_ls)
    
    rec=bag_of_words_based_model(cat_index,11)
    print(rec)
    
    return render_template('result.html', link=rec['link'], titles =  rec['headline'],scores=rec['Euclidean similarity with the queried article'],date=rec['publish_date'],index=rec['index'])


@app.route('/recommend/<related>',methods=['GET'])
def about(related):
    print(related)
    if related:
        rec=tfidf_based_model(related,11)
        
    else:
        global df
        cat_input=request.form.getlist('category')[0]
        cat_ls = df.index[df['category'] == inp[int(cat_input)]][:25]
        cat_index = random.choice(cat_ls)
    
        rec=bag_of_words_based_model(cat_index,11)
    print(rec)
    
    return render_template('result.html', link=rec['link'], titles =  rec['headline'],scores=rec['Euclidean similarity with the queried article'],date=rec['publish_date'],index=rec['index'])

if __name__ == '__main__':
    app.run(debug=True)
