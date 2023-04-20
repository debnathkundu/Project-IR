import math
from flask import Flask, render_template, request,session
import pandas as pd
import pickle
from sklearn.metrics import pairwise_distances
import numpy as np
import random
import numpy as np
import pandas as pd
import string
import uuid  
import math
import random
from datetime import datetime
import csv


#for storing the model
import pickle

# Below libraries are for text processing using NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Below libraries are for feature representation using sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Below libraries are for similarity matrices using sklearn
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances

from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nltk.download('punkt')
nltk.download("wordnet")

# fuction for text preprocessing
def clean_text(text):
      le=WordNetLemmatizer()
      word_tokens=word_tokenize(text)
      tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
      tokens_cleaned = [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens]
      cleaned_text=" ".join(tokens_cleaned)
      return cleaned_text

# Function for recommendation

# To LOAD THE SAVED MODEL
filename = 'tf-idf_vectorizer features_inshorts.sav'
tfidf_vectorizer_features = pickle.load(open(filename,'rb'))

# Function to calculate cosine similarity of one vs all 
def tfidf_based_model(row_index, num_similar_items):
    couple_dist = cosine_similarity(tfidf_vectorizer_features,tfidf_vectorizer_features[row_index])
    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
    
    return indices
# Function to diversify the recommended list

def diversify(recommended_list, thresh):

  # initializing the diversified list
  diversified_list = []
  diversified_list.append(recommended_list[0])

  # initialising candidate_list 
  candidate_list = recommended_list

  # loop from 2 to n in candidate_list
  for i in range(1,len(candidate_list)):
    candidate_item = candidate_list[i]

    # set of items in candidate_list that didnot occur in diversifeid list so far(i.e in diversified_list[:i])
    candidate_temp = [article for article in candidate_list if article not in diversified_list[:i]]

    diversity_values = []
    #compute diversity metric of each ele in candidate temp wrt all items in diversified list
    for ele in candidate_temp:
      t = (1-cosine_similarity_ova(ele,diversified_list),ele)
      diversity_values.append(t)

    #sort the diversity score in reverse
    diversity_values.sort(reverse=True)

    list1={}
    for i in range(len(diversity_values)):
      list1[diversity_values[i][1]] = i

    # calculate weights 
    weights = []
    for i in range(len(candidate_temp)):
      pos_og = recommended_list.index(candidate_temp[i])
      pos_div = list1[candidate_temp[i]]
      w = ( (pos_og*(1-thresh)) + (pos_div)*thresh, candidate_temp[i] )
      weights.append(w)

    weights.sort()

    if (len(weights) != 0):
      diversified_list.append(weights[0][1])

  return diversified_list

def cosine_similarity_ova (one, all):
  sum = 0
  n = len(all)
  for item in all:
    sum += cosine_similarity(tfidf_vectorizer_features[one], tfidf_vectorizer_features[item])[0][0]

  cs = sum/n
  return cs


# Function to calculate ILD

def ILD(li):
  sum = 0
  n = len(li)

  for i in range(len(li)):
    for j in range(len(li)):
      sum += cosine_similarity(tfidf_vectorizer_features[li[i]],tfidf_vectorizer_features[li[j]])[0][0]
  ILD = sum/(n*(n-1))

  return (1-ILD)

# Defining function for final recommendations

def recommend(is_diverse, article_id):

  if(is_diverse not in [0,1]):
    raise Exception("Wrong value of is_diverse! Enter 0 or 1 as is_dierse")

  if(article_id <0 or article_id>815):
    raise Exception("Wrong value of article_id! Enter value between 0 to815")

  top_50 =  list(tfidf_based_model(article_id, 50))

  if(is_diverse == 0):
    
    recomm_list = top_50[:10]
    ild = ILD(recomm_list)
    return recomm_list, ild

  elif(is_diverse ==1):
    recomm_list = diversify(top_50,0.85)

    # check
    if (len(recomm_list) !=10):
      for i in range(10-len(recomm_list)):
        recomm_list.sppend(top_50[-i])
    ild = ILD(recomm_list[:10])
    return recomm_list, ild


file = open('Copy of dataset_preprocessed.pkl', 'rb')
data = pickle.load(file)
file.close()

data_pd=pd.DataFrame(data)
data_pd.set_index("index", inplace = True)


def get_recommendations_data(id=None):
    if id:
        r=id    
        recommended_list,ild=recommend(1,r)
    else:
        r = random.randint(1, 814)
        recommended_list,ild=recommend(0,r)
    
    out_data=[]
    for d in recommended_list:
        result = data_pd.iloc[int(d)]
        out_data.append([result['headlines'].split(',')[0],result['link'],result['short_description'][:200]+'...',d])

    return out_data,ild


app = Flask(__name__)

@app.route('/')
def index():
    session['user_id']=uuid.uuid1()
    out_data,ild=get_recommendations_data()
    threshold=0.85
    return render_template('result.html', out_data = out_data,ild=ild,threshold=threshold) 


@app.route('/recommend/<article_id>')
def recommend_page(article_id):
    user_id = session['user_id']
    now = datetime.now()
    current_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    
    file=open('data.csv','a')
    writer=csv.writer(file)
    category_id=data_pd.iloc[int(article_id)]['category']
    writer.writerow([user_id,current_time,article_id,category_id])
    file.close()
    
    out_data,ild=get_recommendations_data(int(article_id))
    threshold=0.85
    return render_template('result.html', out_data = out_data,ild=ild,threshold=threshold) 

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.run(debug=True)
