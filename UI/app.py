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

def tfidf_based_model(row_index, num_similar_items,cs_given):
    couple_dist = cs_given[row_index]
    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]
    return indices

def diversify(recommended_list, thresh,cs):

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
      t = (1-cosine_similarity_ova(ele,diversified_list,cs),ele)
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
      # pos_div = diversity_values.index(candidate_list[i])
      w = ( (pos_og*(1-thresh)) + (pos_div)*thresh, candidate_temp[i] )
      weights.append(w)

    weights.sort()

    if (len(weights) != 0):
      diversified_list.append(weights[0][1])

  return diversified_list

# Function to calculate cosine similarity of one vs all 
def cosine_similarity_ova (one, all,cs):
  sum = 0
  n = len(all)
  cs_list = cs[one][all]
  for item in cs_list:
    sum += item
  c = sum/n
  return c

# Function to calculate ILD

def ILD(li,cs):
  sum = 0
  n = len(li)

  for i in range(len(li)):
    for j in range(i):
      if(i!=j):
        sum += cs[li[i]][li[j]]

  ILD = sum/(n*(n-1))

  return (1-ILD)

# Defining function for final recommendations

def helper(is_diverse, article_id,cs,cs_given):

  if(is_diverse not in [0,1]):
    raise Exception("Wrong value of is_diverse! Enter 0 or 1 as is_dierse")

  if(article_id <0 or article_id>815):
    raise Exception("Wrong value of article_id! Enter value between 0 to815")

  top_200 =  list(tfidf_based_model(article_id, 200,cs_given))

  if(is_diverse == 0):
    
    recomm_list = top_200[:10]
    ild = ILD(recomm_list,cs)
    return recomm_list, ild

  elif(is_diverse ==1):
    recomm_list = diversify(top_200,0.9,cs)

    # check
    if (len(recomm_list) !=10):
      for i in range(10-len(recomm_list)):
        recomm_list.sppend(top_200[-i])
    ild = ILD(recomm_list[:10],cs)
    return recomm_list, ild

# Defining function for final recommendations

def recommend(is_diverse, article_id):


  col_extracted = ['TextBlob_Subjectivity', 'TextBlob_Polarity','TextBlob_Analysis','topic'] 
  col_given = news_articles_with_features.columns[:520].tolist()
  col_given.append('category')
  cs_given = cosine_similarity(news_articles_with_features[col_given])
  cs = cosine_similarity(news_articles_with_features[col_extracted])

  recomm_list, ild = helper(is_diverse, article_id, cs, cs_given)
  return recomm_list, ild
  
news_articles_with_features = pd.read_pickle('dataset_features.pkl')
file = open('dataset_preprocessed.pkl', 'rb')

# file=news_articles_with_features
data = pickle.load(file)
file.close()

data_pd=pd.DataFrame(data)
data_pd.set_index("index", inplace = True)


def get_recommendations_data(id=None):
    if id:
        r=id    
        recommended_list,ild=recommend(1,int(r))
    else:
        r = random.randint(1, 814)
        recommended_list,ild=recommend(0,int(r))
    
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
    threshold=0.55
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
    threshold=0.70
    return render_template('result.html', out_data = out_data,ild=ild,threshold=threshold) 

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.run(debug=True)
