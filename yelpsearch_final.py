import requests
import json
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import nltk
from textblob import TextBlob
from collections import Counter 
from sklearn import linear_model
from flask import Flask, redirect, render_template, request, session, url_for

## PROCESS DATA
resturant_name = []
resturant_id = []
ratings = []
idandrating = []
api_key= "RZMGNepw-0uL71Ur4gG4bCh6gCVFCy5SXZrzV7vJBTYnTu7z6JFf_ZgIQ0DMAl-r9WdOfKzT-vkgm2LCKFZQu-dyY1MzmQnQQARbmqHqJm7irB4LupeKX04lH6W9XXYx"
headers = {'Authorization': 'Bearer %s' % api_key}
area = "hoboken"  # limit the area to vicinity
url = 'https://api.yelp.com/v3/businesses/search'
params = {'term':'dinner','location':area,'limit':50}
req = requests.get(url, params=params, headers=headers)
parsed = json.loads(req.text)
businesses = parsed["businesses"]
for business in businesses:
    resturant_name.append(business["name"])
    resturant_id = business["id"]
    ratings = business["rating"] 
    results = requests.get("https://api.yelp.com/v3/businesses/" + str(resturant_id) + "/reviews",  headers = headers)
    parsed = json.loads(results.text)
    reviews = parsed["reviews"]
    url = reviews[0]["url"]
    page_review = requests.get(url)
    page_soup = BeautifulSoup(page_review.content, features = 'lxml')
    page_return = page_soup.find_all("span", attrs = {"class":"lemon--span__373c0__3997G","lang" :"en"})
    allreview = []
    for perreview in page_return:
        allreview.append(perreview.text)
        blobreview = TextBlob(str(allreview).replace(r'\xa0', ' '))  
    score = blobreview.sentiment.polarity
    idandrating.append([resturant_id, ratings, score, allreview])  

database = dict(zip(resturant_name, idandrating)) 

###my working part & service 7
## PROCESS DATA for SKlearn
rating_score = [] 
sentiment_score = [] 
for i in database.values(): 
    rating_score.append(i[1]) 
    sentiment_score.append(((i[2]+1)/2)*5) 
rating_blob = np.asarray(rating_score).reshape(-1,1) 
regr = linear_model.LinearRegression() 
regr.fit(rating_blob.reshape(-1,1), sentiment_score)
sentment_pred = regr.predict(rating_blob)

index = 0 
for i in database.keys():
    database[i].append(sentment_pred[index]) 
    index += 1
###done

## FLASK
app = Flask(__name__, template_folder='templates')

@app.route('/yelpsearch', methods=['GET', 'POST'])
def yelpsearch():
    error = None
    if request.method == 'POST':
        # match input restaurant name with database 
        name = request.form['name']
        if name in database.keys():

            # Service 1: Translate
            blob = TextBlob(str(database[name][3]))
            chinese_blob = blob.translate(from_lang='en', to='zh-CN')

            # Service 2: Sentiment Score
            score_blob = round(database[name][2],2)

            # Service 3: Adjective Words
            reviewpos = blob.pos_tags
            adj=[]
            for item in reviewpos:
                if item[1] in ["JJ", "JJR", "JJS"]:
                    adj.append(item[0])
            sorted_adj = Counter(adj).most_common(10)

            # Service 4: Noun Words
            noun = []
            for item in reviewpos:  
                if item[1] == "NN":
                    noun.append(item[0])
            sorted_noun = Counter(noun).most_common(10)

            # Service 5-6
            np = blob.noun_phrases
            f_cs = []
            for i in np:
                i = ''.join(i)
                blob = TextBlob(i)
                f_cs.append([blob.sentiment.polarity, i])
            f_cs = pd.DataFrame(f_cs)
            f_cs = f_cs.sort_values(by=0, ascending=False)
            __re_data = dict()
            for (i, b) in zip(f_cs[0], f_cs[1]):
                __re_data[b] = i
            key = []
            key1 = []
            key2 = []

            for i in __re_data:
                key.append(i)
            
            # Service 5: Top 10 Positive Noun Phrases
            for i in range(len(key)):
                if i > 9:
                    break
                key1.append(key[i])
            
            # Service 6: Top 10 Negative Noun Phrases
            for i in range(len(key)):
                if i > 9:
                    break
                key2.append(key[len(key)-i-1])
           
            # Service 7: SKlearn
            predict_blob = round(database[name][4],2)


            return render_template('success.html',
                                   translate=chinese_blob, 
                                   score=score_blob,
                                   adj=sorted_adj,
                                   noun=sorted_noun,
                                   positive=key1,
                                   negative=key2,
                                   rate=predict_blob)
    

        else:
            error = "Your search of Restaurant Name is not valid."

    return render_template('home.html', error=error)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
