import random
import requests
import json
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import numpy as np
from sklearn import linear_model
from sklearn.cluster import KMeans
from textblob import TextBlob
from collections import Counter
from flask import Flask, redirect, render_template, request, session, url_for

### PROCESS DATA
resturant_name = []
resturant_id = []
ratings = []
idandrating = []
api_key = "ylajG-XgBZYA2vqrwjnkY2rOYLYVO_rmhOZK19nZvkmKaKPd_Et4qUXR3ryaWasIKeukY_Hvp2V696f8m3KqHsbXm1jbkwJY9cYp8tgVA1eA93luR2Um_hXEw6O9XXYx"
headers = {'Authorization': 'Bearer %s' % api_key}
area = "hoboken"  # limit the area to vicinity
url = 'https://api.yelp.com/v3/businesses/search'
params = {'term': 'dinner', 'location': area, 'limit': 50}
req = requests.get(url, params=params, headers=headers)
parsed = json.loads(req.text)
businesses = parsed["businesses"]
__adj = []
for business in businesses:
    resturant_name.append(business["name"])
    resturant_id = business["id"]
    ratings = business["rating"]
    idandrating.append([resturant_id, ratings])
database = dict(zip(resturant_name, idandrating))

## FLASK
app = Flask(__name__, template_folder='templates')


@app.route('/yelpsearch', methods=['GET', 'POST'])
def yelpsearch():
    error = None
    global __adj
    if request.method == 'POST':
        all_services = {'service': ['adj', 'noun', 'translate', 'rate', 'positive', 'negative']}
        if request.form['service'] not in all_services['service']:
            error = 'Invalid Service. Please try again.'
        else:  # match input restaurant name with database
            name = request.form['name']
            if name in database.keys():
                url = "https://api.yelp.com/v3/businesses/" + str(database[name][0]) + "/reviews"
                results = requests.get(url, headers=headers)
                parsed = json.loads(results.text)
                reviews = parsed["reviews"]
                url = reviews[0]["url"]
                page_review = requests.get(url)
                page_soup = BeautifulSoup(page_review.content, features='lxml')
                page_return = page_soup.find_all("span", attrs={"class": "lemon--span__373c0__3997G", "lang": "en"})
                allreview = []
                for perreview in page_return:
                    allreview.append(perreview.text)
                allreview = str(allreview)
                allreview = allreview.replace(r'\xa0', ' ')
                # now we get all reviews of the restaurant you entered

                ## SERVICE 1: translate all reviews into Chinese
                blob = TextBlob(allreview)
                if request.form['service'] == 'translate':
                    chinese_blob = blob.translate(from_lang='en', to='zh-CN')
                    return render_template('success.html', name=chinese_blob)

                ## SERVICE 2: get the overall rating based on the sentiment score of all reviews
                if request.form['service'] == 'rate':
                    score_blob = database[name][1]
                    return render_template('success.html', name=score_blob)

                ## SERVICE 3: top 10 most frequent words (adj only)
                reviewpos = blob.pos_tags
                if request.form['service'] == 'adj':
                    adj = []
                    for item in reviewpos:
                        if item[1] in ["JJ", "JJR", "JJS"]:
                            adj.append(item[0])
                    sorted_adj = Counter(adj).most_common(10)
                    return render_template('success.html', name=sorted_adj)

                ## SERVICE 4: top 10 most frequent words (noun only)
                if request.form['service'] == 'noun':
                    noun = []
                    for item in reviewpos:
                        if item[1] == "NN":
                            noun.append(item[0])
                    sorted_noun = Counter(noun).most_common(10)
                    return render_template('success.html', name=sorted_noun)

                ## SERVICE 5-6
                if request.form['service'] == 'positive' or request.form['service'] == 'negative':
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
                    key2 = []

                    ## SERVICE 5: top 10 most common positive words
                    if request.form['service'] == 'positive':
                        for i in __re_data:
                            key.append(i)
                        for i in range(len(key)):
                            if i > 9:
                                break
                            key2.append(key[i])
                        return render_template('success.html', name=key2)

                    ## SERVICE 6: top 10 most common negative words
                    if request.form['service'] == 'negative':
                        for i in __re_data:
                            key.append(i)
                        for i in range(len(key)):
                            if i > 9:
                                break
                            key2.append(key[len(key) - i - 1])
                        return render_template('success.html', name=key2)


            else:
                error = "Your search of Restaurant Name is not valid."

    return render_template('home.html', error=error)


def graphRegression():
    _rate = []
    global __adj
    rate = []
    _rate = database.values()
    for i in _rate:
        rate.append(i[1])
    parsed = []
    price = []
    for i in _rate:
        url = "https://api.yelp.com/v3/businesses/" + str(i[0])
        results = requests.get(url, headers=headers)
        parsed.append(json.loads(results.text))
    for i in parsed:
        try:
            tmp = (str(i["price"]))
        except KeyError:
            price.append(2.5)
        else:
            if tmp == "$":
                price.append(1)
            if tmp == "$$":
                price.append(2)
            if tmp == "$$$":
                price.append(3)
            if tmp == "$$$$":
                price.append(4)
    print(rate)
    print(price)
    new_rate = np.array(rate)
    new_price = np.array(price)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(new_rate.reshape(-1, 1), new_price.reshape(-1, 1))

    plt.figure(1)
    color = ("red", "green")
    plt.xlim(3, 5)
    plt.ylim(1, 4)
    plt.scatter(new_rate, new_price, c=kmeans.labels_, cmap='rainbow')
    plt.figure(2)
    regr = linear_model.LinearRegression()
    plt.xlim(3, 5)
    plt.ylim(1, 4)
    regr.fit(new_rate.reshape(-1, 1), price)

    plt.scatter(new_rate.reshape(-1, 1), price, color='blue')
    plt.plot(new_rate.reshape(-1, 1), regr.predict(new_rate.reshape(-1, 1)), color='red', linewidth=4)
    plt.show()


graphRegression()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

