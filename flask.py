from flask import Flask,render_template,url_for,request
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, make_scorer
from time import time

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('untitled3.html')

@app.route('/predict',methods=['POST'])
def predict():
    df_scraped = pd.read_csv("labeled_tweets.csv")
    df_public = pd.read_csv("public_data_labeled.csv")
    df_scraped.drop_duplicates(inplace = True)
    df_scraped.drop('id', axis = 'columns', inplace = True)

    df_public.drop_duplicates(inplace = True)
    
    
    df = pd.concat([df_scraped, df_public])
    df.shape
    df['label'] = df.label.map({'Offensive': 1, 'Non-offensive': 0})
    global X, y, train_test_split
    X = df['full_text']
    y = df['label']
    from sklearn.model_selection import train_test_split
    count_vector = CountVectorizer(stop_words = 'english', lowercase = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

                                                        
# Fit the training data and then return the matrix
    training_data= count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
    testing_data = count_vector.transform(X_test)
    clf = MultinomialNB()
    clf_sgd = SGDClassifier()
    clf_sgd.fit(training_data,y_train)
    clf.fit(training_data, y_train)
	
    
    clf_sgd.score(testing_data,y_test)
    clf.score(testing_data,y_test)
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)

	
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = count_vector.transform(data).toarray()
        my_prediction = clf_sgd.predict(vect)
    return render_template('untitled5.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True,use_reloader=False)
