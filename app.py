from flask import Flask, render_template, redirect, request
from flask import send_file
import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# __name__ == __main__
app = Flask(__name__)


@app.route('/')
def hello():
	return render_template("index.html")


@app.route('/', methods = ['POST'])
def submit():
	new_text=request.form['text']
	sid = SentimentIntensityAnalyzer()
	a = sid.polarity_scores(new_text)
	return render_template("index.html", A = a)

@app.route('/amazon', methods = ['POST'])
def amazon():
	sid = SentimentIntensityAnalyzer()
	df = pd.read_csv('static/amazonreviews.tsv', sep='\t')
	df.head()
	df['label'].value_counts()



	df.dropna(inplace=True)


	empty = []

	for i,lb,rv in df.itertuples():
	    if type(rv)==str:
	        if rv.isspace():
	            empty.append(i)


	df.drop(empty, inplace=True)

	df['label'].value_counts()

	sid.polarity_scores(df.loc[0]['review'])

	df.loc[0]['review']

	df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))
	df.head()

	df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])
	df.head()

	df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c>=0 else 'neg')
	df.head()

	accuracy_score(df['label'], df['comp_score'])

	a = classification_report(df['label'], df['comp_score'])

	b = confusion_matrix(df['label'], df['comp_score'])

	return render_template("amazon.html", X = a, Y = b)

@app.route('/home')
def home():
	return redirect('/')

if __name__ == '__main__':
	#app.debug = True
	app.run(debug = True, threaded= False)