from gravityai import gravityai as grav
import pickle
import re
import string
import pandas as pd


vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
LRC = pickle.load(open('logistic_regression_classifier.pkl', 'rb'))
DTC = pickle.load(open('decision_tree_classifier.pkl', 'rb'))
GBC = pickle.load(open('gradient_boosting_classifier.pkl', 'rb'))
RFC = pickle.load(open('random_forest_classifier.pkl', 'rb'))


def clean_text(text):
	text = text.lower()
	text = re.sub('\[.*?\]', '', text)
	text = re.sub("\\W"," ",text)
	text = re.sub('https?://\S+|www\.\S+', '', text)
	text = re.sub('<.*?>+', '', text)
	text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
	text = re.sub('\n', '', text)
	text = re.sub('\w*\d\w*', '', text)
	return text


def output_lablel(n):
	if n == 0:
		return "Fake News"
	if n == 1:
		return "Real News"


def process_dataframe(news_df):
	result = news_df.copy()

	news_df["clear_text"] = news_df["text"].apply(clean_text)
	new_x_test = news_df["clear_text"]
	new_xv_test = vectorizer.transform(new_x_test)

	pred_LRC = pd.DataFrame(LRC.predict(new_xv_test), columns=['label'])
	pred_DTC = pd.DataFrame(DTC.predict(new_xv_test), columns=['label'])
	pred_GBC = pd.DataFrame(GBC.predict(new_xv_test), columns=['label'])
	pred_RFC = pd.DataFrame(RFC.predict(new_xv_test), columns=['label'])

	result['LRC Prediction'] = pred_LRC['label'].apply(output_lablel)
	result['DTC Prediction'] = pred_DTC['label'].apply(output_lablel)
	result['GBC Prediction'] = pred_GBC['label'].apply(output_lablel)
	result['RFC Prediction'] = pred_RFC['label'].apply(output_lablel)

	return result


def process(inPath, outPath):
	input_df = pd.read_csv(inPath)
	output_df = process_dataframe(input_df)
	output_df.to_csv(outPath, index=False)


grav.wait_for_requests(process)
