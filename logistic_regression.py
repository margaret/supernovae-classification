import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import sys


DATA_PATH = "/home/datascience/Documents/project/supernovae-classification/" # Make this the /path/to/the/data

def parse_feature_file(filename):
    df = pd.DataFrame.from_csv(filename)
    return df.reset_index()

# Returns dict of metrics
def compare_test_results(test, actual):
	if(len(test) != len(actual)):
		print "input length mismatch"
	fp_count = 0.0
	tp_count = 0.0
	fn_count = 0.0
	tn_count = 0.0
	total = len(actual)
	total_pos = sum(actual)
	total_neg = total - total_pos

	#print "TOTAL POS :", total_pos
	#print "TOTAL NEG :", total_neg
	for i in range(len(test)):
		if test[i]==actual[i]:
			if test[i] == 1:
				tp_count += 1.0
			else:
				tn_count += 1.0
		elif test[i] == 1 and actual[i] == 0:
			fp_count += 1.0
		elif test[i] == 0 and actual[i] == 1:
			fn_count += 1.0
		else:
			print "input contains invalid value"

	return {"tp" : tp_count/total_pos, "fp" : fp_count/total_neg, "tn" : tn_count/total_neg, "fn" : fn_count/total_pos }

#True Positive/(True Positive + False Positive)
def getPrecision(test, actual):
	results = compare_test_results(test, actual)
	tp = results['tp']
	fp = results['fp']
	return (tp / (tp + fp))

#True Positive/(True Positive + True Negative)
def getRecall(test, actual):
	results = compare_test_results(test, actual)
	tp = results['tp']
	fn = results['fn']
	return (tp / (tp + fn))

#True Positive + True Negative / Everything
def getAccuracy(test, actual):
	results = compare_test_results(test, actual)
	tp = results['tp']
	tn = results['tn']
	return (tp + tn) / len(test)

def results_summary(test, actual):
	precision = getPrecision(test, actual)
	recall = getRecall(test, actual)
	accuracy = getAccuracy(test, actual)
	results = {"precision": precision, "recall": recall, "accuracy": accuracy}
	return results


def predict_using_probs(probs, threshold):
	labels = []
	for pr0, pr1 in probs:
		if(pr1 > threshold):
			labels.append(1)
		else:
			labels.append(0)
	return labels

def find_max_threshold(predicted):
	max_check = -1.0
	max_threshold = 1
	for i in range(40, 49):
		prob_predicted = predict_using_probs(predicted, float(i)/100)
		results = results_summary(prob_predicted, label_test)
		max_temp = results['precision'] + results['recall'] + results['accuracy']
		if(max_temp > max_check):
			max_check = max_temp
			max_threshold = float(i) / 100
	print max_threshold



# Parse features and labels
df = parse_feature_file(DATA_PATH + "samples_features_77811.csv")
labels = [int(x) for x in df['type'].values]
feat_df = df.drop('type', axis=1)
feat = feat_df.values
feat_train, feat_test, label_train, label_test = train_test_split(feat, labels, test_size=0.33, random_state=100)

model = LogisticRegression()
model.fit(feat_train, label_train)
predicted = model.predict_proba(feat_test)
find_max_threshold(predicted)
prob_predicted = predict_using_probs(predicted, .45)
print results_summary(prob_predicted, label_test)