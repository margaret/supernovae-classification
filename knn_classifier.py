#from pylab import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

DATA_PATH = "/home/datascience/supernovae-classification/" # Make this the /path/to/the/data

def parse_feature_file(filename):
    df = pd.DataFrame.from_csv(filename)
    return df.reset_index()

# Returns dict of metrics
def compare_test_results(test, actual):
	if(len(test) != len(actual)):
		raise InputError("input length mismatch")
	fp_count = 0.0
	tp_count = 0.0
	fn_count = 0.0
	tn_count = 0.0
	total = len(actual)
	total_pos = sum(actual)
	total_neg = total - total_pos

	print "TOTAL POS :", total_pos
	print "TOTAL NEG :", total_neg
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
			raise InputError("input contains invalid value")

	return {"tp" : tp_count/total_pos, "fp" : fp_count/total_neg, "tn" : tn_count/total_neg, "fn" : fn_count/total_pos }

def predict_using_probs(probs, threshold):
	labels = []
	for pr0, pr1 in probs:
		if(pr1 > threshold):
			labels.append(1)
		else:
			labels.append(0)
	return labels

# Parse features and labels
df = parse_feature_file(DATA_PATH + "samples_features_77811.csv")
labels = [int(x) for x in df['type'].values]
feat_df = df.drop('type', axis=1)
feat_mat = feat_df.values

neigh = KNeighborsClassifier(n_neighbors=20, weights='distance')

n = 5
## Train and Validate n times
for i in range(n):
	# TODO: keep paired samples together 
	feat_train, feat_test, label_train, label_test = train_test_split(feat_mat, labels, test_size=1.0/n, random_state=i)
	neigh.fit(feat_train, label_train) 
	probs = neigh.predict_proba(feat_test)
	prob_results = predict_using_probs(probs, 0.19)
	#print probs
	print compare_test_results(prob_results, label_test)

# best_thres = 0.0
# best_total = 0.0
# for i in range(100):
# 	thres = i / 100.0
# 	pr = predict_using_probs(probs, thres)
# 	result = compare_test_results(pr, label_test)
# 	if result['tp'] + result['tn'] > best_total:
# 		best_total = result['tp'] + result['tn']
# 		best_thres = thres

# print "Best thresold is ", best_thres
