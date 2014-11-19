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
		print "SOMETHING IS WRONG"
		return
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
			print "SOMETHING IS WRONG"

	return {"tp" : tp_count/total_pos, "fp" : fp_count/total_neg, "tn" : tn_count/total_neg, "fn" : fn_count/total_pos }

df = parse_feature_file(DATA_PATH + "samples_features_77811.csv")
labels = [int(x) for x in df['type'].values]

feat_df = df.drop('type', axis=1)
# feature_mat = feat_df.values
feat_mat = feat_df.values

# TODO: keep paired samples together 
feat_train, feat_test, label_train, label_test = train_test_split(feat_mat, labels, test_size=0.33, random_state=222)


neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(feat_train, label_train) 
test_results = list(neigh.predict(feat_test))

# print "TEST RESULTS:", list(test_results)
# print "ACTUAL:		", label_test

print compare_test_results(test_results, label_test)

