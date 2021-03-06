import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import pylab as pl


# DATA_PATH = "/home/datascience/supernovae-classification/" # Make this the /path/to/the/data
import sys

DATA_PATH = "/Users/margaret/Documents/Class Materials/Fall 2014/CS 194-16/"


def parse_feature_file(filename):
    df = pd.DataFrame.from_csv(filename)
    return df.reset_index()


def find_pairs(df):
    """
    For a total of 5,236 of detections in our labelled set, we also have a paired candi-
    date that was detected at the same position (within 0.001 degrees) within 2.5 hours.

    Param: df is a DataFrame of the features and labels (direct read from samples_features_77811.csv)
    Returns: list of tuples of indices of paired data points.

    just kidding. these are only for the actual images, not the feature set.
    """
    pass


# Returns dict of metrics
def compare_test_results(test, actual):
    if (len(test) != len(actual)):
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
        if test[i] == actual[i]:
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

    return {"tp": tp_count / total_pos, "fp": fp_count / total_neg, "tn": tn_count / total_neg,
            "fn": fn_count / total_pos}

# how is this different than the ones built into models?
def predict_using_probs(probs, threshold):
    labels = []
    for pr0, pr1 in probs:
        if (pr1 > threshold):
            labels.append(1)
        else:
            labels.append(0)
    return labels


# True Positive/(True Positive + False Positive)
def getPrecision(test, actual):
    return sklearn.metrics.precision_score(actual, test)


# True Positive/(True Positive + True Negative)
def getRecall(test, actual):
    return sklearn.metrics.recall_score(actual, test)


#True Positive + True Negative / Everything
def getAccuracy(test, actual):
    return sklearn.metrics.accuracy_score(actual, test)


def results_summary_scikit(test, actual):
    precision = sklearn.metrics.precision_score(actual, test)
    recall = sklearn.metrics.recall_score(actual, test)
    accuracy = sklearn.metrics.accuracy_score(actual, test)
    results = {"precision": precision, "recall": recall, "accuracy": accuracy}
    return results


def graph_roc(predicted, labels):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, predicted[:, 1])
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    print "Area under the ROC curve : %f" % roc_auc

    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Logistic Regression ROC')
    pl.legend(loc="lower right")
    pl.show()


# Parse features and labels
def run_knn():
    df = parse_feature_file(DATA_PATH + "samples_features_77811.csv")
    labels = [int(x) for x in df['type'].values]
    feat_df = df.drop('type', axis=1)
    feat_mat = feat_df.values

    neigh = KNeighborsClassifier(n_neighbors=20, weights='distance')

    # Train and Validate n times

    feat_train, feat_test, label_train, label_test = train_test_split(feat_mat, labels, test_size=.33, random_state=100)
    neigh.fit(feat_train, label_train)
    probs = neigh.predict_proba(feat_test)
    prob_results = predict_using_probs(probs, 0.19)
    print results_summary_scikit(prob_results, label_test)

    # Compute ROC curve and area the curve
    graph_roc(probs, label_test)


# we should have made a generalized model evaluator... maybe later
# also the first half of this should probably just be some kind of setup function
def run_random_forest(find_threshold=False):
    df = parse_feature_file(DATA_PATH + "samples_features_77811.csv")
    labels = [int(x) for x in df['type'].values]
    feat_df = df.drop('type', axis=1)
    feat_mat = feat_df.values

    feat_train, feat_test, label_train, label_test = train_test_split(feat_mat, labels, test_size=.33, random_state=100)
    model = RandomForestClassifier()
    model.fit(feat_train, label_train)
    probs = model.predict_proba(feat_test)

    if find_threshold:
        for i in map(lambda x:x/10.0, xrange(10)):
            print "threshold {0}".format(i)
            prob_results = predict_using_probs(probs, i) # figure out optimal threshold for this.
            print results_summary_scikit(prob_results, label_test)
    else:
        prob_results = predict_using_probs(probs, 0.4)
        print results_summary_scikit(prob_results, label_test)
        # note that threshold of 0 results in {'recall': 0.99338842975206609, 'precision': 0.58842246970994982, 'accuracy': 0.86778565308824673}
        # precision is the fraction of retrieved instances that are relevant
        # recall is the fraction of relevant instances that are retrieved



if __name__ == "__main__":
    # run_knn()
    run_random_forest(find_threshold=True)