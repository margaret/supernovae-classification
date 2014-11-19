#from pylab import *
import pandas as pd

DATA_PATH = "/home/datascience/supernovae-classification/" # Make this the /path/to/the/data



def parse_feature_file(filename):
    df = pd.read_csv(filename)
    return df.reset_index()


df = parse_feature_file(DATA_PATH + "samples_features_small.csv")
labels = [int(x) for x in df['type'].values]

# feat_df = df.drop('type', axis=1)
# feature_mat = feat_df.values
feature_mat = df.values
print feature_mat
