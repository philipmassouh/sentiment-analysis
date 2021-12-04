import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# read in the training data
tweet_url='https://github.com/aasiaeet/cse5522data/raw/master/db3_final_clean.csv'
tweet_dataframe=pd.read_csv(tweet_url)

# put unique words into a dictionary, assigning them with a unique ID
word_dict = {}
for i, w, tweet, l in tweet_dataframe.itertuples():
  for word in tweet.split(" "):
    if word not in word_dict:
      word_dict[word] = len(word_dict)

# create a document word matrix: [tweets x unique words]
x = np.zeros((tweet_dataframe.shape[0], len(word_dict)), dtype='float')

# tweet labels
y = np.array(tweet_dataframe.iloc[:,2])

# mark each word's occurence per tweet in the document-word matrix
for i, w, tweet, l in tweet_dataframe.itertuples():
  for word in tweet.split(" "):
    x[i, word_dict[word]] = 1

# compute totals and percentages on the dataset
neg_tot = np.sum(y<0)
pos_tot = len(y) - neg_tot
neg_pr = neg_tot / (neg_tot + pos_tot)
pos_pr = 1 - neg_pr

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Respectively,
# P(word|Sentiment = +ive)
# P(word|Sentiment = -ive)
# P(Sentiment = +ive)
# P(Sentiment = -ive)
def compute_distributions(x, y):
    pr_word_given_pos = np.mean(x[y>0,:],axis=0)
    pr_word_given_neg = np.mean(x[y<0,:],axis=0)
    prior_pos = np.mean(y>0)
    prior_neg = 1 - prior_pos

distros = compute_distros(xTrain,yTrain)


