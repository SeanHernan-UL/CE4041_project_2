import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from mpl_toolkits.mplot3d import Axes3D

from adaboost import Adaboost

# import train data as a dataframe
# just using dataframes and hoping the small amount of extra overhead is fine...
train = pd.read_csv("datasets/adaboost-test-24.txt", sep=r'\s+', header=None, names=['x', 'y', 'class'])
test = pd.read_csv("datasets/adaboost-train-24.txt", sep=r'\s+', header=None, names=['x', 'y', 'class'])

# we need to add a weights column to the train set, with all weights initially being the same
# the weights will sum to 1
weights = np.ones(len(train))*(1/len(train))
train['weights'] = weights

print(train.head())

# create adaboost object
ada = Adaboost()

# create dict to store our collection of bad models
model_vals = []

# duplicate the train dataframe so the original isn't messed up by training model
training_df = train

# loop x amount of times, such that at some point we will have enough
# weak classifiers to be strong together
big_num = 4
features = ['x','y']
for x in range(0, big_num):

    # train weak learner, h
    stump_vals, bonus_vals = ada.train_treestump(training_df, features[x%2])

    for dict in stump_vals:
        print(dict)

    # set alpha value for given learner
    # judge how bad our weak learner is...
    epsilon = bonus_vals[0]
    alpha = (1/2)*math.log((1-epsilon)/epsilon)

    # store our model values
    model_vals.append([stump_vals, alpha])
    print(model_vals)

    # update weights distribution
    # misclassified points need to account for .5 of the weight...
    # currently have 382 misclassified points, their new weight is 0.5/382
    # we need to find these points, and the other points whose weights will be 0.5/(1200-382)
    training_df = ada.update_weights(training_df, stump_vals, bonus_vals)

    print('poop')

    print(train.head())

# return the strong classifier...
print(model_vals)

print('poop')
