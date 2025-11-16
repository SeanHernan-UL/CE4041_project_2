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

# loop x amount of times, such that at some point we will have enough
# weak classifiers to be strong together
bigNum = 1000
for x in range(0, bigNum):

    # train weak learner, h
    stump_vals, epsilon = ada.train_treestump(train, 'y')

    for dict in stump_vals:
        print(dict)

    # set alpha value for given learner
    # judge how bad our weak learner is...
    alpha = (1/2)*math.log((1-epsilon)/epsilon)

    # update weights distribution
    # misclassified points need to account for .5 of the weight...

    print('poop')

# return the strong classifier...
