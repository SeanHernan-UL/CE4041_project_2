import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

from adaboost import Adaboost

# import train data as a dataframe
# just using dataframes and hoping the small amount of extra overhead is fine...
# dataframes isn't fast enough, use at start and end, but in middle use standard numpy arrays
train_df = pd.read_csv("datasets/adaboost-test-24.txt", sep=r'\s+', header=None, names=['x', 'y', 'class'])
test_df = pd.read_csv("datasets/adaboost-train-24.txt", sep=r'\s+', header=None, names=['x', 'y', 'class'])

# we need to add a weights column to the train set, with all weights initially being the same
# the weights will sum to 1
weights = np.ones(len(train_df))*(1/len(train_df))
train_df['weights'] = weights

print(train_df.head())

# create adaboost object
ada = Adaboost(debug=False)

# create dict to store our collection of bad models
model_vals = []

# cast the dataframe to a numpy array for the training...
train_np = train_df.to_numpy(dtype=np.float64)

# loop x amount of times, such that at some point we will have enough
# weak classifiers to be strong together
big_num = 25
features = ['x','y']
for x in range(0, big_num):
    print(f'## Round {x} ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # train weak learner, h
    stump_vals, bonus_vals = ada.train_stump(train_np, features[x%2])

    # print stump vals
    print(f'feature: {stump_vals[0]}, location: {stump_vals[1]}, polarity: {stump_vals[2]}')

    # set alpha value for given learner
    # judge how bad our weak learner is...
    epsilon = bonus_vals[0]


    print(f'epsilon: {epsilon}')
    alpha = (1/2)*math.log((1-epsilon)/epsilon)
    print(f'alpha: {alpha}')

    # store our model values
    # just taking first value of polarity, since second can be deduced from first...
    model_vals.append([stump_vals[0],stump_vals[1],stump_vals[2], alpha])
    print(model_vals)

    # update weights distribution
    # misclassified points need to account for .5 of the weight...
    # currently have 382 misclassified points, their new weight is 0.5/382
    # we need to find these points, and the other points whose weights will be 0.5/(1200-382)
    train_np = ada.update_weights(train_np, stump_vals, bonus_vals)

    # visualise model, with updated weights...
    # pallete = ['#008080', '#FF00FF']
    # plt.figure(1)
    # # plot data
    # sns.scatterplot(x='x', y='y', data=train, hue='class', palette=pallete, size='weights')
    # # draw horizonta/vertical split line
    # if x%2 == 0:
    #     plt.vlines(x=stump_vals[1]['location'], ymin=-2,  ymax=2, colors='black')
    # else:
    #     plt.hlines(y=stump_vals[1]['location'], xmin=-2, xmax=2, colors='black')
    # plt.title(f'adaboost-train-24: round {x}')
    # plt.show()

    # # store model values in csv
    # final_model = pd.DataFrame(model_vals, columns=['feature', 'location', 'polarity', 'alpha'])
    #
    # # visualise final model...
    # limits = [[-2.2, 2.2], [-2.2, 2.2]]
    # # getting original dataset, so its not modified
    # ada.visualise_model(final_model, limits, train_df)


# return the strong classifier...
for weak in model_vals:
    print(weak)

# store model values in csv
final_model = pd.DataFrame(model_vals, columns=['feature', 'location', 'polarity', 'alpha'])

# get a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# check if models folder exists, if not, create...
os.makedirs('./models', exist_ok=True)

# store to csv
final_model.to_csv(f'models/adaboost_model_{timestamp}.csv', index=False)

# visualise final model...
limits = [[-2.2,2.2],[-2.2,2.2]]
# getting original dataset, so its not modified
ada.visualise_model(final_model,limits,train_df)

print('poop')
