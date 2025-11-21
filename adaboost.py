import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors


class Adaboost():

    def __init__(self,debug=False):
        self.debug = debug

    def train_stump(self, train_np, feature):

        # decode allowed features
        key = Adaboost._decode_allowed_features(self,feature)

        # sort the data for the given feature...
        train_np = train_np[train_np[:,key].argsort()]

        scores = np.zeros(len(train_np), dtype=np.float64)
        # try and split to minimize the misclassifications...
        # we just brute force :)
        smallest_misclassifications = len(train_np)
        idx_misclassified = np.zeros([1,len(train_np)], dtype=np.float64)
        best_split = 0
        best_score = 0
        polarity = [0,0]
        for split in range(0,len(train_np)):
            num_misclassified, score, pol = Adaboost._test_split(self,train_np, split)
            scores[split] = score

            # find the smallest number of misclassifications...
            if score > best_score:
                smallest_misclassifications = num_misclassified
                best_split = split
                best_score = score
                polarity = pol

        print(f'smallest_misclassifications: {smallest_misclassifications}')
        print(f'best_split: {best_split}')
        print(f'best_score: {best_score}')
        print(f'polarity: {polarity}')

        # find where we should draw line based off best split...
        # get value of feature for point on either side, take average... <- TODO apparently this is wrong...
        if best_split == 0:
            line = train_np[0,key]
        elif best_split == len(train_np)-1:
            line = train_np[len(train_np)-1,key]
        else:
            line = (train_np[best_split-1,key] + train_np[best_split,key])/2 #<- TODO apparently this is wrong... you aren't supposed to just average??

        epsilon = 1 - best_score[0] # for weighted misclassification

        return (feature, float(line), polarity), (epsilon, smallest_misclassifications)

    def update_weights(self,train_np, stump_vals, bonus_vals):

        # extract stump_vals
        feature = stump_vals[0]
        line_location = stump_vals[1]
        polarity = stump_vals[2]

        epsilon = bonus_vals[0]
        num_misclassifications = bonus_vals[1]

        key = Adaboost._decode_allowed_features(self, feature)

        # sort the data for the given feature...
        train_np = train_np[train_np[:,key].argsort()]

        unique, counts = np.unique(train_np[:, 2], return_counts=True)
        debug = dict(zip(unique, counts))

        # loop through feature values
        for i in range(0, len(train_np[:,key])):
            # if we are greater than the line
            if train_np[i,key] < line_location:
                # if we are on the correct side of the polarity
                if train_np[i,2] == polarity[0]:
                    # decrease the weight
                    train_np[i, 3] *= 1 / (2 * (1 - epsilon))
                else:
                    # increase the weight
                    train_np[i, 3] *= 1 / (2 * epsilon)
            else: # we are less than the line
                # if we are on the correct side of the polarity
                if train_np[i,2] == polarity[1]:
                    # decrease the weight
                    train_np[i, 3] *= 1 / (2 * (1 - epsilon))
                else:
                    # increase the weight
                    train_np[i, 3] *= 1 / (2 * epsilon)

        # FIXME getting weird behaviour where this starts as 1, but slowly increases
        sum_weights = np.sum(train_np[:,3]) # this needs to always equal 1
        if sum_weights > 1.1:
            raise ValueError(f'Sum of weights is not 1, it is: {sum_weights}')

        return train_np # with update weights...

    def visualise_model(self,model_vals, limits, dataset_points_df):

        # contour & contourf

        ## this is horrible...

        # we want to get the steppy plot that Colin had

        # init big grid to all zeros
        size = 1000
        grid = np.zeros([size,size])
        print(grid.shape)

        # convert model vals to numpy
        if isinstance(model_vals, pd.DataFrame):
            model_vals = model_vals.to_numpy()

        # get x and y axes
        x_axis = np.linspace(limits[0][0], limits[0][1], size)
        y_axis = np.linspace(limits[1][0], limits[1][1], size)

        # iterate over all weak models
        for weak in model_vals:

            # extract stump_vals
            feature = weak[0]
            line_location = weak[1]
            polarity = weak[2]
            alpha = weak[3]

            # if feature == 'x'
            axis = 0
            if feature == 'y':
                axis = 1

            for idx in range(0,size):
                # 'draw' line and add the polarity multiplied by the models alpha to the corresponding
                # values on the grid

                if feature == 'x':
                    if x_axis[idx] > line_location:
                        grid[:,idx] += polarity[1]*alpha
                    else:
                        grid[idx,:] += polarity[0]*alpha
                else: # feature == 'y'
                    if y_axis[idx] > line_location:
                        grid[:,idx] += polarity[1]*alpha
                    else:
                        grid[idx,:] += polarity[0]*alpha

        # round to 1 or -1
        grid_rounded = np.zeros([size,size])
        for i in range(0,size):
            for j in range(0,size):
                if grid[i,j] > 0:
                    grid_rounded[i,j] = 1
                else:
                    grid_rounded[i,j] = -1


        # do a 3d plot with levels visualised with colour
        # Create a 3D line plot with Seaborn
        # doing heatmap for now
        # plt.figure(1)
        # ax = plt.axes()
        # cont = ax.contourf(x_axis, y_axis, grid)
        # plt.colorbar(cont)
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title(f"Model 'Heatmap'")

        # round to [1,-1] values, and plot
        # also plot the train set ontop...
        plt.figure(2)
        pallete = colors.LinearSegmentedColormap.from_list('pallete',['#E3F2FD', '#C8E6C9'])
        ax = plt.axes()
        cont = ax.contourf(x_axis, y_axis, grid_rounded, cmap=pallete)
        plt.colorbar(cont)

        pallete = ['#1B5E20', '#0D47A1']
        sns.scatterplot(x='x', y='y', data=dataset_points_df, hue='class', palette=pallete)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"Model 'Heatmap', with overlayed dataset")

        # do a 3d plot with levels visualised with colour
        # Create a 3D line plot with Seaborn
        # doing heatmap for now
        # plt.figure(3)
        # ax = sns.heatmap(grid, annot=False, square=True)
        # ax.invert_yaxis()
        # plt.xlabel('Actual Value')
        # plt.ylabel('Predicted value')
        # plt.title(f'Confusion Matrix')

        # plt.figure(4)
        # ax = sns.heatmap(grid_rounded, annot=False, square=True)
        # ax.invert_yaxis()
        # plt.xlabel('Actual Value')
        # plt.ylabel('Predicted value')
        # plt.title(f'Confusion Matrix')

        plt.show()


    def test_model(self, train, model_vals):

        # given a dataset, tries to classify it based off given model_vals

            # for each feature I can create bins


    #
    # def _predict(self, model_vals):
    #
    #     # given a point tries to classifiy it based off given model_vals

            # I think best way is to just create a big grid, with each point labelled
            # as 1 or -1 and simply indexing into it, the grid can just be made bigger to
            # add extra resolution

            # this reduces to the already solved problem of visualising the model

    def _test_split(self, train_np, split):

        ## beginning to get slow enough that its probably best to switch to numpy

        ## will handle trying the class labels both ways itself...

        above_misclassified = {-1.0: 0, 1.0: 0} # init to dict
        above_weight_score = np.zeros([2,1],dtype=np.float64)

        below_misclassified = {-1.0: 0, 1.0: 0} # init to dict
        below_weight_score = np.zeros([2,1],dtype=np.float64)

        total_weight_score = np.zeros([2,1],dtype=np.float64)

        # split the array
        # already sorted so all we care about are the class and index
        if split != 0:
            above_split = train_np[:split, :]

            # technically not relevant
            unique, counts = np.unique(above_split[:,2], return_counts=True)
            above_misclassified = dict(zip(unique, counts))

            # main thing we care about
            for i in range(0, len(above_split)):
                if above_split[i,2] == -1:
                    # counting -1 class instances as errors
                    above_weight_score[0] += above_split[i,3]
                else:
                    # counting 1 class instances as  errors
                    above_weight_score[1] += above_split[i,3]

        if split != len(train_np):
            below_split = train_np[split:, :]

            # technically not relevant
            unique, counts = np.unique(below_split[:, 2], return_counts=True)
            below_misclassified = dict(zip(unique, counts))

            # main thing we care about
            for i in range(0, len(below_split)):
                if below_split[i,2] == 1:
                    # counting 1 class instances as errors
                    below_weight_score[0] += below_split[i,3]
                else:
                    # counting -1 class instances as  errors
                    below_weight_score[1] += below_split[i,3]

        # if split == 0:
        #     ## check that our weight scores make sense
        #     # Case 0 (polarity = [1,-1]):
        #     # sum of misclassifications (-1) and correct classification (1) weights
        #     # should equal 1
        #     # rounding to 6 decimal points to avoid floating point noise
        #     Adaboost._check_weights_score_valid(self,above_weight_score[0] + below_weight_score[1],'case0')
        #
        #     # Case 1 (polarity = [-1,1]):
        #     # sum of misclassifications (1) and correct classification (-1) weights
        #     # should equal 1
        #     Adaboost._check_weights_score_valid(self,above_weight_score[1] + below_weight_score[0], 'case1')

        total_weight_score[0] = above_weight_score[0] + below_weight_score[0]
        total_weight_score[1] = above_weight_score[1] + below_weight_score[1]

        # work out which case has better score...
        case = 0
        if total_weight_score[1] < total_weight_score[0]:
            case = 1

        polarity = [[1,-1],[-1,1]]

        # work out the number of misclassified
        # need to check if there are any values for a given key
        total_misclassified = 0
        if polarity[case][0] in above_misclassified.keys():
            total_misclassified += above_misclassified[polarity[case][0]]
        if polarity[case][1] in below_misclassified.keys():
            total_misclassified += below_misclassified[polarity[case][1]]

        # print for debug
        # print(f'{split}\t{above_misclassified}\t{below_misclassified}\t\t{total_weight_score[0]}\t{total_weight_score[1]}')

        # FIXME getting weird behaviour where this starts as 1, but slowly increases, something to do with floating point precision I think
        # take the highest weight score (best case), and round to 6 decimal points to avoid
        # floating point noise
        best_weight_score = max(total_weight_score)

        return total_misclassified, best_weight_score, polarity[case]

    def _decode_allowed_features(self,feature):

        # decode allowed features
        if feature == 'x':
            return 0
        elif feature == 'y':
            return 1
        else:
            # an invalid feature
            raise ValueError(feature)

    # def _check_weights_score_valid(self, weights_score,label):
    #
    #     if self.debug:
    #         print(f'weights_score_{label} = {weights_score}')
    #
    #     if weights_score > 1.01:
    #         raise ValueError(f'Sum of weights is not 1, it is: {weights_score}')