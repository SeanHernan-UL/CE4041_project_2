import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors


class Adaboost():

    def train_stump(self, df, feature):

        # sort the data for the given feature...
        df.sort_values(feature, inplace=True, ignore_index=True)

        scores = np.zeros(len(df))
        # try and split to minimize the misclassifications...
        # we just brute force :)
        smallest_misclassifications = len(df)
        idx_misclassified = pd.DataFrame()
        best_split = 0
        best_score = 0
        polarity = [0,0]
        for split in range(0,len(df)):
            idx, num, score, pol= Adaboost._test_split(self,df, split)
            scores[split] = score

            # find the smallest number of misclassifications...
            if score > best_score:
                smallest_misclassifications = num
                idx_misclassified = idx
                best_split = split
                best_score = score
                polarity = pol

        print(f'idx_misclassified: {idx_misclassified}')
        print(f'smallest_misclassifications: {smallest_misclassifications}')
        print(f'best_split: {best_split}')
        print(f'best_score: {best_score}')

        # find where we should draw line based off best split...
        # get value of feature for point on either side, take average...
        if best_split == 0:
            line = df[feature].iloc[0]
        elif best_split == len(df)-1:
            line = df[feature].iloc[len(df)-1]
        else:
            line = (df[feature].iloc[best_split-1] + df[feature].iloc[best_split])/2

        # calculate our weighted error, epsilon...
        # iterate over our original sorted dataframe, referring to new idx_misclassifications to see
        # which weights to add up...
        epsilon = 0
        for i in range(0, len(df)):
            if idx_misclassified.iloc[i].any():
                epsilon += df['weights'].iloc[i]

        scores_df = pd.DataFrame({'Score': scores}, index=df.index)

        return (feature, float(line), polarity), (epsilon, smallest_misclassifications)

    def update_weights(self,df, stump_vals, bonus_vals):

        # extract stump_vals
        feature = stump_vals[0]
        line_location = stump_vals[1]
        polarity = stump_vals[2]

        epsilon = bonus_vals[0]
        num_misclassifications = bonus_vals[1]

        # sort the data for the given feature...
        df.sort_values(feature, inplace=True, ignore_index=True)

        # loop through feature values
        # work them out in a numpy array, then overwrite the dataframe col
        arr = df['weights'].to_numpy()
        for i in range(0, len(df)):
            # if feature values > line location
            if (df[feature].iloc[i] > line_location):
                if (df['class'].iloc[i] == polarity[0]):
                    # weight = 0.5/misclassifications
                    arr[i] *= 1/(2*epsilon) # 0.5/num_misclassifications
                else:
                    # weight = 0.5/(len(df)-misclassifications)
                    arr[i] *= 1/(2*(1-epsilon)) #0.5/(len(df) - num_misclassifications)
            else:
                if (df['class'].iloc[i] == polarity[1]):
                    # weight = 0.5/misclassifications
                    arr[i] *= 1/(2*epsilon) # 0.5/num_misclassifications
                else:
                    # weight = 0.5/(len(df)-misclassifications)
                    arr[i] *= 1/(2*(1-epsilon)) #0.5/(len(df) - num_misclassifications)

        # update dataframe
        df['weights'] = arr

        return df # with update weights...

    def visualise_model(self,model_vals, limits, dataset_points_df):

        # contour & contourf

        ## this is horrible...

        # we want to get the steppy plot that Colin had

        # init big grid to all zeros
        size = 100
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
        plt.figure(1)
        ax = plt.axes()
        cont = ax.contourf(x_axis, y_axis, grid)
        plt.colorbar(cont)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f"Model 'Heatmap'")

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
        plt.figure(3)
        ax = sns.heatmap(grid, annot=False, square=True)
        ax.invert_yaxis()
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted value')
        plt.title(f'Confusion Matrix')

        plt.figure(4)
        ax = sns.heatmap(grid_rounded, annot=False, square=True)
        ax.invert_yaxis()
        plt.xlabel('Actual Value')
        plt.ylabel('Predicted value')
        plt.title(f'Confusion Matrix')

        plt.show()


    # def test_model(self, train, model_vals):
    #
    #     # given a dataset, tries to classify it based off given model_vals
    #
    # def _predict(self, model_vals):
    #
    #     # given a point tries to classifiy it based off given model_vals

    def _test_split(self, df, split):

        ## beginning to get slow enough that its probably best to switch to numpy

        ## will handle trying the class labels both ways itself...

        above_idx = pd.DataFrame()
        above_misclassified = 0
        above_weight_score = [0,0]

        below_idx = pd.DataFrame()
        below_misclassified = 0
        below_weight_score = [0,0]

        total_weight_score = [0,0]

        # split the dataframe
        # already sorted so all we care about are the class and index
        if split != 0:
            above_split = df.iloc[:split, :]

            above_idx = above_split['class'].isin([-1])
            above_misclassified = sum(above_idx)
            
            for i in range(0, len(above_split)):
                if above_idx.iloc[i]:
                    above_weight_score[0] += above_split['weights'].iloc[i]
                else:
                    above_weight_score[1] += above_split['weights'].iloc[i]

        if split != len(df):
            below_split = df.iloc[split:, :]

            below_idx = below_split['class'].isin([1])
            below_misclassified = sum(below_idx)

            for i in range(0, len(below_split)):
                if below_idx.iloc[i]:
                    below_weight_score[0] += below_split['weights'].iloc[i]
                else:
                    below_weight_score[1] += below_split['weights'].iloc[i]

        total_weight_score[0] = above_weight_score[0] + below_weight_score[0]
        total_weight_score[1] = above_weight_score[1] + below_weight_score[1]

        # work out which case has better score...
        polarity = [1, -1]
        above_misclassified = sum(above_idx)
        below_misclassified = sum(below_idx)
        if total_weight_score[1] < total_weight_score[0]:
            polarity = [-1, 1]
            # invert misclassified
            above_idx = ~above_idx
            below_idx = ~below_idx
            # calculate number
            above_misclassified = sum(above_idx)
            below_misclassified = sum(below_idx)

        # print for debug
        # print(f'{split}\t{above_misclassified}\t{below_misclassified}\t\t{total_weight_score[0]}\t{total_weight_score[1]}')

        return pd.concat([above_idx, below_idx], axis=0), (above_misclassified + below_misclassified), max(total_weight_score), polarity
