# ce4041 adaboost project
# name: Kanvar Murray
# student id: 22374698
# date: 24/11/25

# members = [('Kanvar Murray', 22374698), ('Seán Hernan', 22348948), ('Madeline Ware', 21306591)]

import numpy as np


class WeightedWeakLinear:

    # set up empty parameters
    def __init__(self):
        # store line direction
        self.w = None
        # store cut position
        self.threshold = None
        # store polarity for +1 side
        self.polarity = 1
        # store weighted error
        self.error = None

    # train weak linear model
    def fit(self, x, y, weights):
        x = np.asarray(x)
        y = np.asarray(y).astype(int)
        weights = np.asarray(weights)

        # class masks
        mask_positive = (y == 1)
        mask_neg = (y == -1)

        # sum weights by class
        weights_positive_mask = np.sum(weights[mask_positive])
        w_neg_sum = np.sum(weights[mask_neg])

        # fallback if one class has zero weight cause yikessssss
        if weights_positive_mask == 0 or w_neg_sum == 0:
            self.w = np.array([1.0, 0.0])
        else:
            # weighted means
            mu_pos = np.sum(x[mask_positive] * weights[mask_positive, None], axis=0) / weights_positive_mask
            mu_neg = np.sum(x[mask_neg] * weights[mask_neg, None], axis=0) / w_neg_sum

            # direction between means
            direction = mu_pos - mu_neg
            norm = np.linalg.norm(direction)

            # normalise direction
            self.w = direction / norm if norm != 0 else np.array([1.0, 0.0])
        # project points onto direction like 2D to 1D
        z = x @ self.w

        # sort projections
        idx = np.argsort(z)
        z_sorted = z[idx]
        y_sorted = y[idx]
        w_sorted = weights[idx]

        # handle flat projection
        if np.allclose(z_sorted, z_sorted[0]):
            self.threshold = z_sorted[0]

            # test constant predictions
            pred_pos = np.ones_like(y)
            pred_neg = -np.ones_like(y)
            err_pos = np.sum(weights[pred_pos != y])
            err_neg = np.sum(weights[pred_neg != y])

            # keep lower error choice
            if err_pos <= err_neg:
                self.polarity = 1
                self.error = err_pos
            else:
                self.polarity = -1
                self.error = err_neg
            return

        # candiidate thresholds between sorted values
        candidates = (z_sorted[:-1] + z_sorted[1:]) / 2

        # track best cut
        best_error = np.inf
        best_theta = None
        best_polarity = 1

        # try each cut with both polarities
        for theta in candidates:
            # right side +1
            pred_plus = np.where(z >= theta, 1, -1)
            err_plus = np.sum(weights[pred_plus != y])

            if err_plus < best_error:
                best_error = err_plus
                best_theta = theta
                best_polarity = 1

            # left side +1
            pred_minus = np.where(z <= theta, 1, -1)
            err_minus = np.sum(weights[pred_minus != y])

            if err_minus < best_error:
                best_error = err_minus
                best_theta = theta
                best_polarity = -1

        # store best values
        self.threshold = best_theta
        self.polarity = best_polarity
        self.error = best_error

    # predict labels
    def predict(self, X):
        # require model to be trained
        if self.w is None or self.threshold is None:
            raise ValueError("model not fitted")

        X = np.asarray(X)
        # project points
        z = X @ self.w

        # aply polarity and threshold
        if self.polarity == 1:
            return np.where(z >= self.threshold, 1, -1).astype(int)
        else:
            return np.where(z <= self.threshold, 1, -1).astype(int)


class WeightedDecisionStump:
    # edit added weighted decision stump weak learner based on peers implementation

    def __init__(self):
        # which feature is used 0 for x 1 for y
        self.feature = None
        # threshold value
        self.split = None
        # polarity flag
        self.polarity = 1
        # weighted error
        self.error = None

    def fit(self, points, classes, weights):
        points = np.asarray(points)
        classes = np.asarray(classes).astype(int)
        weights = np.asarray(weights)

        n_samples, n_features = points.shape

        if n_features < 2:
            raise ValueError("expected at least 2 features for stump")

        best_error = np.inf
        best_feature = 0
        best_split = 0.0
        best_polarity = 1

        # loop over x and y features
        for feature in range(2):
            feature_values = points[:, feature]

            # sort along this feature
            idx = np.argsort(feature_values)
            feature_values_sorted = feature_values[idx]
            classes_sorted = classes[idx]
            weights_sorted = weights[idx]

            # if all values are the same handle as constant
            if np.allclose(feature_values_sorted, feature_values_sorted[0]):
                # all plus one
                pred_pos = np.ones_like(classes_sorted)
                err_pos = np.sum(weights_sorted[pred_pos != classes_sorted])

                # all minus one
                pred_neg = -np.ones_like(classes_sorted)
                err_neg = np.sum(weights_sorted[pred_neg != classes_sorted])

                if err_pos < best_error:
                    best_error = err_pos
                    best_feature = feature
                    best_split = feature_values_sorted[0]
                    best_polarity = 1
                if err_neg < best_error:
                    best_error = err_neg
                    best_feature = feature
                    best_split = feature_values_sorted[0]
                    best_polarity = -1
                continue

            # candidate splits midway between sorted points
            candidate_splits = (feature_values_sorted[:-1] + feature_values_sorted[1:]) / 2.0

            for split in candidate_splits:
                # case 1 right side +1 left side -1
                pred_plus = np.where(feature_values >= split, 1, -1)
                err_plus = np.sum(weights[pred_plus != classes])

                if err_plus < best_error:
                    best_error = err_plus
                    best_feature = feature
                    best_split = split
                    best_polarity = 1

                # case 2 left side +1 right side -1
                pred_minus = np.where(feature_values <= split, 1, -1)
                err_minus = np.sum(weights[pred_minus != classes])

                if err_minus < best_error:
                    best_error = err_minus
                    best_feature = feature
                    best_split = split
                    best_polarity = -1

        # store best stump parameters
        self.feature = best_feature
        self.split = best_split
        self.polarity = best_polarity
        self.error = best_error

    def predict(self, point):
        if self.feature is None or self.split is None:
            raise ValueError("stump not fitted")

        point = np.asarray(point)
        feature_values = point[:, self.feature]

        if self.polarity == 1:
            # right side is +1
            return np.where(feature_values >= self.split, 1, -1).astype(int)
        else:
            # left side is +1
            return np.where(feature_values <= self.split, 1, -1).astype(int)
