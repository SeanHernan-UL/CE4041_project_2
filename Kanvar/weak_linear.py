#ce4041 adaboost project
#name: Kanvar Murray
#student id: 22374698
#date: 21/11/25



import numpy as np



class WeightedWeakLinear:

    #set up empty parameters
    def __init__(self):
        #store line direction
        self.w=None
        #store cut position
        self.threshold=None
        #store polarity for +1 side
        self.polarity=1
        #store weighted error
        self.error=None






    #train weak linear model
    def fit(self,X,y,sample_weight):
        X=np.asarray(X)
        y=np.asarray(y).astype(int)
        w=np.asarray(sample_weight)

        #clas masks
        mask_pos=(y==1)
        mask_neg=(y==-1)

        #sum weights by class
        w_pos_sum=np.sum(w[mask_pos])
        w_neg_sum=np.sum(w[mask_neg])

        #fallback if one class has zero weight cause yikessssss
        if w_pos_sum==0 or w_neg_sum==0:
            self.w=np.array([1.0,0.0])
        else:
            #weighted means
            mu_pos=np.sum(X[mask_pos]*w[mask_pos,None],axis=0)/w_pos_sum
            mu_neg=np.sum(X[mask_neg]*w[mask_neg,None],axis=0)/w_neg_sum

            #direction between means
            direction=mu_pos-mu_neg
            norm=np.linalg.norm(direction)

            #normalise direction
            self.w=direction/norm if norm!=0 else np.array([1.0,0.0])
        #project points onto direction like 2D to 1D
        z=X@self.w

        #sort projections
        idx=np.argsort(z)
        z_sorted=z[idx]
        y_sorted=y[idx]
        w_sorted=w[idx]


        #handle flat projection
        if np.allclose(z_sorted,z_sorted[0]):
            self.threshold=z_sorted[0]

            #test constant predictions
            pred_pos=np.ones_like(y)
            pred_neg=-np.ones_like(y)
            err_pos=np.sum(w[pred_pos!=y])
            err_neg=np.sum(w[pred_neg!=y])

            #keep lower error choice
            if err_pos<=err_neg:
                self.polarity=1
                self.error=err_pos
            else:
                self.polarity=-1
                self.error=err_neg
            return

        #candiidate thresholds between sorted values
        candidates=(z_sorted[:-1]+z_sorted[1:])/2

        #track best cut
        best_error=np.inf
        best_theta=None
        best_polarity=1




        #try each cut with both polarities
        for theta in candidates:
            #right side +1
            pred_plus=np.where(z>=theta,1,-1)
            err_plus=np.sum(w[pred_plus!=y])

            if err_plus<best_error:
                best_error=err_plus
                best_theta=theta
                best_polarity=1

            #left side +1
            pred_minus=np.where(z<=theta,1,-1)
            err_minus=np.sum(w[pred_minus!=y])

            if err_minus<best_error:
                best_error=err_minus
                best_theta=theta
                best_polarity=-1

        #store best values
        self.threshold=best_theta
        self.polarity=best_polarity
        self.error=best_error





    #predict labels
    def predict(self,X):
        #require model to be trained
        if self.w is None or self.threshold is None:
            raise ValueError("model not fitted")

        X=np.asarray(X)
        #project points
        z=X@self.w




        #aply polarity and threshold
        if self.polarity==1:
            return np.where(z>=self.threshold,1,-1).astype(int)
        else:
            return np.where(z<=self.threshold,1,-1).astype(int)
