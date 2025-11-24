#ce4041 adaboost project
#name: Kanvar Murray
#student id: 22374698
#date: 21/11/25

import numpy as np





class AdaBoost:

    #set parameters and controls
    def __init__(self,base_learner_class,n_estimators=50,
                 target_train_accuracy=None,avoid_perfect=False):

        #store weak learner class
        self.base_learner_class=base_learner_class
        #max weak learners
        self.n_estimators=n_estimators
        #target train accuracy under 100 percent to avoid that overfit boi
        self.target_train_accuracy=target_train_accuracy
        #stop at perfect if that happens so we can step back a lil
        self.avoid_perfect=avoid_perfect

        #learners and weights
        self.learners=[]
        self.alphas=[]







    #training the BOOSHT
    def fit(self,X,y):
        X=np.asarray(X)
        y=np.asarray(y).astype(int)
        n_samples=X.shape[0]

        #start uniform weights
        w=np.ones(n_samples)/n_samples

        #reset stored learners
        self.learners=[]
        self.alphas=[]

        #running strong score
        F=np.zeros(n_samples)

        #loop over boosting rounds
        for m in range(self.n_estimators):

            #train weak learner
            learner=self.base_learner_class()
            learner.fit(X,y,sample_weight=w)
            y_pred=learner.predict(X)

            #weighted error
            misclassified=(y_pred!=y)
            eps=np.sum(w[misclassified])

            #handle perfect weak learner
            if eps<=0:
                alpha=0.5*np.log((1-1e-10)/(1e-10))

                #add tentatively
                self.learners.append(learner)
                self.alphas.append(alpha)
                F+=alpha*y_pred

                #check strong accuracy
                y_strong=np.sign(F)
                y_strong[y_strong==0]=1
                acc=np.mean(y_strong==y)

                #remove if perfect not allowed
                if self.avoid_perfect and acc==1.0:
                    self.learners.pop()
                    self.alphas.pop()

                break

            #stop if too weak
            if eps>=0.5:
                break

            #compute alpha weight
            alpha=0.5*np.log((1-eps)/eps)

            #update sample weights
            w=w*np.exp(-alpha*y*y_pred)
            w/=np.sum(w)

            #store weak learner
            self.learners.append(learner)
            self.alphas.append(alpha)

            #update strong score
            F+=alpha*y_pred

            #check train accuracy
            y_strong=np.sign(F)
            y_strong[y_strong==0]=1
            acc=np.mean(y_strong==y)

            #stop if target accuracy reached but not overfitted so ideal fr fr
            if self.target_train_accuracy is not None and acc>=self.target_train_accuracy and acc<1.0:
                break

            #stop early
            if self.avoid_perfect and acc==1.0:
                self.learners.pop()
                self.alphas.pop()
                break

        return self



    #predict labels
    def predict(self,X,T=None):
        #ensure model trained
        if not self.learners:
            raise ValueError("model not fitted")

        X=np.asarray(X)

        #limit T to number of learners
        if T is None or T>len(self.learners):
            T=len(self.learners)


        #accumulate weighted votes (its like america)
        F=np.zeros(X.shape[0])
        for t in range(T):
            F+=self.alphas[t]*self.learners[t].predict(X)

        #convert score to sign label
        y_pred=np.sign(F)
        y_pred[y_pred==0]=1


        return y_pred.astype(int)
