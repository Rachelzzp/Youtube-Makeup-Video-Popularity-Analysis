import numpy as np
from scipy.optimize import minimize
import pandas as pd
# from sklearn.model_selection import train_test_split
#define a single class linearRegression
class LogisticRegression:
    def __init__(self,X,y):
        #initialize all attributes
        self.X = np.array(X) #2d-array
        self.labels = np.array(y)
        self.n_observations = len(y) #number of labels
        self.classes = np.unique(self.labels) #calss 1 and class0
        
        def find_neg_loglik(beta):
            z = beta[0] + np.sum(beta[1:] * self.X, axis=1)
            p = 1 / (1 + np.exp(-z))
            #pi => class1
            pi = np.where(self.labels==self.classes[1], p, 1-p)
            #loglik
            loglik = np.sum(np.log(pi)) 
            #neg loglik
            neg_loglik = -loglik
            #return negative score
            return neg_loglik
        
        #inital "guess"  # self.coefficients
        np.seterr(all='ignore')
        beta_guess = np.zeros(self.X.shape[1] + 1) 
        min_results = minimize(find_neg_loglik, beta_guess) 
        self.coefficients = min_results.x
        np.seterr(all='warn')
        
        #calsulate y_predicted
        self.y_predicted = self.predict(self.X)
        
        #calculate accuracy
        self.accuracy = (np.sum(self.labels==self.y_predicted))/(self.n_observations)

        #calculate loglik     
        self.loglik =-find_neg_loglik(self.coefficients)
        
        
        
    def predict_proba(self,X):
        #set class
        new_X = np.array(X)
        z = self.coefficients[0] + np.sum(self.coefficients[1:] * new_X,axis=1)
        p = 1 / (1 + np.exp(-z))
        return p #return predict_prob
        
        
    def predict(self,X,t=0.5):
        the_X = np.array(X)
        prob = self.predict_proba(the_X)
        #pi => class1
        predict = np.where(prob > t, self.classes[1], self.classes[0]) 
        #print out labels
        return predict
    
        
    def score(self,X,y,t=0.5):
        #correct x and y to numpy array
        X_value = np.array(X)
        y_label = np.array(y)
        number_y = len(y_label)
         #calsulate y_predicted
        y_predicted = self.predict(X_value)     
        #calculate accuracy
        accuracy = (np.sum(y_label==y_predicted))/(number_y)      
        return accuracy
    
    def confusion_matrix(self,X,y,t=0.5):
        #TP FP TN FN
        array_X = np.array(X)
        array_y = np.array(y)
    
        predict_label = self.predict(array_X,t)
                 
        tp = np.sum((predict_label == self.classes[1] ) &  (array_y == self.classes[1])) 
        fp = np.sum((predict_label == self.classes[1] ) &  (array_y == self.classes[0]))
        tn = np.sum((predict_label == self.classes[0] ) &  (array_y == self.classes[0]))
        fn = np.sum((predict_label == self.classes[0] ) &  (array_y == self.classes[1]))
        
        cm = pd.DataFrame([[tn,fp],[fn,tp]])
        cm.columns = ['Pred_0','Pred_1']
        cm.index = ['True_0','True_1']
        print(cm)
    
    def summary(self):
        print('+-----------------------------+')
        print('| Logistic Regression Summary |')
        print('+-----------------------------+')
        print('Number of training observations:' ,self.n_observations,'\n')
        print('Coefficient Estimates:',self.coefficients,'\n')
        print('Log-Likelihood:',self.loglik,'\n')
        print('Accuracy:',self.accuracy,'\n')
        print('Class 0:',self.classes[0],'\n')
        print('Class 1:',self.classes[1],'\n')
    
#print matrix
#print 
# X = pd.read_csv('Text_vector_all.csv')
# X = X.drop(columns = ["Unnamed: 0"])

# Y = pd.read_csv('ChannelVideo.csv')
# Y_train = Y["like"] / (Y["dislike"] + Y["like"])
# meanS = Y_train.mean(skipna = True)
# Y_label = []
# for s in Y_train:
#     if s > meanS:
#         Y_label.append("popular")
#     else:
#         Y_label.append("unpopular")
# #print(Y_label)
        
# X_val, X_test, y_val, y_test = train_test_split(X, Y_label, test_size = 0.25, random_state=10)

# #test code go below
# fruit_mod = LogisticRegression(X_val,y_val)
# fruit_mod.summary()
# fruit_mod.confusion_matrix(X_val,y_val)

# print("Test Set Performance:") 
# print(fruit_mod.predict_proba(X_test)) 
# print(fruit_mod.predict(X_test)) 
# print(fruit_mod.score(X_test,y_test))
