#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.datasets import fetch_mldata
# from sklearn.metrics import confusion_matrix, classification_report
# import matplotlib.cm as cm

# from sklearn.model_selection import cross_val_score

#%matplotlib inline

class KNNClassifier:
    
    def __init__(self,X,y,K):
        #initialize all attributes
        self.X = np.array(X) #2d-array
        self.y = np.array(y)
        self.n_observations = len(y) #number of labels
        self.classes = np.unique(self.y) 
        self.K = K
        
    def predict(self,X):
        X_value = np.array(X)
        predictions = []
        for i in range(X_value.shape[0]):
            distances = np.sqrt(np.sum((X_value[i,:]-self.X)**2,axis=1))
            idx = np.argsort(distances)[:self.K]
            knn_labels = self.y[idx]
            knn_distances = distances[idx]
            best_dist = 0
            best_class = 0
            best_count = 0
        
            for label in self.classes:
            #for j in range(len(self.classes)): #for label in slef.classes
                temp_count = np.sum(label==knn_labels)
                #temp_count = np.sum(label == knn_labels)
                temp_dist = np.sum(knn_distances[knn_labels==label])
            
                if(temp_count > best_count):
                    best_dist = np.sum(knn_distances[knn_labels==label])
                    best_class = label
                    best_count = np.sum(knn_labels==label)
                elif ((temp_count == best_count) & (temp_dist < best_dist)):
                    best_dist = np.sum(knn_distances[knn_labels==label])
                    best_class = label
                    best_count = np.sum(knn_labels==label)
                
            predictions.append(best_class)
        
        predictions = np.array(predictions)
        
        return predictions
        
        
        
    def score(self,X,y):
        #correct x and y to numpy array
        X_value = np.array(X)
        y_label = np.array(y)
        number_y = len(y_label)
        #calsulate y_predicted
        y_predicted = self.predict(X_value)     
        #calculate accuracy
        accuracy = (np.sum(y_label==y_predicted))/(number_y)      
        return accuracy
        
    def confusion_matrix(self,X,y): #(y-pred,y_true)     
        array_X = np.array(X)
        array_y = np.array(y)
        cm = []
    
        predict_label = self.predict(array_X)
        
        for i in range(len(self.classes)):#row actual
            new_row = []
            for j in range(len(self.classes)):           
                count = np.sum((predict_label == self.classes[j])) & (array_y==self.classes[i])
                new_row.append(count)
            cm.append(new_row)
            
        return cm
 
    
# #test code

# X = pd.read_csv('Text_vector_all.csv')
# X = X.drop(columns = ["Unnamed: 0"])

# Y = pd.read_csv('ChannelVideo.csv')
# Y_train = Y["like"] / (Y["dislike"] + Y["like"])
# meanS = Y_train.mean(skipna = True)
# Y_label = []
# for s in Y_train:
#     if s > meanS:
#         Y_label.append(1) #popular
#     else:
#         Y_label.append(2) #unpopular
        
# X_val, X_test, y_val, y_test = train_test_split(X, Y_label, test_size = 0.25, random_state=10)

# knn = KNeighborsClassifier(n_neighbors=11)
# scores = cross_val_score(knn, X_val, y_val, cv=10, scoring='accuracy')
# print(scores)

# # print(scores.mean())

# k_range = range(1, 31,2)

# # list of scores from k_range
# k_scores = []

# # 1. we will loop through reasonable values of k
# for k in k_range:
#     # 2. run KNeighborsClassifier with k neighbours
#     knn = KNeighborsClassifier(n_neighbors=k)
#     # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
#     scores = cross_val_score(knn, X_val, y_val, cv=10, scoring='accuracy', n_jobs = 10)
#     # 4. append mean of scores for k neighbors to k_scores list
#     k_scores.append(scores.mean())
# print(k_scores)
# k_scores = [0.687503363566545, 0.7092659552986913, 0.7198328748997473, 0.724786530731332, 0.7267022372460317, 0.7310280454941729, 0.7297403610104525, 0.7290135362565215, 0.728518225188218, 0.7295419596476684, 0.7306647694043444, 0.7281882140628264, 0.72832011820565, 0.7278249706819111, 0.7271312037359683]

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
# plt.plot(k_range, k_scores)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-Validated Accuracy')
# plt.show()



# #test code go below
# fruit_mod = KNNClassifier(X_val,y_val,10)
# print(fruit_mod.confusion_matrix(X_val,y_val))

# print("Test Set Performance:") 
# #print(fruit_mod.predict_proba(X_test)) 
# print(fruit_mod.predict(X_test)) 
# print(fruit_mod.score(X_test,y_test))

# val_acc = []

# for K in range(1,11):
#     mod = KNeighborsClassifier(n_neighbors=K, algorithm='brute')
#     mod.fit(X_val, y_val)
#     temp = mod.score(X_test, y_test)
#     val_acc.append(temp)
    
# plt.close()
# plt.plot(range(1,11), val_acc)
# plt.ylim([0.9,1])
# plt.xlim([0,10])
# plt.xlabel('K')
# plt.ylabel('Accuracy')
# plt.title('Accuracy on Validation Set')
# plt.show()

