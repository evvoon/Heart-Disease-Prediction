# Predicting patients with heart disease using ML (decision trees)

Here I have used cost complexity pruning and scikit learn to build a decision tree that predicts whether a person has heart disease or not 

Classification trees is useful to figure out how exactly the predictions were made 

Steps followed: 
1) imported data 
2) identified missing data and handled it
3) formatted data for decision tree by splitting data into dependent and independent variables
4) used one hot encoding to convert categorical data into columns of binary values
5) built a preliminary decision tree and then optimized it using cost complexity pruning (visualizing alpha, using cross validation to find best value of      alpha)
6) interpreted and evaluated final classification tree
