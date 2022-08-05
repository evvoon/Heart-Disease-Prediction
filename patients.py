import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

df = pd.read_csv('processed.cleveland.data',header=None)

df.head()

df.columns = ['age','sex','cp','restbp','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','hd']

df.head()

df.dtypes

df['ca'].unique() # print out UNIQUE values

df['thal'].unique()

len(df.loc[(df['ca'] == '?')
    |
    (df['thal'] == '?')]) # to figure out how many rows has ?

df.loc[(df['ca'] == '?')
    |
    (df['thal'] == '?')]

len(df) # number of rows in full dataset

# use loc to select all rows that do NOT contain missing values 
# store them in new dataframe called "df_no_missing"
df_no_missing = df.loc[(df['ca'] != '?')
    &
    (df['thal'] != '?')]

len(df_no_missing) # 303 - 6 = 297

df_no_missing['ca'].unique()

df_no_missing['thal'].unique()

# first step in classification tree : split data into columns used
# to make classification and column that we want to predict 
# in this case hd column (heart disease) is what we want to predict
# NOW, make new copy of columns used to make prediction
X = df_no_missing.drop('hd', axis=1).copy()
X.head()

# make new copy of column with hd
y = df_no_missing['hd'].copy()
y.head()

# one-hot encoding : in order to use categorical data with scikit learn decision trees 
 # we have to convert categorical data into columns of binary values 
 # Categorical Data is the data that generally takes a limited number of possible values. 
 # Also, the data in the category need not be numerical, it can be textual in nature.
 X.dtypes

X['cp'].unique()

# so cp contains values 1,2,3,4 we will convert it using one hot coding 
# into series of columns that contain only 0s and 1s
# here we're using getdummies from pandas u can also use 
# ColumnTransformer() method from Scikit-learn library
pd.get_dummies(X,columns=['cp']).head()
#now cp column got converted into 4 rows - each for 1 value - 1,2,3,4

#now that we have seen how get dummies works we will use it on all the 4 columns
# that have such values and save it
X_encoded = pd.get_dummies(X,columns=['cp',
                                     'restecg',
                                     'slope',                     
                                     'thal'])
X_encoded.head()

# since sex, fbs, exang only have two categories and is already 0s and 1s no change is required 
# column w more than two categories (like cp) gets seperated into multiple columns of 0s and 1s
# now if u want, to check if a column is unique
X_encoded['sex'].unique()

# predicting value - heart disease in this has 0 - no hd and 1-4 levels of hd
y.unique()

# to keep it simple here we convert all > 0 nos to 1
y_not_zero_index = y > 0 # get index for each non-zero value in y
y[y_not_zero_index] = 1 # set each non zero value in y to 1 
y.unique() # verify that y only contains 0 and 1 
# we are done formatting the data for the classification tree

# build a preliminary classification tree (tree not optimized)
# split data into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42)

# create a decision tree and fit it into training data
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(X_train,y_train)

# decision tree will be huge for this
plt.figure(figsize=(15,7.5))
plot_tree(clf_dt,
          filled=True,
          rounded=True,
          class_names=['No HD','Yes HD'],
          feature_names=X_encoded.columns);

# plot_confusion_matrix() will run test data down the tree and draw
# a confusion matrix
plot_confusion_matrix(clf_dt,X_test,y_test,display_labels=["Does not have HD","Has HD"])
# now looking at this confusion matrix we see that
# 74% no hd and 79% hd has been correctly classified 
# to do better than this we can prune the data set
# cuz overfitting could be holding back this prediction

# cost complexity pruning (ccp)
path = clf_dt.cost_complexity_pruning_path(X_train,y_train) # determine values for alpha
ccp_alphas = path.ccp_alphas # extract different values for alpha
ccp_alphas = ccp_alphas[:-1] # exclude the maximum value for alpha(since max value would prune all leaves and just give root instead of leaves)

clf_dts = [] # create an array that we will put decision trees into

# now create one decision tree per vslue of alpha and store it in the array 
for ccp_alpha in ccp_alphas: # for i in ccp_alphas
  clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
  clf_dt.fit(X_train,y_train)
  clf_dts.append(clf_dt)

# now lets graph the accuracy of the trees using the training dataset and the testing dataset 
# as a function of alpha
train_scores = [clf_dt.score(X_train,y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test,y_test) for clf_dt in clf_dts]

fig, ax= plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas,train_scores,marker='o', label="train", drawstyle = "steps-post")
ax.plot(ccp_alphas,test_scores,marker='o', label="test", drawstyle = "steps-post")
ax.legend()
plt.show()
# here we see as the trees get smaller our testing accuracy improves
# which proves that pruning gives a better perfomance with testing data
# here we see that accuracy for testing dataset reaches max when alpha is at 0.016 after which accuracy for training dataset drops off

# now lets do cross validation to find best alpha 
clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=0.016) # create the tree with ccp_alpha = 0.016

# now use 5-fold cross validation create 5 different training and testing datasets that
# are then used to train and test the tree
# note : we use 5 - fold because we dont have tons of data
scores = cross_val_score(clf_dt,X_train,y_train,cv=5)
df = pd.DataFrame(data={'tree' : range(5), 'accuracy': scores})

df.plot(x='tree', y='accuracy', marker = 'o', linestyle = '--')
#here it shows ...

# create an array to store the results of each fold during cross validation
alpha_loop_values = []

# for each candidate value for alpha we will run 5-fold cross validation
# then we will store the mean and standard deviation of the scores(the accuracy) 
# for each call to cross_val_score in alpha_loop_values
for ccp_alpha in ccp_alphas:
  clf_dt = DecisionTreeClassifier(random_state=0 , ccp_alpha=ccp_alpha)
  scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
  alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

# now we can draw a graph of the means and standard deviation of the scores
# for each candidate value for alpha
alpha_results = pd.DataFrame(alpha_loop_values,
                             columns=['alpha','mean_accuracy','std']
                             )
alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   yerr='std',
                   marker='o',
                   linestyle = '--')

# using cross validation we can see from the above graph that instead of setting ccp_alpha=0.016 we need to set it to something closer to 0.014
# we can find the exact value with: 
alpha_results[(alpha_results['alpha'] > 0.014)
              &
              (alpha_results['alpha'] < 0.015)]

# now lets store ideal value in new variable so that we can use it to build the best tree
ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.014)
                                &
                                (alpha_results['alpha'] < 0.015)]['alpha']
ideal_ccp_alpha

# we need to convert ideal_ccp_alpha to float cuz python thinks that it is a series ( as 20 index is shown)
ideal_ccp_alpha=float(ideal_ccp_alpha)
ideal_ccp_alpha

# building , evaluating , drawing and interpreting final classification tree
# now we can build final ctree by setting ccp_alpha = ideal_ccp_alpha
clf_dt_pruned = DecisionTreeClassifier(random_state=42,
                                       ccp_alpha=ideal_ccp_alpha)
clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train)

# now lets do another confusion matrix to see if pruned tree does better 
plot_confusion_matrix(clf_dt_pruned,X_test,y_test,display_labels=["Does not have HD","Has HD"])

# yas its done better
# 81 % without hd and 85% with hd has been classified
# now draw pruned tree
plt.figure(figsize=(15,7.5))
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["No HD", 'Yes HD'],
          feature_names = X_encoded.columns
          );
# now this SMOL DECISION TREE DOES BETTERRR THAN THAT HOOGE ONE cuz that was overfitting the data
# also the COLOURS of the boxes are acc to whoever has majority so all oranges have majority of no hd and bluish has majority hd
# the darker the colour , the lower the gini impurity