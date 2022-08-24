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

interpreting the final decision tree:

In each node, we have:

The variable (column name) and the threshold for splitting the observations. For example, in the tree's root, we use **ca** to split the observations. All observations with **ca <= 0.5** go to the left and all observations with **ca> 0.5** go to the right.

**gini** is the gini index or score for that node
**samples** tell us how many samples are in that node
**value** tells us how many samples in the node are in each category. In this example, we have two categories, No and Yes, referring to whether or not a
patient has heart disease. The number of patients with No comes first because the categories are in alphabetical order. Thus, in the root, 118 patients have **No** and 104 patients have **Yes**.

**class** tells us whichever category is represented most in the node. In the root, since 118 people have No and only 104 people have Yes, class is set to No.

The leaves are just like the nodes, except that they do not contain a variable and threshold for splitting the observations.

Lastly, the nodes and leaves are colored by the class. In this case No is different shades of orarige-ish and Yes is different shades of blue. The the darker the shade, the lower the gini score, and that tells us how much the node or leaf is skewed towards one class.
