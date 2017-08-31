# Homework applied-machine-learning

## hw1
- Task 1: Git; 
- Task 2: Continuous integration (TravisCI); 
- Task 3: Documentation (Sphinx); 
- Task 4: Data Visualization and Analysis (matplotlib)


## hw2 - Estimate the market rate for some apartments in NYC 
- Data: https://www.census.gov/housing/nychvs/data/2014/uf_14_occ_web_b.txt

- Task: Create and validate a machine learning approach to predict the monthly rent of an apartment using linear models. 

- Steps:
    1. Manual feature selection.
    After carefully reading the data, I decided first manually filtering some features out.
        * Features that related to people who currently live there, such as householder's sex, age and race, have been removed.
        * Features that represents redundant information have been removed as well, such as uf17 and uf17a.
        * 103/197 features have been excluded during this process.
    
    2. Missing values.
    I treated 'Not reported', 'Not applicable' and 'Don't know' all as missing value, since I do not think they contain valuable or accurate information.
    
    3. Features elimination.
    After replacing different missing value with NaN, I decided to:
        * eliminate cols with less than 1 unique value beside NaN.
        * eliminate cols which only have 2 features and the amount one of them is too less.
        * eliminate cols that have more than 5000 NaN. (In total I have roughly 10k rows)
    
    The intuition behind this elimination is the features with too many missing values are probably very hard to gather in real life. 
    
    4. Delete rows that missing target value (uf17).
    I think it is a bad idea to impute a target value, so all rows without corresponding target value have been removed.
    
    5. Split and make pipeline. 
    I have tried different models and imputation techniques, and ended up with a simple model.
    
    My model first imputed with most frequent value, than polynomialized all features finally apply ridge on the training data. It took less time among all the attempts I did and perform a 0.49 - 0.53 accuracy.
    
    I tried alpha in this log space: np.logspace(-3, 4, 30), and grid search to find the best.
    
## hw3 - Promotional Call Prediction
- Task: A banking institution ran a direct marketing campaign based on phone calls. Often, more than one contact to the same client was required, in order to assess if the product (bank term deposit) would be subscribed or not. Your task is to predict whether someone will subscribe to the term deposit or not based on the given information.

- Steps:
    1. Data Cleaning, feature engineering
    2. Model Set1: SVM, Logistic Regression, etc.
    3. Model Set2: Tree-based models, like decision tree, random forest, gradient boosted trees, etc.
    4. Model Ensemble
    5. Resampling Techniques 

## hw4 - Text classification
- Data: https://data.boston.gov/dataset/vision-zero-entry

- Task: Text classification on a dataset of complaints about traffic conditions to the city of Boston. 

- Steps:
    1. Data Cleaning
          - Load the data, visualize the class distribution. Clean up the target labels. Consolidate some arbitrarily split categories. 
    2. Model 1
          - A baseline multi-class classification model using a bag-of-word approach. Report macro f1-score and visualize the confusion matrix. 
    3. Model 2
          - Improve the model using more complex text features, including n-grams, character n-grams and possibly domain-specific features.
    4. Visualize Results
          - Visualize results of the tuned model (classification results, confusion matrix, important features, example mistakes).
    5. Clustering
          - Apply LDA, NMF and K-Means to the whole dataset. Find clusters or topics that match well with some of the ground truth labels. Use ARI to compare the methods and visualize topics and clusters.
    6. Model 3
          - Improve the class definition for REQUESTTYPE by using the results of the clustering and results of the previous classification model. Re-assign labels using either the results of clustering or using keywords that found during data exploration.
          - The data has a large “other” category. Apply the topic modeling and clustering techniques to this subset of the data to find possible splits of this class.
          Report accuracy using macro average f1 score (should be above .53) 
    7. Word2vec
    
    
## hw5 - Neural Network with Keras

- Task 1: 
    Run a multilayer perceptron (feed forward neural network) with two hidden layers and rectified linear nonlinearities on the iris dataset using the keras Sequential interface. 

- Task 2: 
    Train a multilayer perceptron on the MNIST dataset. Compare a “vanilla” model with a model Qusing drop-out. Visualize the learning curves.

- Task 3:
    Train a convolutional neural network on the SVHN dataset in format 2 (single digit classification)
    
- Task 4:
    Load the weights of a pre-trained convolutional neural network, for example AlexNet or VGG, and use it as feature extraction method to train a linear model or MLP on the pets dataset. The pets dataset can be found here: http://www.robots.ox.ac.uk/~vgg/data/pets/ (37 class classification task).
