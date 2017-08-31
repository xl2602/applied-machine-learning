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
    1. Feature selection.
    
    2. Imputation.
   
    3. Build linear models.
    
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
