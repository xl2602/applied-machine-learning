# homework-iv-starter

- Data: https://data.boston.gov/dataset/vision-zero-entry

- Task: Text classification on a dataset of complaints about traffic conditions to the city of Boston. 

- Steps:
    1.  Data Cleaning
          - Load the data, visualize the class distribution. Clean up the target labels. Consolidate some arbitrarily split categories. 
    2.  Model 1
          - A baseline multi-class classification model using a bag-of-word approach. Report macro f1-score and visualize the confusion matrix. 
    3)  Model 2
          - Improve the model using more complex text features, including n-grams, character n-grams and possibly domain-specific features.
    4)  Visualize Results
          - Visualize results of the tuned model (classification results, confusion matrix, important features, example mistakes).
    5)  Clustering
          - Apply LDA, NMF and K-Means to the whole dataset. Find clusters or topics that match well with some of the ground truth labels. Use ARI to compare the methods and visualize topics and clusters.
    6)  Model 3
          - Improve the class definition for REQUESTTYPE by using the results of the clustering and results of the previous classification model. Re-assign labels using either the results of clustering or using keywords that found during data exploration.
          - The data has a large “other” category. Apply the topic modeling and clustering techniques to this subset of the data to find possible splits of this class.
          Report accuracy using macro average f1 score (should be above .53) 
    7)  Word2vec