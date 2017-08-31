# homework2 - xl2602

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



Travis CI link: https://travis-ci.com/xl2602/homework-ii-xl2602.svg?token=bXJ4duUptPFzVS8PcSf2&branch=master





