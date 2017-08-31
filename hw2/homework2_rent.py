import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, Imputer


def manual_preprocess_rent():
    nan_dict = {"uf1_1": 8, "uf1_2": 8, "uf1_3": 8, "uf1_4": 8, "uf1_5": 8, "uf1_6": 8, "uf1_7": 8, "uf1_8": 8,
                "uf1_9": 8, "uf1_10": 8, "uf1_11": 8, "uf1_12": 8, "uf1_13": 8, "uf1_14": 8, "uf1_15": 8, "uf1_16": 8,
                "uf1_35": 8, "uf1_17": 8, "uf1_18": 8, "uf1_19": 8, "uf1_20": 8, "uf1_21": 8, "uf1_22": 8, "sc23": 8,
                "sc24": 8, "sc36": 8, "sc37": 8, "sc38": 8, "sc114": 4, "sc117": [3, 8, 9], "sc118": [3, 8, 9],
                "sc120": [5, 8, 9], "sc121": [8, 9],
                "uf5": [9999998, 9999999], "uf6": [9999998, 9999999], "sc127": 9,
                "uf7": [99998, 99999], "sc134": [9998, 9999], "uf7a": [9998, 9999], "uf9": [98, 99],
                "sc140": [8, 9], "sc141": [8, 9], "uf8": [9998, 9999], "sc143": [3, 8, 9], "sc144": [3, 8, 9],
                "uf10": [98, 99], "sc147": [3, 8], "sc173": [3, 8, 9],
                "sc171": [3, 8, 9], "sc154": [8, 9], "sc157": [8, 9], "uf17": 99999,
                "sc185": 8, "sc186": 8, "sc197": [4, 8], "sc198": 8, "sc187": 8,
                "sc188": 8, "sc571": [5, 8], "sc189": [5, 8], "sc190": 8, "sc191": 8, "sc192": 8, "sc193": 8,
                "sc194": 8, "sc196": 8, "sc199": 8, "sc575": [3, 8], "rec15": [10, 11, 12], "rec21": 8,
                "sc27": 98, "rec54": 7, "rec53": 9}

    # only select features that apply to pricing an apartment that is not currently rented
    select_features = ["boro", "uf1_1", "uf1_2", "uf1_3", "uf1_4", "uf1_5", "uf1_6", "uf1_7", "uf1_8", "uf1_9",
                       "uf1_10", "uf1_11", "uf1_12", "uf1_13", "uf1_14", "uf1_15", "uf1_16", "uf1_35", "uf1_17",
                       "uf1_18", "uf1_19", "uf1_20", "uf1_21", "uf1_22", "sc23", "sc24", "sc36", "sc37", "sc38",
                       "sc114", "sc115", "sc116", "sc117", "sc118", "sc120", "sc121", "uf5", "uf6", "sc127", "uf7",
                       "sc134", "uf7a", "uf9", "sc140", "sc141", "uf8", "sc143", "sc144", "uf10", "uf48", "sc147",
                       "uf11", "sc149", "sc173", "sc171", "sc150", "sc151", "sc152", "sc153", "sc154", "sc155", "sc156",
                       "sc157", "sc158", "uf17", "sc185", "sc186", "sc197", "sc198", "sc187", "sc188", "sc571", "sc189",
                       "sc190", "sc191", "sc192", "sc193", "sc194", "sc196", "sc199", "sc575", "uf19", "new_csr",
                       "rec15", "sc26", "uf23", "rec21", "sc27", "rec62", "rec64", "rec54", "rec53", "cd"]

    df_rent = pd.read_csv("https://ndownloader.figshare.com/files/7586326", na_values=nan_dict, usecols=select_features)
    df_rent_select_nona = df_rent[df_rent['uf17'].notnull()]
    # print(df_rent_select_nona.shape)

    # features elimination
    # 1) eliminate cols with less than 1 unique value beside NaN.
    # 2) eliminate cols which only have 2 features and the amount one of them is too less.
    # 3) eliminate cols that have more than 2000 NaN
    filtered_col_list = []
    for col in df_rent_select_nona:
        # print(col, df_rent_select_nona[col].unique())
        if np.sum(~np.isnan(df_rent_select_nona[col].unique())) <= 1:
            # print("1st: ", col)
            filtered_col_list.append(col)
        elif np.sum(~np.isnan(df_rent_select_nona[col].unique())) == 2 \
                and np.min(df_rent_select_nona[col].value_counts()) < 200:
            # print("2nd: ", col)
            filtered_col_list.append(col)
        elif np.sum(np.isnan(df_rent_select_nona[col])) > 5000:
            # print("too many nan", col)
            filtered_col_list.append(col)

    # print(len(filtered_col_list))
    df_rent_select_feature = df_rent_select_nona.ix[:, ~df_rent_select_nona.columns.isin(filtered_col_list)]


    # print(X.shape, y.shape)
    return df_rent_select_feature


def score_rent():
    print(grid.score(X_test, y_test))
    return grid.score(X_test, y_test)


def predict_rent():
    y_pred = grid.predict(X_test)
    return df_process_X, y_test, y_pred


# 1. Manual feature selection.
# After carefully reading the data, I decided first manually filtering some features out.
#     * Features that related to people who currently live there, such as householder's sex, age and race, have been removed.
#     * Features that represents redundant information have been removed as well, such as uf17 and uf17a.
#     * 103/197 features have been excluded during this process.
#
# 2. Missing values.
# I treated 'Not reported', 'Not applicable' and 'Don't know' all as missing value, since I do not think they contain valuable or accurate information.
#
# 3. Features elimination.
# After replacing different missing value with NaN, I observed that some columns are
#     * eliminate cols with less than 1 unique value beside NaN.
#     * eliminate cols which only have 2 features and the amount one of them is too less.
#     * eliminate cols that have more than 5000 NaN. (In total I have roughly 10k rows)
#
# 4. Delete rows that missing y value (uf17).


# Delete rows that missing y value (uf17).


df_process = manual_preprocess_rent()

df_process_X = df_process.ix[:, df_process.columns != 'uf17']
df_process_y = df_process['uf17']

X = df_process_X.as_matrix()
y = df_process_y.as_matrix().reshape(df_process_y.shape[0], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y)


ridge_pipe = make_pipeline(Imputer(strategy="most_frequent"), PolynomialFeatures(), Ridge())
param_grid = {'ridge__alpha': np.logspace(-3, 4, 30)}
grid = GridSearchCV(ridge_pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)

score_rent()
predict_rent()
























