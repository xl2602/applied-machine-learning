import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from keras import backend as K


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import os

from keras import applications
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

import sklearn

if sklearn.__version__.startswith('0.17'):
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegressionCV


# build the VGG16 network
print('Loading VGG16...')
model = applications.VGG16(include_top=False, weights='imagenet')

# get all data than split later
print('Reading list.txt...')
data = pd.read_csv(os.path.join("annotations", "list.txt"), skiprows=6, sep=" ",
                   names=['image_name', 'class_id', 'species', 'breed_id'])

img_name_list = data.image_name.tolist()

print('Loading image...')
img_list = [image.load_img(os.path.join("images", name+".jpg"), target_size=(224, 224))
            for name in img_name_list]

print('Converting image to array...')
X = np.array([image.img_to_array(img) for img in img_list])

y = data.class_id.values

print('Pre-processing features...')
feature_file = 'task4_features.txt'

if not os.path.isfile(feature_file):
    X_pre = preprocess_input(X)
    features = model.predict(X_pre)
    features_ = features.reshape(7349, -1)
    np.savetxt(feature_file, features_)
else:
    features_ = np.loadtxt(feature_file)

print('Splitting...')
X_train, X_test, y_train, y_test = train_test_split(features_, y, stratify=y, random_state=6)


print('Logistic Regression...')
lr = LogisticRegressionCV(verbose=1).fit(X_train, y_train)

print('Params: ', lr.get_params())
print('Train Score', lr.score(X_train, y_train))
print('Test Score', lr.score(X_test, y_test))


