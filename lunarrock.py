# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 16:31:14 2019

@author: 91998
"""

from cv2 import imread, resize, INTER_CUBIC, COLOR_BGR2GRAY, cvtColor
import pandas as pd
from os import listdir, path
from gc import collect

def load_images_from_folder(folder):
    filenames = []
    images = []
    for filename in listdir(folder):
        img = imread(path.join(folder,filename))
        if img is not None:
            img = resize(img, dsize=(65,60), interpolation=INTER_CUBIC)
            #img = cvtColor(img, COLOR_BGR2GRAY)
            images.append(img)
            filenames.append(filename)
    return {"images":images, "filenames":filenames}
cwd = r"D:\Narendra\Lunar Rock classification\DataSet"
print("Getting Train Images")
train_images_large = pd.DataFrame(load_images_from_folder(cwd+"\Train Images\Large"))
print("Got Train Images Large")
collect()

train_images_small = pd.DataFrame(load_images_from_folder(cwd+"\Train Images\Small"))
print("Got Train Images Small")
collect()

print("Read Train labels")
train_data = pd.read_csv(r"D:\Narendra\Lunar Rock classification\DataSet\train.csv", names = ['filenames', 'class'])
print(train_images_large.dtypes, train_data.dtypes, sep=';')
print(train_images_large['images'][0].shape)
train_images_data = pd.concat([train_images_large, train_images_small])
print(train_images_data.head())

print(train_images_data.shape, train_data.shape, sep=';')
train_data_joined = train_data.merge(train_images_data, on='filenames', how = 'inner')

print(train_data_joined.shape)
print(train_data.iloc[0,:])
print(train_data_joined.iloc[0,:])

import matplotlib.pyplot as plt
plt.imshow(train_data_joined['images'][0])

import numpy as np
#train_data_joined = np.random.shuffle(train_data_joined)
train_data_joined['Large'] = np.where(train_data_joined['class']=='Large',1,0)
train_data_joined['Small'] = np.where(train_data_joined['class']=='Small',1,0)
print(train_data_joined.iloc[0,:])
print(train_data_joined.loc[0:5,['Large', 'Small']])
x_train=[]
y_train=[]
for i in range(train_data_joined.shape[0]):
    x_train.append(train_data_joined['images'][i])
    y_train.append(train_data_joined.loc[i,['Large', 'Small']])
x_train = (np.array(x_train))/255
y_train = np.array(y_train)
#x_train, y_train = train_data_joined['images'], train_data_joined['class']
# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
print(x_train.shape, input_shape,sep=';')
#x_train = x_train.reshape(x_train.shape[0], 60, 65, 1)
#plt.imshow(x_train[0])

from sklearn.model_selection import train_test_split
train_x, x_val, train_y, y_val = train_test_split(x_train, y_train, test_size = 0.35, stratify = y_train, shuffle=True)
print(np.unique(y_train, return_counts=True))
print(np.unique(y_val, return_counts=True))

from keras.models import load_model
from keras import applications
from keras.models import Sequential,Model, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (60,65,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(2, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)
from keras.optimizers import Adam
adam = Adam(lr=0.0001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
training_history = model.fit(train_x, train_y, validation_data = (x_val, y_val), epochs = 20, batch_size = 64, shuffle=True)

print(model.summary())

#test_data = pd.read_csv(r"D:\Narendra\Lunar Rock classification\DataSet\train.csv")
test_images = pd.DataFrame(load_images_from_folder(cwd+"\Test Images"))
print(test_images.shape, test_images.columns)
x_test = []
for i in range(test_images.shape[0]):
    x_test.append(test_images['images'][i])
x_test = (np.array(x_test))/255
predictions = np.argmax(model.predict(x_test), axis = 1)
print(predictions[0:10])
print(np.unique(predictions, return_counts = True))

from sklearn.metrics import confusion_matrix, f1_score
val_predictions = np.argmax(model.predict(x_val), axis = 1)
y_val = np.argmax(y_val, axis = 1)
print(val_predictions.shape, y_val.shape)
print(confusion_matrix(val_predictions, y_val))
print(f1_score(val_predictions, y_val))
pred_results = pd.concat([test_images['filenames'], pd.Series(predictions, name='class', dtype='int32')], axis = 1)
print(pred_results.shape)

test_csv = train_data = pd.read_csv(r"D:\Narendra\Lunar Rock classification\DataSet\test.csv")
print(test_csv.columns, pred_results.columns)
pred_results.rename(columns={pred_results.columns[0]:test_csv.columns[0], 
                             pred_results.columns[1]:test_csv.columns[1]}, inplace=True)
result_csv = test_csv.merge(pred_results, on = pred_results.columns[0], how='left')
print(result_csv.shape)
print(result_csv.columns)
print(result_csv.isna().sum())
result_csv.drop(columns = ['Class_x'], inplace=True)
print(result_csv.shape)
print(result_csv.columns)
print(result_csv.isna().sum())
result_csv.rename(columns={'Class':'Class_y'})
y_val = np.argmax(y_val, axis = 1)
print(val_predictions.shape, y_val.shape)
print(val_predictions[:10], y_val[:10])
# 0 : Large ; 1 : Small
result_csv['Class'] = np.where(result_csv['Class_y']==0, 'Large', 'Small')
result_csv.drop(columns = ['Class_y'], inplace=True)
print(result_csv.head())
result_csv.to_csv(r"D:\Narendra\Lunar Rock classification\DataSet\submission.csv", index=False)

#Saving the model to disk and loading again

# serialize model to JSON
model_json = model.to_json()
with open(r"D:\Narendra\Lunar Rock classification\DataSet\model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(r"D:\Narendra\Lunar Rock classification\DataSet\model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
from keras.models import model_from_json

json_file = open(r"D:\Narendra\Lunar Rock classification\DataSet\model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(r"D:\Narendra\Lunar Rock classification\DataSet\model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(x_val, y_val, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
