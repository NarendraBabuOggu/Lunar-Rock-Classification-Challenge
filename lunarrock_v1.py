# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 16:31:14 2019

@author: 91998
"""

from cv2 import imread, resize, INTER_CUBIC
import pandas as pd
from os import listdir, path
from gc import collect

from sklearn.metrics import f1_score, recall_score, precision_score
from keras.callbacks import Callback

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_pred = self.model.predict(X_val)

        y_pred_cat = y_pred.argmax(axis=1)

        _val_f1 = f1_score(np.argmax(y_val, axis = 1), y_pred_cat, average='macro')
        _val_recall = recall_score(np.argmax(y_val, axis = 1), y_pred_cat, average='macro')
        _val_precision = precision_score(np.argmax(y_val, axis = 1), y_pred_cat, average='macro')

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)

        print((f"val_f1: {_val_f1:.4f}"
               f" — val_precision: {_val_precision:.4f}"
               f" — val_recall: {_val_recall:.4f}"))

        return

f1_metrics = Metrics()

def load_images_from_folder(folder):
    filenames = []
    images = []
    for filename in listdir(folder):
        img = imread(path.join(folder,filename))
        if img is not None:
            img = resize(img, dsize=(64,64), interpolation=INTER_CUBIC)
            #img = cvtColor(img, COLOR_BGR2GRAY)
            images.append(img)
            filenames.append(filename)
    return {"images":images, "filenames":filenames}
cwd = r"D:\Narendra\Lunar Rock classification\DataSet"
print("Getting Train Data")
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
train_data = train_data.merge(train_images_data, on='filenames', how = 'inner')



import numpy as np
#train_data_joined = np.random.shuffle(train_data_joined)
train_data['Large'] = np.where(train_data['class']=='Large',1,0)
train_data['Small'] = np.where(train_data['class']=='Small',1,0)

#Reducing Data Usage
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            print("min for this col: ",mn)
            print("max for this col: ",mx)
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist

train_data, NAlist = reduce_mem_usage(train_data)

x_train=[]
y_train=[]
for i in range(train_data.shape[0]):
    x_train.append(train_data['images'][i])
    y_train.append(train_data.loc[i,['Large', 'Small']])
x_train = (np.array(x_train))/255
y_train = np.array(y_train)

del train_images_small, train_images_large, train_images_data, train_data, NAlist
collect()

from keras import applications 
from keras.models import Model
from keras.layers import Dense, Dropout, Input

base_model = applications.resnet.ResNet101(include_top=False, weights='imagenet', input_tensor=Input(shape=(64,64,3)), pooling='avg', classes=1000)

x = base_model.output
#x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(2, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
lr_scheduler = LearningRateScheduler(lambda x: 0.001 * 0.9 ** x)
adam = Adam(lr=0.001)
model.compile(optimizer= adam, loss='binary_crossentropy', metrics=['accuracy'])
training_history = model.fit(x_train, y_train, validation_split=0.3, 
                             callbacks = [f1_metrics, lr_scheduler], epochs = 1, batch_size = 64, shuffle=True)

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

pred_results = pd.concat([test_images['filenames'], pd.Series(predictions, name='class', dtype='int32')], axis = 1)

test_csv = train_data = pd.read_csv(r"D:\Narendra\Lunar Rock classification\DataSet\test.csv")
pred_results.rename(columns={pred_results.columns[0]:test_csv.columns[0], 
                             pred_results.columns[1]:test_csv.columns[1]}, inplace=True)
result_csv = test_csv.merge(pred_results, on = pred_results.columns[0], how='left')
result_csv.drop(columns = ['Class_x'], inplace=True)
result_csv.rename(columns={'Class':'Class_y'})
result_csv['Class'] = np.where(result_csv['Class_y']==0, 'Large', 'Small')
result_csv.drop(columns = ['Class_y'], inplace=True)
result_csv.to_csv(r"D:\Narendra\Lunar Rock classification\DataSet\submission_v2.csv", index=False)

#Saving the model to disk and loading again

# serialize model to JSON
model_json = model.to_json()
with open(r"D:\Narendra\Lunar Rock classification\DataSet\model_v2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(r"D:\Narendra\Lunar Rock classification\DataSet\model_v2.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
from keras.models import model_from_json

json_file = open(r"D:\Narendra\Lunar Rock classification\DataSet\model_v2.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(r"D:\Narendra\Lunar Rock classification\DataSet\model_v2.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
