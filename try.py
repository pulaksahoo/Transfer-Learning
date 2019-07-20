from keras.models import Sequential
            #       from kerfrom keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.applications import imagenet_utils
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
import cv2
import random
from keras.utils import to_categorical
from keras.applications import ResNet50
from keras.applications.inception_v3 import preprocess_input
import keras
from keras.models import Model

DATADIR = 'hello-world/dataset'

IMG_SIZE=224

training_data = []
inputShape=(224,224)
preprocess = imagenet_utils.preprocess_input
k_folds=['k1','k2','k3','k4']
for folds in k_folds:
    test_list=[folds]
    train_list=[]
    for rem in k_folds:
        if(rem!=folds):
            train_list.append(rem)
    training_data=[]

    for i in train_list: 
        path = os.path.join(DATADIR,i)  # create path to dogs and cats
        CATEGORIES = sorted(os.listdir(path))

        for category in CATEGORIES:
            class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
            p=os.path.join(path,category)
            for img in os.listdir(p):
                try:
                    img_array = cv2.imread(os.path.join(p,img))  # convert to array
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                    # new_array=np.array(new_array).reshape((224,224,3))
                    #     new_array=preprocess_input(new_array)
                    training_data.append([np.array(new_array), class_num])  # add this to our training_data
                except Exception as e:  # in the interest in keeping the output clean...
                    pass
    testing_data = []
    for i in test_list: 
        path = os.path.join(DATADIR,i)  
        CATEGORIES = sorted(os.listdir(path))
         
        for category in CATEGORIES:
            class_num = CATEGORIES.index(category) 
            p=os.path.join(path,category)
            #       from keras.applications import ResNet50
            for img in os.listdir(p):
                try:
                    img_array = cv2.imread(os.path.join(p,img))  # convert to array
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
    
                    testing_data.append([np.array(new_array), class_num])  # add this to our training_data
                except Exception as e:  # in the interest in keeping the output clean...
                    pass
    print(len(training_data))
    print(len(testing_data))
    random.shuffle(training_data)
    random.shuffle(testing_data)
    X = []
    y = []
    xtest = []
    ytest = []
    for features,label in training_data:
        X.append(features)
        y.append(label)
    for features,label in testing_data:
        xtest.append(features)
        ytest.append(label)
    X=np.array(X)/255.0 
    xtest = np.array(xtest)/255.0
    y=to_categorical(y)
    ytest = to_categorical(ytest)
    #print(y.shape)

    base_model = ResNet50(weights = "imagenet",include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(23, activation='softmax')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # First: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss="categorical_crossentropy", optimizer="sgd",
                metrics=["accuracy"])
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
    model.fit(X, y,batch_size=32,epochs=1,validation_data=(xtest,ytest))
    #print('Accuracy: ',model.evaluate())

                   


