from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout,Flatten,Dense
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np
import math
import pathlib
import os

# 이미지크기
img_width, img_height = 256  , 256

top_model_weights_path = '..\\CNN_Deeplearning\\Models\\Bearing_ResNet50\\model\\top_model\\resnet_fc_model.h5'
pathlib.Path(top_model_weights_path).mkdir(parents=True, exist_ok=True)

train_feature_save_path = '..\\CNN_Deeplearning\\Models\\Bearing_ResNet50\\model\\feature\\resnet50_feature_train.npy'
pathlib.Path(train_feature_save_path).mkdir(parents=True, exist_ok=True)

test_feature_save_path = '..\\CNN_Deeplearning\\Models\\Bearing_ResNet50\\model\\feature\\bottleneck_features_test.npy'
pathlib.Path(test_feature_save_path).mkdir(parents=True, exist_ok=True)

train_data_dir = '..\\CNN_Deeplearning\\Data\\Dataset\\Bearing\\train'
test_date_dir = '..\\CNN_Deeplearning\\Data\\Dataset\\Bearing\\test'
nb_train_samples  = 280
nb_test_samples = 120

epochs = 100
batch_size=10

predict_size_train = int(math.ceil(nb_train_samples / batch_size))
predict_size_test = int(math.ceil(nb_test_samples / batch_size))


def save_resnet_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # VGG 16 네트워크 가져오기
    model = applications.ResNet50(include_top=False, weights='imagenet')

    # 훈련데이터 가져오기
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )

   # 이미지를 예측하고 , 훈련 데이터 저장하기
    resnet_features_train = model.predict_generator(generator, predict_size_train)
    np.save(open(train_feature_save_path, 'wb'), resnet_features_train)

    generator = datagen.flow_from_directory(
        test_date_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    resnet_features_test = model.predict_generator(generator, predict_size_test)
    np.save(open(test_feature_save_path, 'wb'),
            resnet_features_test)


save_resnet_features()

# 출력층 학습하기
def train_top_model():
    train_data = np.load(open(train_feature_save_path, 'rb'))
    train_labels = np.array(
        [0] * (nb_train_samples // 4) + [1] * (nb_train_samples // 4)+ [2] * (nb_train_samples // 4)+ [3] * (nb_train_samples // 4))
    train_labels_categorical = np_utils.to_categorical(train_labels)

    test_data = np.load(open(train_feature_save_path, 'rb'))
    test_labels = np.array(
        [0] * (nb_test_samples // 4) + [1] * (nb_test_samples // 4)+ [2] * (nb_test_samples // 4)+ [3] * (nb_test_samples // 4))
    test_labels_categorical = np_utils.to_categorical(test_labels)
    # print(test_labels)
    # print(train_data.shape[1:])
    #
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(4),activation='softmax')

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stoppping_callback = EarlyStopping(monitor='val_loss', patience=3)

    model.fit(train_data,train_labels_categorical,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(test_data,test_labels_categorical),
              callbacks=[early_stoppping_callback]
              )
    model.save_weights(top_model_weights_path)


train_top_model()

