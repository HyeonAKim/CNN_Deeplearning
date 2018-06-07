from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout,Flatten,Dense
import numpy as np
import math
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# 이미지크기
img_width, img_height = 256  , 256

top_model_weights_path = '..\\CNN_Deeplearning\\Models\\Bearing_VGG16\\model\\top_model\\bottleneck_fc_model.h5'
train_data_dir = '..\\CNN_Deeplearning\\Data\\Dataset\\Bearing\\train'
test_date_dir = '..\\CNN_Deeplearning\\Data\\Dataset\\Bearing\\test'
nb_train_samples  = 280
nb_test_samples = 120

epochs = 100
batch_size=10

predict_size_train = int(math.ceil(nb_train_samples / batch_size))
predict_size_test = int(math.ceil(nb_test_samples / batch_size))

#
# def save_bottleneck_features():
#     datagen = ImageDataGenerator(rescale=1. / 255)
#
#     # VGG 16 네트워크 가져오기
#     model = applications.VGG16(include_top=False, weights='imagenet')
#
#     # 훈련데이터 가져오기
#     generator = datagen.flow_from_directory(
#         train_data_dir,
#         target_size=(img_width, img_height),
#         batch_size=batch_size,
#         class_mode=None,
#         shuffle=False
#     )
#
#    # 이미지를 예측하고 , 훈련 데이터 저장하기
#     bottleneck_features_train = model.predict_generator(generator, predict_size_train)
#     np.save(open('..\\CNN_Deeplearning\\Models\\Bearing_VGG16\\model\\feature\\bottleneck_feature_train.npy'
#                    , 'wb'), bottleneck_features_train)
#
#     generator = datagen.flow_from_directory(
#         test_date_dir,
#         target_size=(img_width, img_height),
#         batch_size=batch_size,
#         class_mode=None,
#         shuffle=False
#     )
#     bottleneck_features_test = model.predict_generator(generator, predict_size_test)
#     np.save(open('..\\CNN_Deeplearning\\Models\\Bearing_VGG16\\model\\feature\\bottleneck_features_test.npy'
#                  , 'wb'),
#             bottleneck_features_test)
#
#
# save_bottleneck_features()

# 출력층 학습하기 
def train_top_model():
    train_data = np.load(open('..\\CNN_Deeplearning\\Models\\Bearing_VGG16\\model\\feature\\bottleneck_feature_train.npy'
                              , 'rb'))
    train_labels = np.array(
        [0] * (nb_train_samples // 4) + [1] * (nb_train_samples // 4)+ [2] * (nb_train_samples // 4)+ [3] * (nb_train_samples // 4))
    train_labels_categorical = np_utils.to_categorical(train_labels)

    test_data = np.load(
        open('..\\CNN_Deeplearning\\Models\\Bearing_VGG16\\model\\feature\\bottleneck_features_test.npy', 'rb'))
    test_labels = np.array(
        [0] * (nb_test_samples // 4) + [1] * (nb_test_samples // 4)+ [2] * (nb_test_samples // 4)+ [3] * (nb_test_samples // 4))
    test_labels_categorical = np_utils.to_categorical(test_labels)
    # print(test_labels)
    # print(train_data.shape[1:])
    #
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stoppping_callback = EarlyStopping(monitor='val_loss', patience=10)

    model.fit(train_data,train_labels_categorical,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(test_data,test_labels_categorical),
              callbacks=[early_stoppping_callback]
              )
    model.save_weights(top_model_weights_path)


train_top_model()

