from keras.models import save_model, load_model, Sequential
import matplotlib.pyplot as plt
import pandas as pd
import os
import operator
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np



# 예측 클래스 정보
train_data_dir = '..\\CNN_Deeplearning\\Data\\Dataset\\Bearing\\train'
img_width, img_height = 256, 256
batch_size = 1

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

labels = train_generator.class_indices

# predict 모델 및  데이터 가져오기
predict_data_dir = '..\\CNN_Deeplearning\\Predict\\Bearing'
model = load_model('..\\CNN_Deeplearning\\Models\\Bearing_VGG16\model\\fine_tunning\\02-0.6568-0.8667.h5')


label_name= sorted(labels.items(), key=operator.itemgetter(1))
label_keys = []
i = 0
for n in label_name:
    label_keys.append(label_name[i][0])
    i +=1




for filename in os.listdir(predict_data_dir):

    img_dir = os.path.join(predict_data_dir,filename)
    img = load_img(img_dir, target_size=(img_height,img_width))
    x = img_to_array(img)
    x = x /255
    x = x.reshape((1,)+x.shape)

    predict_result = []
    for result in np.squeeze(model.predict(x)):
        predict_result.append("%.2f"%result)

    result_dict = dict(zip(label_keys, predict_result))
    print('Predict Result: ',sorted(result_dict.items(),key=operator.itemgetter(1),reverse=True))
    plt.imshow(img)
    plt.title(filename)
    plt.show()
