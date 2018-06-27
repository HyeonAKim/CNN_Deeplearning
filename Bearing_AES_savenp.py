from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# 1. 유사도 특질 추출 모델 설계
inChannel = 3
x, y = 100,100
input_img = Input(shape=(x,y,inChannel))

def autoencoder_similar(input_img):
    # encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1')(input_img)
    pool1 = MaxPool2D(pool_size=(2, 2), name='pool1')(conv1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2')(pool1)
    pool2 = MaxPool2D(pool_size=(2, 2), name='pool2')(conv2)
    conv3 = Conv2D(316, (3, 3), activation='relu', padding='same', name='conv3')(pool2)
    pool3 = MaxPool2D(pool_size=(2, 2), name='pool3')(conv3)
    # flatten = Flatten()(pool3)
    return pool3

AES = Model(input_img, autoencoder_similar(input_img))

print(AES.summary())

# 2. 학습된 AE 가중치 업로드
AES.load_weights('C:\\Users\\HyunA\\PycharmProjects\\CNN_Deeplearning\\Models\\Bearing_AutoEncoder\\bearing_denosinge_autoencoder_weights.h5',by_name=True )

# 3. 기존 이미지 출력값 저장
## 기존 이미지 데이터 불러오기
img_width, img_height = 100  , 100

train_data_dir = 'C:\\Users\\HyunA\\PycharmProjects\\CNN_Deeplearning\\Data\\Dataset\\Bearing\\train'
nb_train_samples  = 280
predict_size_train = 1

datagen = ImageDataGenerator(rescale=1. / 255)
image_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=nb_train_samples,
        class_mode=None,
        shuffle=False
    )

## 기존 이미지 예측하기
AES_features_train = AES.predict_generator(image_generator, predict_size_train)

## 기존 이미지 예측 값 저장
np.save(open('C:\\Users\\HyunA\\PycharmProjects\\CNN_Deeplearning\\Models\\Bearing_AutoEncoder\\AES_features_train.npy'
                 , 'wb'),AES_features_train)

# 4. 예측이미지 업로드

test_date_dir = 'C:\\Users\\HyunA\\PycharmProjects\\CNN_Deeplearning\\Data\\Dataset\\Bearing\\test'
nb_test_samples = 120
predict_size_test = 1

datagen = ImageDataGenerator(rescale=1. / 255)
image_generator = datagen.flow_from_directory(
    test_date_dir,
    target_size=(img_width, img_height),
    batch_size=nb_test_samples,
    class_mode=None,
    shuffle=False
    )

## 테스트 이미지 예측하기
AES_features_test = AES.predict_generator(image_generator, predict_size_test)

## 테스트 이미지 예측 값 저장
np.save(open('C:\\Users\\HyunA\\PycharmProjects\\CNN_Deeplearning\\Models\\Bearing_AutoEncoder\\AES_features_test.npy'
                 , 'wb'),AES_features_test)

