## 필요한 라이브러리
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from matplotlib import pyplot as plt


## 1. 이미지 데이터 셋팅
img_width, img_height = 100  , 100

train_data_dir = 'C:\\Users\\HyunA\\PycharmProjects\\CNN_Deeplearning\\Data\\Dataset\\Bearing\\train'
test_date_dir = 'C:\\Users\\HyunA\\PycharmProjects\\CNN_Deeplearning\\Data\\Dataset\\Bearing\\test'
nb_train_samples  = 280
nb_test_samples = 120

# predict_size_train = int(math.ceil(nb_train_samples / batch_size))
# predict_size_test = int(math.ceil(nb_test_samples / batch_size))

## 2. 이미지 데이터 불러오기
datagen = ImageDataGenerator(rescale=1. / 255)
image_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=nb_train_samples,
        class_mode='categorical',
        shuffle=False
    )

# # Returns
#             A DirectoryIterator yielding tuples of `(x, y)` where `x` is a nu                                                                                            mpy array containing a batch
#             of images with shape `(batch_size, *target_size, channels)` and `y` is a numpy array of corresponding labels.


## 3. 이미지 데이터 확인하기

x_batch, y_batch = next(image_generator)

# x (image) , y(label) shape 확인

print(x_batch.shape)  # (batch_size, target_size, channels)
print(y_batch.shape)  # (batch_size, classe num)

# 훈련 image 확인
plt.figure(figsize=[5, 5])

# display the first image in training data
plt.subplot(121)
curr_img = x_batch[0].reshape((100, 100, -1))
plt.imshow(curr_img, cmap='gray')
plt.title(y_batch[0])

plt.subplot(122)
curr_img = x_batch[1].reshape((100, 100, -1))
plt.imshow(curr_img, cmap='gray')
plt.title(y_batch[1])

plt.show()

## 4. label_list 생성하기

# generator 라벨링
label_dict = image_generator.class_indices

label_list = []
for key, val in label_dict.items():
    label_list.append((val, key))

label_list = sorted(label_list)
print(label_list)

## 5. train, validation 데이터 분리하기
from sklearn.model_selection import train_test_split

train_X, valid_X, train_labels, valid_labels = train_test_split(x_batch, y_batch, test_size=0.2, random_state=13)

print(train_X.shape)
print(train_labels.shape)
print(valid_X.shape)
print(valid_labels.shape)


## 6. 이미지 데이터에 노이즈 주기
# noise 추가
noise_factor = 0.5
x_train_noisy = train_X + noise_factor * np.random.normal(loc=0.0,scale=1.0, size = train_X.shape)
x_valid_noisy = valid_X + noise_factor * np.random.normal(loc=0.0,scale=1.0, size = valid_X.shape)

x_train_noisy = np.clip(x_train_noisy,0.,1.)
x_valid_noisy = np.clip(x_valid_noisy,0.,1.)


plt.figure(figsize=[5,5])
print(x_train_noisy[1].shape)
# display the first image in training data
plt.subplot(121)
curr_img = x_train_noisy[0].reshape((100,100,-1))
plt.imshow(curr_img, cmap='gray')

# display the first image in testing data
plt.subplot(122)
curr_img = x_train_noisy[1].reshape((100,100,-1))
plt.imshow(curr_img, cmap='gray')

plt.show()

from keras import Model
from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import RMSprop

## 7. denosing autoencoder network
batch_size = 70
epoch = 50
inChannel = 3
x, y = 100, 100
input_img = Input(shape=(x, y, inChannel))


def autoencoder(input_img):
    # encoder
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1')(input_img)
    pool1 = MaxPool2D(pool_size=(2, 2), name='pool1')(conv1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2')(pool1)
    pool2 = MaxPool2D(pool_size=(2, 2), name='pool2')(conv2)
    conv3 = Conv2D(316, (3, 3), activation='relu', padding='same', name='conv3')(pool2)

    # decoder
    conv4 = Conv2D(316, (3, 3), activation='relu', padding='same', name='conv4')(conv3)
    up2 = UpSampling2D((2, 2), name='up2')(conv4)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv5')(up2)
    up3 = UpSampling2D((2, 2), name='up3')(conv5)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decoded')(up3)
    return decoded


autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())

print(x_train_noisy.shape)
print(train_X.shape)
print(input_img.shape)

autoencoder_trian = autoencoder.fit(x_train_noisy,train_X, batch_size=batch_size, epochs = epoch, verbose = 1, validation_data = (x_valid_noisy,valid_X))


# 모델 저장

autoencoder.save("C:\\Users\\HyunA\\PycharmProjects\\CNN_Deeplearning\\Models\\Bearing_AutoEncoder\\bearing_denosinge_autoencoder.h5")

# 가중치 저장
autoencoder.save_weights("C:\\Users\\HyunA\\PycharmProjects\\CNN_Deeplearning\\Models\\Bearing_AutoEncoder\\bearing_denosinge_autoencoder_weights.h5")