from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np

# 1. test 데이터 업로드
## 1. 이미지 데이터 셋팅
img_width, img_height = 100  , 100

test_date_dir = 'C:\\Users\\HyunA\\PycharmProjects\\CNN_Deeplearning\\Data\\Dataset\\Bearing\\test'
nb_test_samples = 120


## 2. 이미지 데이터 불러오기
datagen = ImageDataGenerator(rescale=1. / 255)
image_generator = datagen.flow_from_directory(
    test_date_dir,
    target_size=(img_width, img_height),
    batch_size=nb_test_samples,
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
# classes = np.argmax(pred_lbl)

# 2. train 모델 업로드

autoencoder = load_model('C:\\Users\\HyunA\\PycharmProjects\\CNN_Deeplearning\\Models\\Bearing_AutoEncoder\\bearing_denosinge_autoencoder.h5')

# 3. test 데이터 predict
# predict test dataset
pred = autoencoder.predict(x_batch)
plt.figure(figsize=(20,4))
print("Test images")

for i in range(10,20,1):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_batch[i], cmap='gray')
    curr_lbl = np.argmax(y_batch[i])

    plt.title("("+str(label_list[curr_lbl][1])+")")
plt.show()

plt.figure(figsize=(20,4))
print("Reconstruction of Noisy Test Image")
for i in range(10,20,1):
    plt.subplot(2,10,i+1)
    plt.title("("+str(label_list[curr_lbl][1])+")")
    plt.imshow(pred[i,...,0],cmap='gray')
plt.show()