from scipy import spatial
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

# 1. 훈련이미지, 테스트이미지 유사도 업로드
train_data = np.load(open('C:\\Users\\HyunA\\PycharmProjects\\CNN_Deeplearning\\Models\\Bearing_AutoEncoder\\AES_features_train.npy', 'rb'))
test_data = np.load(open('C:\\Users\\HyunA\\PycharmProjects\\CNN_Deeplearning\\Models\\Bearing_AutoEncoder\\AES_features_test.npy', 'rb'))

print(train_data.shape)  # (건수, feature수 )
print(test_data.shape)

# 2. 기존이미지, 새로운 이미지 출력값 유사도 계산
result = []
for train in train_data:
    result.append(1 - spatial.distance.cosine(test_data[1],train))

# 3. 유사도가 가장 높은 값 10 개 보여주기
## result tuple 생성
result_index = []

for i in result:
    # print(i)
    (x, x_index) = i, result.index(i)
    result_index.append((x, x_index))

sorted_result = sorted(result_index,reverse=True)[:10]
print(sorted_result)

# 4. 테스트 이미지 보여주기
## 이미지 데이터 셋팅
img_width, img_height = 100  , 100

train_data_dir = 'C:\\Users\\HyunA\\PycharmProjects\\CNN_Deeplearning\\Data\\Dataset\\Bearing\\train'
nb_train_samples  = 280
test_date_dir = 'C:\\Users\\HyunA\\PycharmProjects\\CNN_Deeplearning\\Data\\Dataset\\Bearing\\test'
nb_test_samples = 120


## 이미지 데이터 불러오기
datagen = ImageDataGenerator(rescale=1. / 255)

train_image_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=nb_train_samples,
        class_mode='categorical',
        shuffle=False
    )

test_image_generator = datagen.flow_from_directory(
    test_date_dir,
    target_size=(img_width, img_height),
    batch_size=nb_test_samples,
    class_mode='categorical',
    shuffle=False
    )

# # Returns
#             A DirectoryIterator yielding tuples of `(x, y)` where `x` is a nu                                                                                            mpy array containing a batch
#             of images with shape `(batch_size, *target_size, channels)` and `y` is a numpy array of corresponding labels.

x_train, y_train = next(train_image_generator)
x_test, y_test = next(test_image_generator)

# test 이미지 확인
curr_img = x_test[1].reshape((100, 100, -1))
plt.imshow(curr_img, cmap='gray')
plt.title(y_test[1])
plt.show()

# 5. 유사도 높은 이미지 보여주기
plt.figure(figsize=(20,4))
print("similar images")

for i in range(0,10,1):
    print(sorted_result[i][1])
    plt.subplot(2, 10, i+1)
    plt.imshow(x_train[(sorted_result[i][1])], cmap='gray')
    curr_lbl = np.argmax(y_train[i])

plt.show()
