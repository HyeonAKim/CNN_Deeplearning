from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

# 가중치 저장 경로
weight_path = '..\\CNN_Deeplearning\\Models\\Bearing_ResNet50\\model\\fine_tunning\\ResNet50_weights.h5'
top_model_weights_path = '..\\CNN_Deeplearning\\Models\\Bearing_ResNet50\\model\\top_model\\resnet50_fc_model.h5'

# 이미지 사이즈
img_width, img_height = 256, 256
train_data_dir = '..\\CNN_Deeplearning\\Data\\Dataset\\Bearing\\train'
test_date_dir = '..\\CNN_Deeplearning\\Data\\Dataset\\Bearing\\test'
nb_train_samples  = 280
nb_test_samples = 120

epochs = 10
batch_size=10

# build the ResNet50 network
model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
print('Model loaded')

# build a classifier model to put on top of convolution model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(4, activation='softmax'))

# 사전에 학습된 fully_trained로 시작해야함
# classifier, including the top classifier
# in order to successfully do fine-tuning

top_model.load_weights(top_model_weights_path)

# add the model on top of the convolution base
model = Model(input=model.input, output=top_model(model.output))
# model.add(top_model)

# set the first 25 layers (( 마지막 conv block)
# to - non-trianable ( weight will not be updated)
for layer in model.layers[:165]:
    layer.trainable = False

# complie the model with a SGD/ momentun optimizer
# and a very slow learning rate
model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    test_date_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


modelpath = "..\\CNN_Deeplearning\\Models\\Bearing_ResNet50\\model\\fine_tunning\\{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.h5"

# 모델 업데이트 및 저장
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)

# 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_test_samples,
    callbacks=[early_stopping_callback, checkpointer])

model.save_weights(weight_path)
model.save(modelpath)