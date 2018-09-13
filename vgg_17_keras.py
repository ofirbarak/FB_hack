import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_normal
from keras.utils import np_utils
from keras import backend as K

from oxford_classes import classes


if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")

# Import Tensorflow with multiprocessing
import tensorflow as tf
import multiprocessing as mp

core_num = mp.cpu_count()
print(core_num)
config = tf.ConfigProto(
    inter_op_parallelism_threads=core_num,
    intra_op_parallelism_threads=core_num)
sess = tf.Session(config=config)

# Declare variables

BATCH_NORM = False

INPUT_SHAPE = 3,224,224
num_classes = 18


def base_model():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=INPUT_SHAPE, name='block1_conv1'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same', name='block1_conv2'))
    model.add(BatchNormalization()) if BATCH_NORM else None
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    #
    # model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv1'))
    # model.add(BatchNormalization()) if BATCH_NORM else None
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv2'))
    # model.add(BatchNormalization()) if BATCH_NORM else None
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    #
    # model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv1'))
    # model.add(BatchNormalization()) if BATCH_NORM else None
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv2'))
    # model.add(BatchNormalization()) if BATCH_NORM else None
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv3'))
    # model.add(BatchNormalization()) if BATCH_NORM else None
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv4'))
    # model.add(BatchNormalization()) if BATCH_NORM else None
    # model.add(Activation('relu'))
    #
    # model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    #
    # model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv1'))
    # model.add(BatchNormalization()) if BATCH_NORM else None
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv2'))
    # model.add(BatchNormalization()) if BATCH_NORM else None
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv3'))
    # model.add(BatchNormalization()) if BATCH_NORM else None
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv4'))
    # model.add(BatchNormalization()) if BATCH_NORM else None
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    #
    # model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv1'))
    # model.add(BatchNormalization()) if BATCH_NORM else None
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv2'))
    # model.add(BatchNormalization()) if BATCH_NORM else None
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv3'))
    # model.add(BatchNormalization()) if BATCH_NORM else None
    # model.add(Activation('relu'))
    #
    # model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv4'))
    # model.add(BatchNormalization()) if BATCH_NORM else None
    # model.add(Activation('relu'))
    #
    # model.add(Flatten())
    #
    # model.add(Dense(4096))
    # model.add(BatchNormalization()) if BATCH_NORM else None
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    #
    # model.add(Dense(4096, name='fc2'))
    # model.add(BatchNormalization()) if BATCH_NORM else None
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    #
    # model.add(Dense(num_classes))
    # model.add(BatchNormalization()) if BATCH_NORM else None
    # model.add(Activation('softmax'))

    sgd = SGD(lr=0.0005, decay=0, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def train_model(model, X, Y):
#     model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, Y, epochs=1, batch_size=20)
    return model

def save_weights(model, weights_path="vgg17_weights.h5"):
    model.save_weights(weights_path)
    
def load_model_and_weights(model_path='model.json', weights_path="model.h5"):
    # load json and create model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)
    return loaded_model

def run(image, model_path='model.json', weights_path="model.h5"):
    # resize image
#     resized_im = cv2.resize(im, (224, 224)).astype(np.float32)
#     resized_im = resized_im.transpose((2,0,1))
#     resized_im = np.expand_dims(resized_im, axis=0)
    
    loaded_model = VGG_17(weights_path="vgg17_weights.h5")
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    loaded_model.compile(optimizer=sgd, loss='categorical_crossentropy')    
    predict = loaded_model.predict(image)
    return classes[np.argmax(predict)]
    
    
    
    
    
    
    
    
    
# if __name__ == "__main__":
#     im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
#     im[:,:,0] -= 103.939
#     im[:,:,1] -= 116.779
#     im[:,:,2] -= 123.68
#     im = im.transpose((2,0,1))
#     im = np.expand_dims(im, axis=0)

#     # Test pretrained model
#     model = VGG_16('vgg16_weights.h5')
#     sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(optimizer=sgd, loss='categorical_crossentropy')
#     out = model.predict(im)
#     print(np.argmax(out))