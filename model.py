import numpy as np
import cv2
import os

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Add
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend

# input parameters
image_shape = (576, 160)
width = image_shape[0]
height = image_shape[1]
num_classes = 3

img_dir = "./data/data_road/training/image_2/" # végén a /
label_dir = "./data/data_road/training/gt_image_2/"

# color palette used for one-hot encoding
palette = {(0, 0, 255):0,
           (255, 0, 255):1,
           (0, 0, 0):2}

def input_image_array(path, width, height):
    img = cv2.imread(path, 1)
    img_array = np.float32(cv2.resize(img, (width, height))) / 255.0
    return img_array

def input_label_array(path, width, height, num_classes, color_codes):
    label = cv2.imread(path)
    label = cv2.resize(label, (width, height))

    int_array = np.ndarray(shape=(height, width), dtype=int)
    int_array[:,:] = 0

    # rgb to integer
    for rgb, idx in color_codes.items():
        int_array[(label==rgb).all(2)] = idx

    one_hot_array = np.zeros((height, width, num_classes))

    # one-hot encoding
    for c in range(num_classes):
        one_hot_array[:, :, c] = (int_array == c).astype(int)

    return one_hot_array

# lists to append input images and labels
X = []
y = []

images = os.listdir(img_dir)
images.sort()
labels = os.listdir(label_dir)
labels.sort()

for img, label in zip(images, labels):
    X.append(input_image_array(img_dir + img, width, height))
    y.append(input_label_array(label_dir + label, width, height, num_classes, palette))

# input layer takes NumPy Arrays
X, y = np.array(X), np.array(y)

# could be put separately into a class
def build_headnet(baseModel, classes):

    # random initializer and regularizer + adding dropout
    init = RandomNormal(stddev=0.01)
    reg = l2(1e-3)

    # input layers from encoder
    pool_3 = baseModel.get_layer(index=10).output
    pool_4 = baseModel.get_layer(index=14).output
    pool_5 = baseModel.output

    # convolutions
    score_3 = Conv2D(classes, kernel_size=(
    1, 1), strides=(1, 1), padding='same', kernel_initializer=init, kernel_regularizer=reg)(pool_3)
    score_4 = Conv2D(classes, kernel_size=(
    1, 1), strides=(1, 1), padding='same', kernel_initializer=init, kernel_regularizer=reg)(pool_4)
    score_7 = Conv2D(classes, kernel_size=(
    1, 1), strides=(1, 1), padding='same', kernel_initializer=init, kernel_regularizer=reg)(pool_5)

    # deconvolution - score_7 - 2x upsample -- kernel_size, stride 4,2?
    upsample_32 = Conv2DTranspose(classes, kernel_size=(
    3, 3), strides=(2, 2), padding="same", kernel_initializer=init, kernel_regularizer=reg)(score_7)

    # upsample_32 + score_4 skip connection
    score_4_7 = Add()([upsample_32, score_4])

    # deconvolution - score_4_7 - 2x upsample -- kernel_size, stride 4,2?
    upsample_16 = Conv2DTranspose(classes, kernel_size=(
    3, 3), strides=(2, 2), padding="same", kernel_initializer=init, kernel_regularizer=reg)(score_4_7)

    # upsample_16 + score_3 skip connection
    score_3_4_7 = Add()([upsample_16, score_3])

    # deconvolution - score_3_4_7 - 8x upsample
    upsample_8 = Conv2DTranspose(classes, kernel_size=(
    16,16), strides=(8,8), padding="same", kernel_initializer=init, kernel_regularizer=reg,
    activation="softmax")(score_3_4_7)

    return upsample_8

# https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
# encoder - pre-trained on ImageNet - fully connected layer is dropped
encoder = VGG16(weights="imagenet", include_top=False,
                input_tensor=Input(shape=(image_shape[1], image_shape[0], 3)))

# Head of the network - only this part is trained when we apply Transfer Learning
decoder = build_headnet(encoder, num_classes)

# Fully Convolutional Network - Encoder and Decoder connected
fcn_network = Model(inputs=encoder.input, outputs=decoder)

# Freezing the encoder for fine-tuning - only training the decoder part
for layer in encoder.layers:
    layer.trainable = False

#compiling - loss function is categorical_crossentropy
adam = tf.keras.optimizers.Adam()
fcn_network.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
fcn_network.summary()

epochs = 200
batch_size = 20

# Backpropagation
fcn_network.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Saving the model
fcn_network.save("fcn-8.model")

# Saving weights
#fcn_network.save_weights("fcn8.h5")
