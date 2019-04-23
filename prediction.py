import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# loading in the trained model
model = tf.keras.models.load_model("fcn-8.model")

# color map for segmentation - same as the palette in predict
colors = np.array([[255, 0, 0],
                   [255,0,255],
                   [0,  0,  0]])

# reshaping at the end to fit model dimensions
def prepare(path):
    img = cv2.imread(path)
    b,g,r = cv2.split(img)       # get b,g,r
    img_array_rgb = cv2.merge([r,g,b])     # switch it to rgb
    img_array = cv2.resize(img_array_rgb, (576, 160))

    return img_array.reshape(-1, 160, 576, 3)

image_path = 'umm_000090.png'

prediction = model.predict([prepare(image_path)])

# taking the depthwise argmax - choose given pixel from channel with the highest prob.
mask = np.argmax(prediction[0], axis=2)

# final segmentation result
colored_mask = colors[mask]

def prepare2(path):
    img = cv2.imread(path)
    b,g,r = cv2.split(img)       # get b,g,r
    img_array_rgb = cv2.merge([r,g,b])     # switch it to rgb
    img_array = cv2.resize(img_array_rgb, (576, 160))

    return img_array

fig = plt.figure()

fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(prepare2(image_path))
ax2.imshow(colored_mask)

plt.show()
