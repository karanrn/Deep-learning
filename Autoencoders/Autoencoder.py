# Autoencoder
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import UpSampling2D
from keras.models import Sequential
from keras.preprocessing import image
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# autoencoder
autoencoder = Sequential()
autoencoder.add(Convolution2D(64, 3, 3, input_shape = (128, 128, 3), activation= 'relu', border_mode = 'same'))
autoencoder.add(MaxPooling2D((2,2), border_mode = 'same'))
autoencoder.add(Convolution2D(64, 3, 3, activation= 'relu', border_mode = 'same'))
autoencoder.add(MaxPooling2D((2,2), border_mode = 'same'))
autoencoder.add(Convolution2D(16, 3, 3, activation= 'relu', border_mode = 'same'))
autoencoder.add(MaxPooling2D((2,2), border_mode = 'same'))

# Decoder
autoencoder.add(Convolution2D(16, 3, 3, activation= 'relu', border_mode = 'same'))
autoencoder.add(UpSampling2D((2,2)))
autoencoder.add(Convolution2D(32, 3, 3, activation= 'relu', border_mode = 'same'))
autoencoder.add(UpSampling2D((2,2)))
autoencoder.add(Convolution2D(64, 3, 3, activation= 'relu', border_mode = 'same'))
autoencoder.add(UpSampling2D((2,2)))
autoencoder.add(Convolution2D(3, 3, 3, activation= 'softmax', border_mode = 'same'))

# Compile the model
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
autoencoder.summary()

# Training image
img1 = image.load_img('superman.jpg', target_size = (128, 128))
img_train = image.img_to_array(img1)
img_train = np.expand_dims(img_train, axis=0)  
img_train /= 255.

# Training
autoencoder.fit(img_train, img_train, epochs=150)
x = autoencoder.predict(img_train)
plt.imsave('superman_Autoencoder.jpg', x[0])