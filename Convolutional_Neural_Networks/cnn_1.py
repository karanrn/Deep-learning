# Convolution neural networks
# Building CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising CNN
classifier = Sequential()

# First CNN layer
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64,3), activation= 'relu'))
# Pooling step - Maxpooling
classifier.add(MaxPooling2D( pool_size=(2, 2)))

# Second CNN layer
classifier.add(Convolution2D(32, 3, 3, activation= 'relu'))
# Pooling step - Maxpooling
classifier.add(MaxPooling2D( pool_size=(2, 2)))

# Flattening step
classifier.add(Flatten())

# Fully connected layer
classifier.add(Dense( output_dim = 128, activation = 'relu'))
classifier.add(Dense( output_dim = 1, activation = 'sigmoid'))

# Compiling model
classifier.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Image preprocessing step
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   vertical_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)
                        
classifier.save('dogs_cats.h5')

# Prediction
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
model = load_model('dogs_cats.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img = image.load_img('dataset/test_set/dogs/dog.4001.jpg', target_size = (64, 64))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)  
img_tensor /= 255.

plt.imshow(img_tensor[0])
plt.show()

classes = model.predict_classes(img_tensor)

if classes[0][0] == 0:
    print("It's cat.")
else:
    print("It's dog.")