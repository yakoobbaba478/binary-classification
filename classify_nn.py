import os
import sys
import cv2
from os import listdir
from os.path import join,splitext,basename
from  imgaug import augmenters as iaa

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

global classifier


classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

def train(train_path, test_path):
	train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

	test_datagen = ImageDataGenerator(rescale = 1./255)

	training_set = train_datagen.flow_from_directory(train_path, target_size = (64, 64), batch_size = 32, class_mode = 'binary')

	test_set = test_datagen.flow_from_directory(test_path, target_size = (64, 64), batch_size = 32, class_mode = 'binary')

	classifier.fit_generator(training_set, steps_per_epoch = 8000, epochs = 25, validation_data = test_set, validation_steps = 2000)
	
	mc = keras.callbacks.ModelCheckpoint('weights{epoch:08d}.h5', 
                                     save_weights_only=True, period=1)
	classifier.fit(X_train, Y_train, callbacks=[mc])

	#classifier.save("/home/baba/classify_train.h5")


def evaluating(src_path):
	
	global training_set
	global test_set
	classifier = load_model("/home/baba/classify_train.h5")
	
	test_image = image.load_img(src_path, target_size = (64, 64))

	test_image = image.img_to_array(test_image)

	test_image = np.expand_dims(test_image, axis = 0)

	result = classifier.predict(test_image)
	
	print(result)
	
	#training_set.class_indices

	if result[0][0] == 1:
		prediction = 'dog'
		print("the given image is a Dog")
	else:
		prediction = 'cat'
		print("the given image is a Cat")


if __name__ =="__main__":

	mode = sys.argv[1]
	if(mode == "train"):
		train_path = "/home/baba/Documents/classify__nn/training_set"
		test_path = "/home/baba/Documents/classify__nn/test_set"
		#file_path = "/home/baba/Documents/data_saved.h5"
		train(train_path, test_path)
		print("training completed")
		#classifier.save("/home/baba/classify_train.h5")
	elif(mode == "eval"):
	
		src_path = sys.argv[2]
		evaluating(src_path) 
