import csv
import os
import numpy as np
import cv2
from keras.models import Sequential, Model
from keras.layers import Lambda, Flatten, Dense, Cropping2D,Convolution2D,MaxPooling2D,Dropout,Activation
from sklearn.model_selection import train_test_split
import sklearn
import random

def load_samples(file_path):
	samples = []
	with open(file_path) as csvfile:
		reader = csv.reader(csvfile)
		flag = False
		for line in reader:
			if flag != False:
				samples.append(line)
			flag = True
	return samples

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		random.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			right_cam_correction = 0.165
			left_cam_correction = 0.165
			for batch_sample in batch_samples:
				name = batch_sample[0]
				center_image = cv2.imread(name)
				center_image = center_image[60:140, :] 
				center_image = cv2.resize(center_image, (80,40), interpolation = cv2.INTER_AREA)
				center_angle = float(batch_sample[3])
				images.append(center_image)
				angles.append(center_angle)
				
				img_flipped = np.fliplr(center_image)
				y_flipped = -center_angle
				images.append(img_flipped)
				angles.append(y_flipped)
				
				temp = str(batch_sample[1])
				temp2 = temp[1:]
				img2 = cv2.imread(str(temp2))
				img2 = img2[60:140, :]
				img2 = cv2.resize(img2, (80,40), interpolation = cv2.INTER_AREA)
				y_left = center_angle + left_cam_correction
				images.append(img2)
				angles.append(y_left)
		
				temp = str(batch_sample[2])
				temp2 = temp[1:]
				img3 = cv2.imread(str(temp2))
				img3 = img3[60:140, :]
				img3 = cv2.resize(img3, (80,40), interpolation = cv2.INTER_AREA)
				y_right = center_angle - right_cam_correction
				images.append(img3)
				angles.append(y_right)

			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)
	
	
def MNet(model):
	model.add(Lambda(lambda x: (x / 255.0) - 0.5,input_shape=(40,80,3)))
	model.add(Convolution2D(24, 5, 5, subsample = (2,2), activation='relu'))
	model.add(Convolution2D(36, 5, 5, activation='relu'))
	model.add(Convolution2D(48, 5, 5, activation='relu'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	return model
	
samples = load_samples("driving_log.csv")
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model = MNet(model)

model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= \
			len(train_samples), validation_data=validation_generator, \
			nb_val_samples=len(validation_samples), nb_epoch=3)
model.save("model.h5")