import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import applications
from keras import optimizers
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Lambda, Input, Cropping2D


def load_data(data_dirs):
	img_paths, angles = [], []
	for data_dir in data_dirs:
		csv_path = os.path.join(data_dir, 'driving_log.csv')
		columns = ['center', 'left', 'right', 'steering_angle', 'throttle', 'break', 'speed']
		df = pd.read_csv(csv_path, header=None, names=columns)
		df_straight = df[df['steering_angle'] == 0]
		df_turn = df[df['steering_angle'] != 0]
		df_train = pd.concat([df_turn, df_straight])
		for i, row in df_train.iterrows():
			img_paths += row[['center', 'left', 'right']].tolist()
			angle = row['steering_angle']
			angles += [angle, angle + 0.2, angle - 0.2]    
	return img_paths, angles


def random_flip(img, angle):
	if np.random.rand() > 0.5:
		img = img[:, ::-1, :]
		angle = -angle
	return img, angle


def random_shift(img, angle):
	h, w = img.shape[:2]
	tx = np.random.uniform(-10, 10)
	ty = np.random.uniform(-10, 10)
	angle += tx * 0.002
	affine_matrix = np.float32([[1, 0, tx],
								[0, 1, ty]])
	img = cv2.warpAffine(img, affine_matrix, (w, h))
	return img, angle


def augment(img, angle):
	img, angle = random_flip(img, angle)
	img, angle = random_shift(img, angle)
	return img, angle


def batch_generator(img_paths, angles, batch_size, is_augment=True):
	assert len(img_paths) == len(angles), 'img_paths and angles must be the same length'
	n = len(img_paths)
	batch_idx = 0
	while 1:
		if batch_idx == 0:
			idx_array = np.random.permutation(n)
		current_idx = (batch_idx * batch_size) % n
		img_batch = []
		angle_batch = []
		for idx in idx_array[current_idx:current_idx + batch_size]:
			img_path = img_paths[idx]
			angle = angles[idx]
			img = cv2.imread(img_path)[:, :, ::-1]
			if is_augment:
				img, angle = augment(img, angle)
			img_batch.append(img)
			angle_batch.append(angle)
		if n > current_idx + batch_size:
			batch_idx += 1
		else:
			batch_idx = 0
			idx_array = np.random.permutation(n)	
		yield np.array(img_batch), np.array(angle_batch)


def create_model():
	img_shape = (160, 320, 3)
	base_model = applications.VGG19(weights='imagenet', include_top=False, input_shape=(img_shape))
	for layer in base_model.layers:
		layer.trainable = False

	inp = Input(img_shape)
	x = Cropping2D(cropping=((70, 25), (0, 0)))(inp)
	x = Lambda(lambda x: x / 127.5 - 1)(x)
	x = base_model(x)
	x = Dropout(0.5)(x)
	x = Flatten()(x)
	x = Dense(128, activation='relu')(x)
	x = Dense(64, activation='relu')(x)
	x = Dense(32, activation='relu')(x)
	pred = Dense(1)(x)
	model = Model(input=inp, output=pred)
	return model


def main():
	# load training and validation data
	data_dirs = ['train_data/03', 'train_data/04_reverse']
	img_paths, angles = load_data(data_dirs)
	X_train, X_valid, y_train, y_valid = train_test_split(img_paths, angles, train_size=0.8, shuffle=True)
	print('Number of training data: {}'.format(len(X_train)))
	print('Number of validation data: {}'.format(len(X_valid)))

	# training parameters
	batch_size = 32
	epochs = 20
	steps_per_epoch = np.ceil(len(X_train) / batch_size)
	valid_steps = np.ceil(len(X_valid) / batch_size)
	lr = 0.0001

	# compile model
	model = create_model()
	adam = optimizers.Adam(lr)
	model.compile(loss='mse', optimizer=adam)
	model.summary()	

	# train model
	history = model.fit_generator(
		batch_generator(X_train, y_train, batch_size, is_augment=True),
		steps_per_epoch,
		epochs,
		validation_data=batch_generator(X_valid, y_valid, batch_size, is_augment=False),
		validation_steps=valid_steps,
		verbose=1
	)

	# save model
	model.save('model_20.h5')

	# plot training history
	plt.plot(history.history['loss'], label='Training')
	plt.plot(history.history['val_loss'], label='Validation')
	plt.title('Training history')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.xticks(np.arange(epochs + 1))
	plt.legend()
	if not os.path.exists('figures'):
		os.mkdir('figures')
	plt.savefig(os.path.join('figures', 'training_history_20.jpg'))


if __name__ == '__main__':
	main()
