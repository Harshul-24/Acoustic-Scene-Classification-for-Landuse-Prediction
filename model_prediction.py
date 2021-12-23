from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

import os
import numpy as np
from tensorflow import keras


test = ImageDataGenerator()

test_data = test.flow_from_directory("./image_datasets/validation", shuffle = True, seed = 42, target_size=(512, 316))

model = keras.models.load_model("./model_save_state/model1")

dir_path = './image_datasets/validation/Industrial'

for i in os.listdir(dir_path):
	img = image.load_img(dir_path + "//" + i, target_size = (316, 316))
	plt.imshow(img)
	plt.show()

	img_arr = image.img_to_array(img)
	img_arr_expanded = np.expand_dims(img_arr, axis=0)
	images =  np.vstack([img_arr_expanded])

	model.predict(images)
	val = model.predict(images)
	if val[0][0] == 1:
		print("Commercial")

	elif val[0][1] == 1:
		print("Industrial")

	elif val[0][2] == 1:
		print("Residential")
