import numpy as np
import glob
from PIL import Image
import h5py

ind_images = np.array(glob.glob("./image_datasets/Industrial/*"))[:4000]
comm_images = np.array(glob.glob("./image_datasets/Commercial/*"))[:2200]
res_images = np.array(glob.glob("./image_datasets/Residential/*"))[:4000]


ind_images_y = [0 for _ in range(len(ind_images))]
comm_images_y = [1 for _ in range(len(comm_images))]
res_images_y = [2 for _ in range(len(res_images))]


train_images_path = np.concatenate((ind_images[:3000], comm_images[:1700], res_images[:3000]), axis=0)
test_images_path = np.concatenate((ind_images[3000:], comm_images[1700:], res_images[3000:]), axis=0)


train_x = []
test_x = []


for fname in train_images_path:
	print("Current audio processed: \n", fname)
	arr = Image.open(fname)
	train_x.append(np.array(arr))


for fname in test_images_path:
	print("Current audio processed: \n", fname)
	arr = Image.open(fname)
	test_x.append(np.array(arr))


if len(train_x) != 0 and len(test_x) != 0:
	print("All arrays processed....")


train_y = np.concatenate((ind_images_y[:3000], comm_images_y[:1700], res_images_y[:3000]), axis=0)
test_y = np.concatenate((ind_images_y[3000:], comm_images_y[1700:], res_images_y[3000:]), axis=0)

train_x = np.asarray(train_x)[:, :, :, 0:3]
test_x = np.asarray(test_x)[:, :, :, 0:3]



print(train_x.shape)		#(7700, 316, 512, 4)
print(test_x.shape)		#(2500, 316, 512, 4)
print(train_y.shape)		#(7700,)
print(test_y.shape)		#(2500,)

# Save model data into hdf5 dataset

f1 = h5py.File("model_data.hdf5", "w")

dset1 = f1.create_dataset("train_x", train_x.shape, dtype='i', data=train_x)

dset2 = f1.create_dataset("test_x", test_x.shape, dtype='i', data=test_x)

dset3 = f1.create_dataset("train_y", train_y.shape, dtype='i', data=train_y)

dset4 = f1.create_dataset("test_y", test_y.shape, dtype='i', data=test_y)

f1.close()

# dset1 = f1.create_dataset("train_x", ())

#with open('model_data.npy', 'wb') as f:
#	print("Saving Training X data...")
#	np.save(f, np.array(train_x))			# training set of features/images
#	print("Saving Testing X data...")	
#	np.save(f, np.array(test_x))			# testing set of features/images
#	print("Saving Training Y data...")	
#	np.save(f, train_y)				# training set of output predictions 
#	print("Saving testing Y data...")	
#	np.save(f, test_y)				# testing set of output predictions
