import tensorflow as tf
import glob
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import ssl
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
ssl._create_default_https_context = ssl._create_unverified_context

# Normalize pixel values to be between 0 and 1


train = ImageDataGenerator(rescale=1/255.0)
validation = ImageDataGenerator(rescale=1/255.0)

train_set = train.flow_from_directory("image_datasets/train/", target_size=(316, 316), batch_size= 32, class_mode="categorical", shuffle = True, seed=42)

validation_set = validation.flow_from_directory("image_datasets/validation/", target_size=(316, 316), batch_size=32, class_mode="categorical", shuffle = True, seed=42)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(316, 316, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_set, epochs=1, 
                    validation_data=validation_set)
                    
                    
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(validation_set, verbose=2)     



print(model.predict(validation_set[0][0]))    


model.save("./model_save_state/model1")
   
