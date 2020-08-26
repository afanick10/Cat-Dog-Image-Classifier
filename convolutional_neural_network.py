# import tensorflow and keras packages
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# other packages to import
import os
import numpy as np
import matplotlib.pyplot as plt

# function used for outputting images from dataset
def plotImages(images_arr, probabilities = False):
    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5, len(images_arr) * 3))
    if probabilities is False:
        for img, ax in zip(images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
    else:
        for img, probability, ax in zip(images_arr, probabilities, axes):
            ax.imshow(img)
            ax.axis('off')
            if probability > 0.5:
                ax.set_title("%.2f" % (probability * 100) + "%" + " dog")
            else:
                ax.set_title("%.2f" % ((1 - probability) * 100) + "%" + " cat")
    plt.show()

# url to zip file
URL = 'https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip'

# gets path to the data directory
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=URL, extract=True)

# sets path to the data
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs')

# creates paths for training, validation, and testing data
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

# gets number of files in each data directory
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = len(os.listdir(test_dir))

# parameters for creating neural network
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WEIGHT = 150

# images generators for training, validation, and testing data
train_image_generator = ImageDataGenerator(train_dir, rescale=1./255)
validation_image_generator = ImageDataGenerator(validation_dir, rescale=1./255)
test_image_generator = ImageDataGenerator(test_dir, rescale=1./255)

# creates new directory for testing data since there isn't one initially
new_dir = os.path.join(test_dir, 'data')
if not os.path.isdir(new_dir):
    os.mkdir(new_dir)

# moves all of the testing data into the new directory
for current_file in os.listdir(test_dir):
    if os.path.isfile(test_dir + '/' + current_file):
        original_file = open(test_dir + '/' + current_file, 'rb')
        file_copy = open(new_dir + '/' + current_file, 'wb')
        file_copy.write(original_file.read())
        original_file.close()
        file_copy.close()

# data generators for training, validation, and testing data
train_data_gen = train_image_generator.flow_from_directory(
                        train_dir,
                        batch_size=batch_size,
                        target_size=(IMG_HEIGHT, IMG_WEIGHT),
                        class_mode='binary'
)

val_data_gen = validation_image_generator.flow_from_directory(
                        validation_dir,
                        batch_size=batch_size,
                        target_size=(IMG_HEIGHT, IMG_WEIGHT),
                        class_mode='binary'
)

test_data_gen = test_image_generator.flow_from_directory(
                        test_dir,
                        batch_size=batch_size,
                        target_size=(IMG_HEIGHT, IMG_WEIGHT),
                        class_mode=None,
                        shuffle=False
)

# add random transformations to training images
train_image_generator = ImageDataGenerator(
    train_dir,
    rescale=1./255,
    zoom_range=0.6,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=0.7,
    channel_shift_range=0.2
)

train_data_gen = train_image_generator.flow_from_directory(
    train_dir,
    batch_size=batch_size,
    target_size=(IMG_HEIGHT, IMG_WEIGHT),
    class_mode='binary'
)

# creates sequential model (a neural network) with layers of 32 neurons, 64 neurons, and 128 neurons
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WEIGHT, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(1, activation='relu')) # adds a dense layer that allocates one neuron for the output

# compiles the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy']
)

print(model.summary())

# fits the training data to the model
history = model.fit(train_data_gen,
                    steps_per_epoch=15,
                    epochs=epochs,
                    validation_data=val_data_gen,
                    validation_steps=2
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# plots the training and validation accuracies of the model
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# plots the training and validation losses of the model
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# model predicts whether each image is of a cat or a dog outputs the percentage of images correctly identified
probabilities = list(model.predict(test_data_gen).flatten())
test_images = test_data_gen[0]
plotImages(test_images, probabilities=probabilities)

answers = [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
           1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
           1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
           1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
           0, 0, 0, 0, 0, 0]

correct = 0

for probability, answer in zip(probabilities, answers):
    if round(probability) == answer:
        correct += 1

percentage_identified = (correct / len(answers))

passed_challenge = percentage_identified > 0.63

print(f"Your model correctly identified {round(percentage_identified, 2) * 100}% of the images of cats and dogs.")

if passed_challenge:
    print("You passed the challenge!")
else:
    print("You haven't passed yet. Your model should identify at least 63% " + "of the images. Keep trying. You will get it!")
