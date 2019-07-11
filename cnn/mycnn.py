"""
Simple CNN-based solution to Summer School 2019
for dirty/cleaned bowls classification

Valery Yakovlev
"""

import os
import pandas as pd
import re
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Some default CNN here

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Online augmentation for train data

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('superbowllsh/train',
                                                 target_size=(256, 256),
                                                 batch_size=32,
                                                 class_mode='binary',
                                                 subset='training')

my_labels = (training_set.class_indices)

validation_set = train_datagen.flow_from_directory('superbowllsh/train',
                                                   target_size=(256, 256),
                                                   batch_size=32,
                                                   class_mode='binary',
                                                   subset='validation')

# load test data (test images itself should be in sub-folder(!))
test_set = test_datagen.flow_from_directory("superbowllsh/test",
                                            target_size=(256, 256),
                                            batch_size=1,
                                            class_mode=None,
                                            shuffle=False,
                                            seed=42)

batch_size = 2

print('validation steps ', validation_set.samples)

# fit the network

classifier.fit_generator(training_set,
                         steps_per_epoch=training_set.samples // batch_size,
                         epochs=6,
                         validation_data=validation_set,
                         validation_steps=validation_set.samples // batch_size)

# this can be useful but currently not used

loss = classifier.evaluate_generator(generator=validation_set,
                                     steps=validation_set.samples // batch_size)

# predict test data

STEP_SIZE_TEST = test_set.n // test_set.batch_size
test_set.reset()
pred = classifier.predict_generator(test_set,
                                    steps=STEP_SIZE_TEST,
                                    verbose=1)
my_classes = []

# Didn't know how to convert probability to index (argmax didn't worked for me)

for _ in pred:
    if _ > 0.5:
        my_classes.append(1)
    else:
        my_classes.append(0)

print('predicted : ', my_classes)

ffflabels = training_set.class_indices

labels = dict((v, k) for k, v in ffflabels.items())
predictions = [labels[k] for k in my_classes]
filenames = test_set.filenames

# Leave only digits as image indices
my_filenames = []
for _ in filenames:
    my_filenames.append(re.sub("[^0-9]", "", _))

results = pd.DataFrame({"id": my_filenames,
                        "label": predictions})

results.to_csv("results.csv", index=False)
