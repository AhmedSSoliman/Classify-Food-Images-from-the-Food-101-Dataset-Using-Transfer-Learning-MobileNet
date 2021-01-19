

import tensorflow as tf
import keras
from keras.layers import Dense
from keras import Model
from keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

### English Dataset

train_path = './Dataset/train'
valid_path = './Dataset/val'
test_path = './Dataset/test'

"""
### Arabic Dataset

train_path = './Arabic-Food-Dataset/train'
valid_path = './Arabic-Food-Dataset/val'
test_path = './Arabic-Food-Dataset/test'
"""
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)

train_batches.classes
train_batches.class_indices


mobile = keras.applications.mobilenet.MobileNet()
mobile.summary()



x = mobile.layers[-6].output

"""
### For English dataset as it has 101 classes
output = Dense(units=101, activation='softmax')(x)
"""

### For Arabic dataset as it has 20 classes
output = Dense(units=20, activation='softmax')(x)


model = Model(inputs=mobile.input, outputs=output)

for layer in model.layers[:-5]:
    layer.trainable = False

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=train_batches,
            steps_per_epoch=len(train_batches),
            validation_data=valid_batches,
            validation_steps=len(valid_batches),
            epochs=30,
            verbose=2
)


### For saving English model 
model.save('./English-FoodModel.h5')

"""
### For saving Arabic model
model.save('./Arabic-FoodModel-30Epochs.h5')
"""

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



"""
test_labels = test_batches.classes
predictions = model.predict(x=test_batches, verbose=0)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


test_batches.class_indices

cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))



cm_plot_labels = ['Apple pie','Baby back ribs','Baklava','Beef carpaccio','Beef tartare','Beet salad','Beignets','Bibimbap','Bread pudding','Breakfast burrito','Bruschetta','Caesar salad','Cannoli','Caprese salad','Carrot cake','Ceviche','Cheesecake','Cheese plate','Chicken curry','Chicken quesadilla','Chicken wings','Chocolate cake','Chocolate mousse','Churros','Clam chowder','Club sandwich','Crab cakes','Creme brulee','Croque madame','Cup cakes','Deviled eggs','Donuts','Dumplings','Edamame','Eggs benedict','Escargots','Falafel','Filet mignon','Fish and chips','Foie gras','French fries','French onion soup','French toast','Fried calamari','Fried rice','Frozen yogurt','Garlic bread','Gnocchi','Greek salad','Grilled cheese sandwich','Grilled salmon','Guacamole','Gyoza','Hamburger','Hot and sour soup','Hot dog','Huevos rancheros','Hummus','Ice cream','Lasagna','Lobster bisque','Lobster roll sandwich','Macaroni and cheese','Macarons','Miso soup','Mussels','Nachos','Omelette','Onion rings','Oysters','Pad thai','Paella','Pancakes','Panna cotta','Peking duck','Pho','Pizza','Pork chop','Poutine','Prime rib','Pulled pork sandwich','Ramen','Ravioli','Red velvet cake','Risotto','Samosa','Sashimi','Scallops','Seaweed salad','Shrimp and grits','Spaghetti bolognese','Spaghetti carbonara','Spring rolls','Steak','Strawberry shortcake','Sushi','Tacos','Takoyaki','Tiramisu','Tuna tartare','Waffles',]
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
"""

"""
history=classifier.fit_generator(training_set,
                         samples_per_epoch = 1098,
                         nb_epoch = 100,
                         validation_data = test_set,
                         nb_val_samples =273 )
"""

