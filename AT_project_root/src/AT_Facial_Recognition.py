#imports
import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.utils import to_categorical


#Assistance from Copilot (lines 12-67)
from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
data_path = Path("AT_project_root\data")

# sanity checks
if not data_path.exists():
    raise FileNotFoundError(f"data_path does not exist: {data_path}")

# list class subdirectories and counts
class_dirs = [p for p in data_path.iterdir() if p.is_dir()]
classes = [p.name for p in class_dirs]
print("Found classes:", classes)

counts = {p.name: sum(1 for _ in p.rglob('*') if _.is_file()) for p in class_dirs}
print("Image counts per class:", counts)

# helper to get flattened lists of image paths and labels
def get_image_paths_and_labels():
    paths = []
    labels = []
    for class_dir in class_dirs:
        for img in class_dir.rglob('*'):
            if img.is_file():
                paths.append(str(img))
                labels.append(class_dir.name)
    return paths, labels

# optional: create a tf.data.Dataset (useful for training pipeline)
try:
    IMG_SIZE = (224, 224)
    BATCH = 32
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        str(data_path),
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        validation_split=0.2,
        subset="training",
        seed=123,
    )
    ds_val = tf.keras.preprocessing.image_dataset_from_directory(
        str(data_path),
        labels="inferred",
        label_mode="int",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        validation_split=0.2,
        subset="validation",
        seed=123,
    )
    print("Created tf.data datasets. Example batch shape:", next(iter(ds_train))[0].shape)
except Exception as e:
    print("Failed to create tf.data datasets:", e)


#Load data into data frame
#Adapted with help from Mr. Marsh
(paths), (labels) = get_image_paths_and_labels()
#print(paths[:22], labels[:22])  # print first 5 for sanity check
data = pd.DataFrame({'filepath': paths, 'label': labels})
print(data.head())


#Check to make sure there are no NaN
#print("Any NaN Training: ", np.isnan(paths).any())
#print("Any NaN Testing: ", np.isnan(paths).any())

#Here I will tell the model what shape to expect
input_shape = (224, 224, 3)  #224x224 pixels, with 3 color channels

#Reshape the training and testing data
#Help from Copilot (lines 87-103; 105-111)
def load_images_from_paths(paths, target_size):
    imgs = []
    for p in paths:
        img = load_img(p, target_size=target_size)
        arr = img_to_array(img).astype('float32') / 255.0
        imgs.append(arr)
    return np.stack(imgs, axis=0)

# load all filepaths into a numpy array of floats (normalized to [0,1])
x_all = load_images_from_paths(data["filepath"].tolist(), IMG_SIZE)

# for now use all data as x_train (you can split into train/test later)
x_train = x_all
print("Loaded images:", x_train.shape, x_train.dtype)
#x_test = data.astype('float32')/255.0 #Currently using all data for training, will split later

#Convert labels to one-hot, rather than sparse
num_categories = 2
# encode string labels to integer indices, set num_categories accordingly, then one-hot
label_names = sorted(data['label'].unique())
label_to_index = {name: i for i, name in enumerate(label_names)}
y_int = data['label'].map(label_to_index).astype(int).values
num_categories = len(label_names)
y_train = to_categorical(y_int, num_categories)
#y_test = to_categorical(y_test, num_categories) #Currently using all data for training, will split later

#Optional expansion: make num_categories a variable depending on the number of categories scanned

batch_size = 64
num_classes = num_categories
epochs = 3     #maybe switch to 2 epochs

#Build time
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(64, (10, 10), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Conv2D(128, (10, 10), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.35),
        tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.45),
        
        #tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='relu', input_shape=input_shape),
        #tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        #tf.keras.layers.MaxPool2D(),
        #tf.keras.layers.Dropout(.55),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ]
)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2)

model.summary()

#plot out training and validation accuracy and loss
fig, ax = plt.subplots(2, 1)

ax[0].plot(history.history['loss'],  color = 'b', label="Training Loss")
ax[0].plot(history.history['val_loss'],  color = 'r', label="Validation Loss")
legend = ax[0].legend(loc='best', shadow=True)
ax[0].set_title("Loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")

ax[1].plot(history.history['acc'],  color = 'b', label="Training Accuracy")
ax[1].plot(history.history['val_acc'],  color = 'r', label="Validation Accuracy")
legend = ax[1].legend(loc='best', shadow=True)
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")

plt.tight_layout()
plt.show()




model.save("name.keras")