!git clone https://github.com/mayureshagashe2105/DSN-Project-Exibition.git

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import glob
from tqdm.notebook import tqdm
import os

print(tf.image.decode_jpeg(tf.io.read_file(MI_patients[0])).shape)
plt.figure(figsize=(20, 20))
img = tf.image.resize(tf.image.decode_and_crop_jpeg(tf.io.read_file(MI_patients[0]), [400, 145, 1060, 1995], channels=1), (500, 500))
plt.imshow(tf.reshape(img, (500, 500)))

MI_patients_list = []
for image in tqdm(MI_patients):
  MI_patients_list.append(tf.image.resize(tf.image.decode_and_crop_jpeg(tf.io.read_file(image), [400, 145, 1060, 1995], channels=1), (500, 500)))
MI_patients_list = tf.stack(MI_patients_list, 0)

MI_history_patients_list = []
for image in tqdm(MI_history_patients):
  MI_history_patients_list.append(tf.image.resize(tf.image.decode_and_crop_jpeg(tf.io.read_file(image), [400, 145, 1060, 1995], channels=1), (500, 500)))
MI_history_patients_list = tf.stack(MI_history_patients_list, 0)

abnormal_ECG_patients_list = []
for image in tqdm(abnormal_ECG_patients):
  abnormal_ECG_patients_list.append(tf.image.resize(tf.image.decode_and_crop_jpeg(tf.io.read_file(image), [400, 145, 1060, 1995], channels=1), (500, 500)))
abnormal_ECG_patients_list = tf.stack(abnormal_ECG_patients_list, 0)

class_1_images = tf.concat([MI_patients_list, MI_history_patients_list, abnormal_ECG_patients_list], axis=0)
class_1_images.shape

normal_ECG_people_list = []
for image in tqdm(normal_ECG_patients):
  normal_ECG_people_list.append(tf.image.resize(tf.image.decode_and_crop_jpeg(tf.io.read_file(image), [400, 145, 1060, 1995], channels=1), (500, 500)))
normal_ECG_people_list = tf.stack(normal_ECG_people_list, 0)

class_0_images = normal_ECG_people_list
class_0_images.shape

X = tf.concat([class_0_images, class_1_images], axis=0)
X.shape

labels = np.zeros((859+823,), dtype='int32')
labels[:859] = 0
labels[859:] = 1
labels = tf.Variable(labels, dtype='int32')

def input_pipeline(X, y, batch_size, shuffle_buffer, val_split=0.2):
  split_index = int(len(X) * (1 - val_split))
  ds = tf.data.Dataset.from_tensor_slices((X / 255.0, y))
  ds = ds.shuffle(shuffle_buffer)
  train_ds = ds.take(split_index)
  val_ds = ds.skip(split_index)
  return (train_ds.batch(batch_size).prefetch(1), val_ds.batch(batch_size).prefetch(1))
  

train_ds, val_ds = input_pipeline(X, labels, 32, 1682)
train_ds, val_ds

def build_model():
  inputs = layers.Input((500, 500, 1))
  x = layers.Conv2D(32, (5, 5) , activation='relu')(inputs)
  x = layers.MaxPooling2D((2, 2))(x)
  x = layers.Conv2D(32, (3, 3) , activation='relu')(x)
  x = layers.MaxPooling2D((2, 2))(x)
  x = layers.Conv2D(32, (3, 3) , activation='relu')(x)
  x = layers.MaxPooling2D((2, 2))(x)
  x = layers.Flatten()(x)
  x = layers.Dense(128, activation='relu')(x)
  x = layers.Dropout(0.3)(x)
  outputs = layers.Dense(1, activation='sigmoid')(x)

  model = tf.keras.Model(inputs=[inputs], outputs=[outputs], name='ECG_Classifier')
  return model

model = build_model()
model.summary()


tf.keras.utils.plot_model(model, 'ECG_Classifier.png')

METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=METRICS)


 early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='recall', 
    verbose=1,
    patience=3,
    mode='max',
    restore_best_weights=True)

histroy = model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=[early_stopping])



ECG_model = model
ECG_X = X
ECG_y = labels

model.evaluate(val_ds)

model.save('ECG_Classifier1.h5')

history = histroy

epochs = np.arange(len(history.history['loss']))
fig, axes = plt.subplots(1, 3, figsize=(20, 10))
axes[0].plot(epochs, histroy.history['loss'], color='r', label='training_loss')
axes[0].plot(epochs, histroy.history['val_loss'], color='b', label='validation_loss')
axes[0].legend()

axes[1].plot(epochs, histroy.history['accuracy'], color='r', label='training_acuuracy')
axes[1].plot(epochs, histroy.history['val_accuracy'], color='b', label='validation_accuracy')
axes[1].legend()

axes[2].plot(histroy.history['precision'], histroy.history['recall'], color='r', label='training_precision_vs_loss')
axes[2].plot(histroy.history['val_precision'], histroy.history['val_recall'], color='b', label='validation_precision_vs_loss')
axes[2].legend()

plt.show()


def visualize_interm_convs(input_image, model):
  successive_outputs = [layer.output for layer in model.layers[1:]]
  visualization_model = tf.keras.Model(inputs=model.input, outputs=successive_outputs, name='visualization_model')

  if len(input_image.shape) != 4:
    input_image = tf.expand_dims(input_image, 0)
  
  img = input_image / 255.0

  predictions = visualization_model.predict(img)

  layer_names = [layer.name for layer in model.layers[1:]]

  for layer_name, feature_map in zip(layer_names, predictions):
    if len(feature_map.shape) == 4:
      n_features = feature_map.shape[-1]
      size = feature_map.shape[1]

      disp_grid = np.zeros((size, size * n_features))

      for i in range(n_features):
        disp_grid[:, i * size:(i + 1) * size] = feature_map[0, :, :, i]


      scale = 20. / n_features
      plt.figure( figsize=(scale * n_features, scale * 2) )
      plt.title ( layer_name )
      plt.grid  ( False )
      plt.imshow( disp_grid, aspect='auto', cmap='viridis' )



visualize_interm_convs(X[0], model)

visualize_interm_convs(X[-1], model)

11image = plt.imread('/content/DSN-Project-Exibition/Datasets/ECG images/Normal Person ECG Images (859)/Normal (1).jpg')

image = image.astype('float32')

image.shape

plt.figure(figsize=(20, 20))
plt.imshow(image)


img = tf.image.crop_to_bounding_box(tf.expand_dims(image, 0), 300, 0, image.shape[0] - 400, 2213)
img.shape

plt.imshow(tf.reshape(img, (-1, 2213, 3)))


y = layers.Cropping2D(((350, 150), (0, 0)))(tf.expand_dims(image, 0))


plt.figure(figsize=(20, 20))
plt.imshow(tf.reshape(y, (-1, image.shape[1], 3)))




