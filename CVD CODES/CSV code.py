import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

data = pd.read_csv('/content/DSN-Project-Exibition/Datasets/CSV Data/cardio_train.csv', sep=';')
data.head()

data['gender'] = data['gender'].replace(1, 0)
data['gender'] = data['gender'].replace(2, 1)

data['cholesterol'] = data['cholesterol'].replace(1, 0)
data['cholesterol'] = data['cholesterol'].replace(2, 1)
data['cholesterol'] = data['cholesterol'].replace(3, 2)

data['age'] = round(data['age'] / 365.0)
data.head()

data['height'] = data['height'] / 100.0
BMI = np.array(data['weight'] / np.square(data['height']))

BMI

for i in range(len(BMI)):
  if BMI[i] < 18.5:
    BMI[i] = 0
  elif BMI[i] >= 18.5 and BMI[i] < 24.9:
    BMI[i] = 1
  elif BMI[i] >= 24.9 and BMI[i] <29.9:
    BMI[i] = 2
  else:
    BMI[i] = 3

BMI = BMI.astype('int32')

data['bmi'] = BMI


data.head()

data['age'] = (data['age'] - data['age'].mean()) / data['age'].std()
data['ap_hi'] = (data['ap_hi'] - data['ap_hi'].mean()) / data['ap_hi'].std()
data['ap_lo'] = (data['ap_lo'] - data['ap_lo'].mean()) / data['ap_lo'].std()

data1 = data.drop(['id', 'weight', 'height'], axis=1)
data1.head()

data.head()

X = np.array(data.drop(['id', 'cardio', 'weight', 'height'], axis=1))
y = np.array(data['cardio'], dtype='int32')

X = X[:-1]
y = y[:-1]

def train_test_split(X, y, test_split_size=0.2):
  rand_row_num = np.random.randint(0, len(y), int(len(y) * test_split_size))

  X_test = np.array([X[i] for i in rand_row_num])
  X_train = np.delete(X, rand_row_num, axis=0)

  y_test = np.array([y[i] for i in rand_row_num])
  y_train = np.delete(y, rand_row_num, axis=0)

  return X_train, y_train, X_test, y_test

  X_train, y_train, X_test, y_test = train_test_split(X, y)

 def build_model():
  inputs = layers.Input((10, ))
  x = layers.Dense(1024, activation='relu')(inputs)
  # x = layers.Dense(128, activation='relu')(x)
  # x = layers.Dense(128, activation='relu')(x)
  x = layers.Dropout(0.1)(x)

  outputs = layers.Dense(1, activation='sigmoid')(x)

  model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
  return model

  model = build_model()

 model.summary()

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

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='auc', 
    verbose=1,
    patience=5,
    mode='max',
    restore_best_weights=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
              loss='binary_crossentropy',
              metrics=METRICS)


history = model.fit(X_train, y_train, batch_size=128, epochs=30, validation_data=(X_test, y_test), callbacks=[early_stopping])

plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(history.history['accuracy'])), history.history['loss'], color='b', label='training_loss')
plt.plot(np.arange(len(history.history['accuracy'])), history.history['val_loss'], color='r', label='val_loss')
plt.legend()
plt.show()

s = history.history['tp'][-1] + history.history['fp'][-1] + history.history['tn'][-1] + history.history['fn'][-1]

(history.history['tp'][-1] + history.history['tn'][-1]) / s

history.history['fn'][-1] / s

history.history['fp'][-1] / s

model.evaluate(X_test, y_test, batch_size=64)
















