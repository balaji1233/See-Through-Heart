import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

model = tf.keras.models.load_model('../input/model-csv/csv_model_heart_failure_prediction (1).h5')

model.summary()

tf.keras.utils.plot_model(model, 'csv_model.png')

data = pd.read_csv('../input/heart-disease-uci/heart.csv')
data.head()

for i in data.columns:
    print(data[i].isnull().value_counts())


plt.bar(('0', '1'), (len(data['target'][data['target'] == 0]), len(data['target'][data['target'] == 1])))

min(data['trestbps'])

for i in range(len(data['chol'])):
    if data['chol'][i] < 200:
               data['chol'][i] = 0
    elif 200 <= data['chol'][i] <= 239:
               data['chol'][i] = 1
    else:
               data['chol'][i] = 2


data.head()

data['age'] = (data['age'] - data['age'].mean()) / data['age'].std()
data['trestbps'] = (data['trestbps'] - data['trestbps'].mean()) / data['trestbps'].std()
data['thalach'] = (data['thalach'] - data['thalach'].mean()) / data['thalach'].std()
data['oldpeak'] = (data['oldpeak'] - data['oldpeak'].mean()) / data['oldpeak'].std()

data.head()

X = np.array(data.drop(['target'], axis=1))
y = np.array(data['target'])

def train_test_split(X, y, test_split_size=0.2):
  rand_row_num = np.random.randint(0, len(y), int(len(y) * test_split_size))

  X_test = np.array([X[i] for i in rand_row_num])
  X_train = np.delete(X, rand_row_num, axis=0)

  y_test = np.array([y[i] for i in rand_row_num])
  y_train = np.delete(y, rand_row_num, axis=0)

  return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = train_test_split(X, y)

X_train.shape, X_test.shape

def build_model():
  inputs = layers.Input((13, ))
  x = layers.Dense(128, activation='relu')(inputs)
  x = layers.Dropout(0.2)(x)

  x = layers.Dense(128, activation='relu')(x)
  x = layers.Dense(128, activation='relu')(x)


  x = layers.Dropout(0.2)(x)

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


model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=METRICS)



lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-6 * 10 ** (epoch / 20))



history = model.fit(X_train, y_train, batch_size=16, epochs=100, callbacks=[lr_scheduler])


plt.figure(figsize=(10, 6))
plt.semilogx(1e-6, 1, 0, 1.75)
plt.plot(history.history['lr'], history.history['loss'])
plt.xlabel('lr')
plt.ylabel('loss')
plt.title('lr vs loss')
plt.show()




early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    verbose=1,
    patience=3,
    mode='max',
    restore_best_weights=True)



model = build_model()
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
model.compile(optimizer=tf.keras.optimizers.Adam(3e-3),
              loss='binary_crossentropy',
              metrics=METRICS)
final_history = model.fit(X_train, y_train, batch_size=16, epochs=15, validation_data=(X_test, y_test),
                          callbacks=[early_stopping])



model.evaluate(X_test, y_test, batch_size=16)


np.unique(y_test, return_counts=True)


def plot_metrics(history):
  metrics = ['loss', 'prc', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color='b', label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color='b', linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()


plot_metrics(final_history)



plot_metrics(final_history)