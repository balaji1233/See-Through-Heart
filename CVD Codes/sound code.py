
  
 


import zipfile
import glob
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from matplotlib import style
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

!wget http://www.peterjbentley.com/heartchallenge/wav/Btraining_normal.zip


!wget http://www.peterjbentley.com/heartchallenge/wav/Btraining_murmur.zip


!wget http://www.peterjbentley.com/heartchallenge/wav/Btraining_extrasystole.zip


!wget http://www.peterjbentley.com/heartchallenge/wav/Bunlabelledtest.zip


if not os.path.exists('/content/SoundData'):

  _root_dir = '/content/SoundData'
  !mkdir {_root_dir}

  _train_dir = os.path.join(_root_dir, 'train')
  _test_dir = os.path.join(_root_dir, 'test')
  !mkdir {_train_dir}
  !mkdir {_test_dir}

  with zipfile.ZipFile('/content/Btraining_extrasystole.zip', 'r') as zip_ref:
    zip_ref.extractall(_train_dir)
  
  with zipfile.ZipFile('/content/Btraining_murmur.zip', 'r') as zip_ref:
    zip_ref.extractall(_train_dir)
  
  with zipfile.ZipFile('/content/Btraining_normal.zip', 'r') as zip_ref:
    zip_ref.extractall(_train_dir)

  with zipfile.ZipFile('/content/Bunlabelledtest.zip', 'r') as zip_ref:
    zip_ref.extractall(_test_dir)

  print('Files created successfully')
  
else:
  print('Files already exists!')

extrasystole_list = glob.glob('/content/SoundData/train/Btraining_extrastole/*')

murmur_list = glob.glob('/content/SoundData/train/Btraining_murmur/*')
murmur_list.remove('/content/SoundData/train/Btraining_murmur/Btraining_noisymurmur')
noisy_murmur_list = glob.glob('/content/SoundData/train/Btraining_murmur/Btraining_noisymurmur/*')
murmur_list += noisy_murmur_list

normal_list = glob.glob('/content/SoundData/train/Training B Normal/*')
normal_list.remove('/content/SoundData/train/Training B Normal/Btraining_noisynormal')
noisy_normal_list = glob.glob('/content/SoundData/train/Training B Normal/Btraining_noisynormal/*')
normal_list += noisy_murmur_list


extrasystole_samples = []
for i in tqdm(extrasystole_list):
  audio, _ = tf.audio.decode_wav(tf.io.read_file(i), desired_channels=1)
  extrasystole_samples.append(audio)

max_len = 4000#max([len(i) for i in extrasystole_samples])
extrasystole_samples = tf.convert_to_tensor(pad_sequences(extrasystole_samples, maxlen=max_len, padding='pre', dtype='float32'))


murmur_samples = []
for i in tqdm(murmur_list):
  audio, _ = tf.audio.decode_wav(tf.io.read_file(i), desired_channels=1)
  murmur_samples.append(audio)

max_len = 4000#max([len(i) for i in extrasystole_samples])
murmur_samples = tf.convert_to_tensor(pad_sequences(murmur_samples, maxlen=max_len, padding='pre', dtype='float32'))

normal_samples = []
for i in tqdm(normal_list):
  audio, _ = tf.audio.decode_wav(tf.io.read_file(i), desired_channels=1)
  normal_samples.append(audio)

print(normal_samples[0])
max_len = 4000#max([len(i) for i in extrasystole_samples])
normal_samples = tf.convert_to_tensor(pad_sequences(normal_samples, maxlen=max_len, padding='pre', dtype='float32'))
normal_samples[0]


X = tf.concat([extrasystole_samples, murmur_samples, normal_samples], axis=0)
y = np.zeros((X.shape[0], ), dtype='int32')
y[len(extrasystole_samples):len(extrasystole_samples) + len(murmur_samples)] = 1
y[len(extrasystole_samples) + len(murmur_samples):] = 2
y = tf.convert_to_tensor(y, dtype='int32')
X.shape
y.shape


a = tf.stack([X[0], X[1]])
a.shape


X[0]


mel = get_spectrogram(tf.reshape(a, (2, 1, -1)))


fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(4000)
axes[0].plot(timescale, tf.squeeze(a).numpy())
axes[0].set_title('Waveform')
axes[0].set_xlim([0, 4000])

plot_spectrogram(mel.numpy(), axes[1])
axes[1].set_title('Spectrogram')
axes[1].set_xlim([0, 3500])
plt.show()

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    # assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec.reshape(129, -1))
  m = ax.pcolormesh(X, Y, log_spec.reshape(129, -1))
  print(m)


plt.imshow(np.squeeze(mel.numpy()))

def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=129)
  print(spectrogram.shape, 'SDS')
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = tf.squeeze(spectrogram)
  spectrogram = tf.expand_dims(spectrogram, -1)

  return spectrogram

def map_func(waveform, label):
  
  spectro = get_spectrogram(tf.reshape(waveform, (1, 4000)))
  return spectro, label

 def input_pipeline(X, y, batch_size, shuffle_buffer, val_split=0.2):
  split_index = int(len(X) * (1 - val_split))
  ds = tf.data.Dataset.from_tensor_slices((X, y))
  ds = ds.map(map_func)
  ds = ds.shuffle(shuffle_buffer)
  train_ds = ds.take(split_index)
  val_ds = ds.skip(split_index)
  return (train_ds.batch(batch_size).prefetch(1), val_ds.batch(batch_size).prefetch(1))
  
import tensorflow.keras.utils as ku
y_ = ku.to_categorical(y, num_classes=3)
train_ds, val_ds = input_pipeline(X, y_, 16, 370)
train_ds, val_ds


def build_model():
  inputs = layers.Input((30, 129, 1))
  x = layers.Conv2D(128, 3, activation='relu')(inputs)
  x = layers.MaxPooling2D(2)(x)
  x = layers.Conv2D(128, 3, activation='relu')(inputs)
  x = layers.MaxPooling2D(2)(x)
  x = layers.Flatten()(x)
  x = layers.Dropout(0.3)(x)
  x = layers.Dense(128, activation='relu')(x)
  x = layers.Dense(128, activation='relu')(x)
  x = layers.Dropout(0.2)(x)

  outputs = layers.Dense(3, activation='softmax')(x)

  model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

  return model
model1 = build_model()


model1.summary()


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
    monitor='recall', patience=5, mode='max', restore_best_weights=True, verbose=1
)

model1.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=METRICS)


history1 = model1.fit(train_ds, epochs=100, validation_data=val_ds, callbacks=[early_stopping])

PCG_model = model1
PCG_X = X
PCG_y = y

def map_func1(waveform):
  
  spectro = get_spectrogram(tf.reshape(waveform, (1, 4000)))
  return spectro
a = map_func1(X[0])
p = model1.predict(tf.expand_dims(a, 0))

print(p.shape)


epochs = np.arange(len(history1.history['loss']))
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
axes[0].plot(epochs, history1.history['loss'], color='r', label='training_loss')
axes[0].plot(epochs, history1.history['val_loss'], color='b', label='validation_loss')
axes[0].legend()

axes[1].plot(epochs, history1.history['accuracy'], color='r', label='training_acuuracy')
axes[1].plot(epochs, history1.history['val_accuracy'], color='b', label='validation_accuracy')
axes[1].legend()

plt.show()

from matplotlib import style
style.use('ggplot')

labels = ['', 'Loss', 'Recall', 'Accuracy', 'Precision', 'ROC-AUC']
lstm = [history.history['loss'][-1], history.history['recall'][-1], history.history['accuracy'][-1], history.history['precision'][-1], history.history['prc'][-1]]
stft = [history1.history['loss'][-1], history1.history['recall'][-1], history1.history['accuracy'][-1], history1.history['precision'][-1], history1.history['prc'][-1]]

x = np.arange(len(labels) - 1)  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(15, 10))
rects1 = ax.bar(x - width/2, lstm, width, label='1D-CONV + Bi-LSTM', color='g', alpha=0.7)#, tick_label=labels)
rects2 = ax.bar(x + width/2, stft, width, label='STFT + 2D-CONV', color='b', alpha=0.7)#, tick_label=labels)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by various metrics')
# ax.set_xticks(x, labels)
ax.set_xticklabels(labels)
ax.legend()


fig.tight_layout()

plt.show()


model.save('PCG.h5')

 










