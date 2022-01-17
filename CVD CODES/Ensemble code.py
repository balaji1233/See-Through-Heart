
V_model = tf.keras.models.load_model('/content/csv_model_heart_failure_prediction (1) (1).h5')

data = pd.read_csv('/content/heart (1).csv')

for i in range(len(data['chol'])):
    if data['chol'][i] < 200:
               data['chol'][i] = 0
    elif 200 <= data['chol'][i] <= 239:
               data['chol'][i] = 1
    else:
               data['chol'][i] = 2

data['age'] = (data['age'] - data['age'].mean()) / data['age'].std()
data['trestbps'] = (data['trestbps'] - data['trestbps'].mean()) / data['trestbps'].std()
data['thalach'] = (data['thalach'] - data['thalach'].mean()) / data['thalach'].std()
data['oldpeak'] = (data['oldpeak'] - data['oldpeak'].mean()) / data['oldpeak'].std()

V_X = np.array(data.drop(['target'], axis=1))
V_y = np.array(data['target'])

data.head()

def label_binarizer(labels, keep_key=0):
    labels = np.array(labels)
    labels = labels == keep_key
    labels = labels.astype('int32')
    return labels



 PCG_y1 = label_binarizer(PCG_y)   

def map_func1(waveform):
    spectro = get_spectrogram(tf.reshape(waveform, (-1, 4000)))
    return spectro

  PCG_X1 = map_func1(PCG_X)


  class EnsembleModel:
  def __init__(self, models, data_for_models, names):
    self.models = models
    self.inputattr = data_for_models
    self.names = names
    self.log = {}
    self.outputs = None
    self.probs = None
  
  def classification_ensemble(self):
    probs = []
    for i in range(len(self.models)):
      li = np.array([self.inputattr[j][i].numpy() for j in range(len(self.inputattr))])
      predictions = self.models[i].predict(li)
      print(predictions, 'w')
      if predictions.shape[1] == 3:
        mean_preds = np.mean(predictions[:, 1:], axis=1)
        predictions = np.concatenate([np.expand_dims(predictions[:, 0], -1), np.expand_dims(mean_preds, -1)], axis=1)
        print(predictions.shape)
      probs.append(predictions)
      print(predictions.shape)

    probs = np.array(probs)
    self.probs = probs

  def score_level_fusion(self, weightage):
    outputs = []
    for i in range(self.probs.shape[0]):
      outputs.append(self.probs[i] * weightage[i])
    
    self.outputs = np.argmax(np.sum(outputs, axis=0), axis=1)
  

  @classmethod
  def sync_data(cls, data_for_models, labels_for_models, n_samples):
    label_log = {}
    for i in range(len(labels_for_models)):
      class_0 = np.argwhere(labels_for_models[i] == 0)[:n_samples]
      class_1 = np.argwhere(labels_for_models[i] == 1)[:n_samples]

      label_log[i] = np.append(class_0, class_1)
    
    patients_data = {}
    for i in range(2 * n_samples):
      patients_data[i] = [label_log[j][i] for j in range(len(labels_for_models))]
    
    data_vals_patients = []
    for key, value in patients_data.items():
      temp_data = []
      ind = 0
      for i in data_for_models:
        if not isinstance(i[ind], tf.Tensor):
          vartensor = tf.Variable(i[ind], dtype='float32')
          temp_data.append(vartensor)
        else:
          temp_data.append(i[ind])
        ind += 1
      data_vals_patients.append(temp_data)


    return data_vals_patients

  @staticmethod
  def label_binarizer(labels, keep_key=0):
    labels = labels == keep_key
    labels = labels.astype('bool')
    return labels
  
  @staticmethod
  def map_func1(waveform):
    spectro = get_spectrogram(tf.reshape(waveform, (-1, 4000)))
    return spectro


data_for_models = EnsembleModel.sync_data([V_X, ECG_X, PCG_X1], [V_y, ECG_y, PCG_y1], 10)


model_final = EnsembleModel([V_model, ECG_model, PCG_model], data_for_models, 'we')

model_final.classification_ensemble()

