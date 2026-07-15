import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

BUCKET ="gs://bucket-for-kpf-project"
MODEL_SAVE =os.getenv("AIP_MODEL_DIR")
#DATA_URI=os.getenv("AIP_TRAINING_DATA_URI")
DATA_URI = "gs://bucket-for-kpf-project/UCI_Credit_Card.csv"
df = pd.read_csv(DATA_URI)

train_dataset = df.sample(frac=0.8,random_state=0)
test_dataset = df.drop(train_dataset.index)
train_stats = train_dataset.describe()
train_stats.pop("default.payment.next.month")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('default.payment.next.month')
test_labels = test_dataset.pop('default.payment.next.month')

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():

  model = Sequential()
  model.add(Dense(256, activation='relu', input_shape=(train_dataset.shape[1],)))  # Input layer
  model.add(Dense(256, activation='relu'))  # Hidden layer
  model.add(Dense(1, activation='sigmoid'))  # Output layer (sigmoid for binary classification)

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  return model

model = build_model()

model.fit(normed_train_data, train_labels, 
                    epochs=10, validation_split = 0.2)

model.export(MODEL_SAVE)
model.save('gs://custom-trin-bucket/custom_model_test1.keras')