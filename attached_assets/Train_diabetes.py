import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from tensorflow import keras
from kerastuner.tuners import RandomSearch
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
from tensorflow.keras.utils import to_categorical
from keras.layers import LSTM, Bidirectional
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv('' \
'diabetes_risk_prediction_dataset.csv')  # Update path if needed

# Print columns to verify
print("Columns in dataset:", df.columns)

# Label Encoding (applies to all columns)
l = LabelEncoder()
df = df.apply(l.fit_transform)

# Use "class" as target column
X = df.drop('class', axis=1)
y = df['class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model using Hyperparameter tuning
def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units1', min_value=32, max_value=512, step=32),
                    activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(units=hp.Int('units2', min_value=32, max_value=512, step=32),
                    activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameter tuning using RandomSearch
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='random_search_dir',
    project_name='diabetes_classification'
)

tuner.search_space_summary()
tuner.search(X_train, y_train, epochs=10, validation_split=0.2)
tuner.results_summary()

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Evaluate the model
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

# Predict and show confusion matrix
y_pred = (best_model.predict(X_test) > 0.5).astype("int32")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Assuming 'model' is your trained Keras/TensorFlow model
best_model.save('trained_model.h5')  # Saves the model as a .h5 file

