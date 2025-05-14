#!/usr/bin/env python
# coding: utf-8

# # Anomaly Detection in ECG Signals: Identifying Abnormal Heart Patterns using Deep Learning

# In[16]:


#Import all the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import plotly.express as px
from scipy.signal import medfilt, butter, filtfilt
import pywt
from sklearn.model_selection import train_test_split
import scipy.signal
from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score



# ## Data Extraction

# In[17]:


#import the dataset from tensorflow datasets
df = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)


# ## Exploratory Data Analysis

# In[13]:


df.head()


# In[11]:


df.shape


# In[9]:


df.columns


# In[10]:


df[140].unique()


# In[3]:


df[140].value_counts() # 1 is normal & 0 is abnormal ecg


# In[4]:


df.info()


# In[8]:


df.dtypes


# In[6]:


#looking at missing values for each column
df.isna().sum()


# In[7]:


#missing values for entire dataset
df.isna().sum().sum()


# In[18]:


#plot graphs of normal and abnormal ECG to visualise the trends 
abnormal = df[df.loc[:,140] ==0][:10]
normal = df[df.loc[:,140] ==1][:10]
# Create the figure 
fig = go.Figure()
#create a list to display only a single legend 
leg  = [False] * abnormal.shape[0]
leg[0] = True
# Plot training and validation error

for i in range(abnormal.shape[0]): 
    fig.add_trace(go.Scatter( x=np.arange(abnormal.shape[1]),y=abnormal.iloc[i,:],name='Abnormal ECG', mode='lines',  marker_color='rgba(255, 0, 0, 0.9)', showlegend= leg[i]))

for j in range(normal.shape[0]): 
    fig.add_trace(go.Scatter( x=np.arange(normal.shape[1]),y=normal.iloc[j,:],name='Normal ECG',  mode='lines',  marker_color='rgba(0, 255, 0, 1)', showlegend= leg[j]))



fig.update_layout(xaxis_title="time", yaxis_title="Signal", title= {'text': 'Difference between different ECG', 'xanchor': 'center', 'yanchor': 'top', 'x':0.5} , bargap=0,)
fig.update_traces(opacity=0.5)
fig.show()


# ## Data preprocessing

# In[19]:


# split the data into labels and features 
ecg_data = df.iloc[:,:-1]
labels = df.iloc[:,-1]

# Normalize the data between -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))
ecg_data = scaler.fit_transform(ecg_data)


# In[20]:


#filtering the raw signal 

# Plot original ECG signal
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(ecg_data.shape[0]), y=ecg_data[30], mode='lines', name='Original ECG signal'))

# Median filtering
ecg_medfilt = medfilt(ecg_data, kernel_size=3)

# Low-pass filtering
lowcut = 0.05
highcut = 20.0
nyquist = 0.5 * 360.0
low = lowcut / nyquist
high = highcut / nyquist
b, a = butter(4, [low, high], btype='band')
ecg_lowpass = filtfilt(b, a, ecg_data)

# Wavelet filtering
coeffs = pywt.wavedec(ecg_data, 'db4', level=1)
threshold = np.std(coeffs[-1]) * np.sqrt(2*np.log(len(ecg_data)))
coeffs[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in coeffs[1:])
ecg_wavelet = pywt.waverec(coeffs, 'db4')

# Plot filtered ECG signals
fig.add_trace(go.Scatter(x=np.arange(ecg_medfilt.shape[0]), y=ecg_medfilt[30], mode='lines', name='Median filtered ECG signal'))
fig.add_trace(go.Scatter(x=np.arange(ecg_lowpass.shape[0]), y=ecg_lowpass[30], mode='lines', name='Low-pass filtered ECG signal'))
fig.add_trace(go.Scatter(x=np.arange(ecg_wavelet.shape[0]), y=ecg_wavelet[30], mode='lines', name='Wavelet filtered ECG signal'))
fig.show()


# In[21]:


#select the optimum filtering using the below code
def pad_data(original_data,filtered_data):
  # Calculate the difference in length between the original data and filtered data
  diff = original_data.shape[1] - filtered_data.shape[1]
    # pad the shorter array with zeroes
  if diff > 0:
          # Create an array of zeros with the same shape as the original data
      padding = np.zeros((filtered_data.shape[0], original_data.shape[1]))
      # Concatenate the filtered data with the padding
      padded_data = np.concatenate((filtered_data, padding))
 
  elif diff < 0:
      padded_data = filtered_data[:,:-abs(diff)]

  elif diff == 0:
      padded_data = filtered_data

  return padded_data

def mse(original_data, filtered_data):
    filter_data = pad_data(original_data,filtered_data)
    return np.mean((original_data - filter_data) ** 2)

# Calculate MSE
mse_value_m = mse(ecg_data, ecg_medfilt)
mse_value_l = mse(ecg_data, ecg_lowpass)
mse_value_w = mse(ecg_data, ecg_wavelet)
print("MSE value of Median Filtering:", mse_value_m)
print("MSE value of Low-pass Filtering:", mse_value_l)
print("MSE value of Wavelet Filtering:", mse_value_w)


# Wavlet filtering is chosen based on visualisation and the error comparison,both. 

# ## Splitting Train & Test set

# In[22]:


# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(ecg_wavelet, labels, test_size=0.2, random_state=42)


# ## Feature Extraction

# In[23]:


# Initializing an empty list to store the features
features = []

# Extracting features for each sample
for i in range(X_train.shape[0]):
    #Finding the R-peaks
    r_peaks = scipy.signal.find_peaks(X_train[i])[0]

    #Initialize lists to hold R-peak and T-peak amplitudes
    r_amplitudes = []
    t_amplitudes = []

    # Iterate through R-peak locations to find corresponding T-peak amplitudes
    for r_peak in r_peaks:
        # Find the index of the T-peak (minimum value) in the interval from R-peak to R-peak + 200 samples
        t_peak = np.argmin(X_train[i][r_peak:r_peak+200]) + r_peak
        #Append the R-peak amplitude and T-peak amplitude to the lists
        r_amplitudes.append(X_train[i][r_peak])
        t_amplitudes.append(X_train[i][t_peak])

    # extracting singular value metrics from the r_amplitudes
    std_r_amp = np.std(r_amplitudes)
    mean_r_amp = np.mean(r_amplitudes)
    median_r_amp = np.median(r_amplitudes)
    sum_r_amp = np.sum(r_amplitudes)
    # extracting singular value metrics from the t_amplitudes
    std_t_amp = np.std(t_amplitudes)
    mean_t_amp = np.mean(t_amplitudes)
    median_t_amp = np.median(t_amplitudes)
    sum_t_amp = np.sum(t_amplitudes)

    # Find the time between consecutive R-peaks
    rr_intervals = np.diff(r_peaks)

    # Calculate the time duration of the data collection
    time_duration = (len(X_train[i]) - 1) / 1000 # assuming data is in ms

    # Calculate the sampling rate
    sampling_rate = len(X_train[i]) / time_duration

    # Calculate heart rate
    duration = len(X_train[i]) / sampling_rate
    heart_rate = (len(r_peaks) / duration) * 60

    # QRS duration
    qrs_duration = []
    for j in range(len(r_peaks)):
        qrs_duration.append(r_peaks[j]-r_peaks[j-1])
    # extracting singular value metrics from the qrs_durations
    std_qrs = np.std(qrs_duration)
    mean_qrs = np.mean(qrs_duration)
    median_qrs = np.median(qrs_duration)
    sum_qrs = np.sum(qrs_duration)

    # Extracting the singular value metrics from the RR-interval
    std_rr = np.std(rr_intervals)
    mean_rr = np.mean(rr_intervals)
    median_rr = np.median(rr_intervals)
    sum_rr = np.sum(rr_intervals)

    # Extracting the overall standard deviation 
    std = np.std(X_train[i])
    
    # Extracting the overall mean 
    mean = np.mean(X_train[i])

    # Appending the features to the list
    features.append([mean, std, std_qrs, mean_qrs,median_qrs, sum_qrs, std_r_amp, mean_r_amp, median_r_amp, sum_r_amp, std_t_amp, mean_t_amp, median_t_amp, sum_t_amp, sum_rr, std_rr, mean_rr,median_rr, heart_rate])

# Converting the list to a numpy array
features = np.array(features)


# In[13]:


pd.DataFrame(features)


# Feature Extraction of test set

# In[24]:


# Initializing an empty list to store the features
X_test_fe = []

# Extracting features for each sample
for i in range(X_test.shape[0]):
    # Finding the R-peaks
    r_peaks = scipy.signal.find_peaks(X_test[i])[0]

    # Initialize lists to hold R-peak and T-peak amplitudes
    r_amplitudes = []
    t_amplitudes = []

    # Iterate through R-peak locations to find corresponding T-peak amplitudes
    for r_peak in r_peaks:
        # Find the index of the T-peak (minimum value) in the interval from R-peak to R-peak + 200 samples
        t_peak = np.argmin(X_test[i][r_peak:r_peak+200]) + r_peak
        # Append the R-peak amplitude and T-peak amplitude to the lists
        r_amplitudes.append(X_test[i][r_peak])
        t_amplitudes.append(X_test[i][t_peak])
    #extracting singular value metrics from the r_amplitudes
    std_r_amp = np.std(r_amplitudes)
    mean_r_amp = np.mean(r_amplitudes)
    median_r_amp = np.median(r_amplitudes)
    sum_r_amp = np.sum(r_amplitudes)
    #extracting singular value metrics from the t_amplitudes
    std_t_amp = np.std(t_amplitudes)
    mean_t_amp = np.mean(t_amplitudes)
    median_t_amp = np.median(t_amplitudes)
    sum_t_amp = np.sum(t_amplitudes)

    # Find the time between consecutive R-peaks
    rr_intervals = np.diff(r_peaks)

    # Calculate the time duration of the data collection
    time_duration = (len(X_test[i]) - 1) / 1000 # assuming data is in ms

    # Calculate the sampling rate
    sampling_rate = len(X_test[i]) / time_duration

    # Calculate heart rate
    duration = len(X_test[i]) / sampling_rate
    heart_rate = (len(r_peaks) / duration) * 60

    # QRS duration
    qrs_duration = []
    for j in range(len(r_peaks)):
        qrs_duration.append(r_peaks[j]-r_peaks[j-1])
    #extracting singular value metrics from the qrs_duartions
    std_qrs = np.std(qrs_duration)
    mean_qrs = np.mean(qrs_duration)
    median_qrs = np.median(qrs_duration)
    sum_qrs = np.sum(qrs_duration)

    # Extracting the standard deviation of the RR-interval
    std_rr = np.std(rr_intervals)
    mean_rr = np.mean(rr_intervals)
    median_rr = np.median(rr_intervals)
    sum_rr = np.sum(rr_intervals)
    
      # Extracting the standard deviation of the RR-interval
    std = np.std(X_test[i])
    
    # Extracting the mean of the RR-interval
    mean = np.mean(X_test[i])

    # Appending the features to the list
    X_test_fe.append([mean, std,  std_qrs, mean_qrs,median_qrs, sum_qrs, std_r_amp, mean_r_amp, median_r_amp, sum_r_amp, std_t_amp, mean_t_amp, median_t_amp, sum_t_amp, sum_rr, std_rr, mean_rr,median_rr,heart_rate])

# Converting the list to a numpy array
X_test_fe = np.array(X_test_fe)


# In[25]:


X_test_fe.shape


# ## Model Building and Training
# 

# In[52]:


# Define the number of features in the train dataframe
num_features = features.shape[1]
# Reshape the features data to be in the right shape for LSTM input
features = np.asarray(features).astype('float32')
features = features.reshape(features.shape[0], features.shape[1], 1)

X_test_fe = np.asarray(X_test_fe).astype('float32')
X_test_fe = X_test_fe.reshape(X_test_fe.shape[0], X_test_fe.shape[1], 1)

# Define the model architecture
model = Sequential()
model.add(LSTM(64, input_shape=(features.shape[1], 1)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(features, y_train, validation_data=(X_test_fe, y_test), epochs=50, batch_size=32)

# Make predictions on the validation set
y_pred = model.predict(X_test_fe)

# Convert the predicted values to binary labels
y_pred = [1 if p>0.5 else 0 for p in y_pred]






# ## Evaluation of the Model

# In[47]:


# calculate the accuracy
acc = accuracy_score(y_test, y_pred)

#calculate the AUC score
auc = round(roc_auc_score(y_test, y_pred),2)

#classification report provides all metrics e.g. precision, recall, etc. 
all_met = classification_report(y_test, y_pred)

# Print the accuracy
print("Accuracy: ", acc*100, "%")
print(" \n")
print("AUC:", auc)
print(" \n")
print("Classification Report: \n", all_met)
print(" \n")


# Calculate the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
conf_mat_df = pd.DataFrame(conf_mat, columns=['Predicted Negative', 'Predicted Positive'], index=['Actual Negative', 'Actual Positive'])

fig = px.imshow(conf_mat_df, text_auto= True, color_continuous_scale='Blues')
fig.update_xaxes(side='top', title_text='Predicted')
fig.update_yaxes(title_text='Actual')
fig.show()



# Visualising the training error and validation error after every epoch of the model training

# In[51]:


# Plot training and validation error
fig = go.Figure()
fig.add_trace(go.Scatter( y=history.history['loss'], mode='lines', name='Training'))
fig.add_trace(go.Scatter( y=history.history['val_loss'], mode='lines', name='Validation'))
fig.update_layout(xaxis_title="Epoch", yaxis_title="Error", title= {'text': 'Model Error', 'xanchor': 'center', 'yanchor': 'top', 'x':0.5} , bargap=0)
fig.show()


model.save('ecg_lstm_model.h5')  