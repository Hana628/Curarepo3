import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = "attached_assets/ecg_lstm_model.h5"
SEQUENCE_LENGTH = 7  # Original sequence length for sliding window
MODEL_SEQUENCE_LENGTH = 19  # What the model expects
anomaly_model = None
skip_tf = False

try:
    import tensorflow as tf
    from models.custom_tf_loader import load_tf_model_safely
    logger.info("TensorFlow imported successfully for anomaly detection")

    try:
        anomaly_model = load_tf_model_safely(MODEL_PATH)


        if anomaly_model is not None:
            anomaly_model.compile(optimizer='adam', loss='mse')
            logger.info("Successfully loaded and compiled the anomaly model")
        else:
            logger.warning("No model found. Creating default architecture.")
            anomaly_model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(MODEL_SEQUENCE_LENGTH, 1)),
                tf.keras.layers.LSTM(16, activation='relu', return_sequences=True),
                tf.keras.layers.LSTM(8, activation='relu', return_sequences=False),
                tf.keras.layers.RepeatVector(MODEL_SEQUENCE_LENGTH),
                tf.keras.layers.LSTM(8, activation='relu', return_sequences=True),
                tf.keras.layers.LSTM(16, activation='relu', return_sequences=True),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
            ])
            anomaly_model.compile(optimizer='adam', loss='mse')
            logger.info("Created and initialized default anomaly detection model")
    except Exception as e:
        logger.error(f"Error loading or initializing anomaly model: {str(e)}")
        anomaly_model = None

except ImportError as e:
    logger.error(f"TensorFlow import failed: {str(e)}")
    skip_tf = True
    anomaly_model = None


def detect_anomalies(data):
    try:
        required_features = [
            'r_peaks', 't_peaks', 'heart_rate', 'qrs_duration',
            'std_r_amp', 'mean_r_amp', 'median_r_amp', 'sum_r_amp',
            'std_t_amp', 'mean_t_amp', 'median_t_amp', 'sum_t_amp',
            'std_qrs', 'mean_qrs', 'median_qrs',
            'raw_ecg_data'
        ]

        if 'raw_ecg_data' in data:
            raw_ecg = np.array(data['raw_ecg_data'], dtype=np.float32)
            features = extract_ecg_features(raw_ecg)
            metric_name = 'ecg_signal'
            series = features.flatten()
        elif all(feature in data for feature in required_features[:5]):
            features = [data.get(name, 0.0) for name in required_features[:-1]]
            series = np.array(features, dtype=np.float32)
            metric_name = 'ecg_features'
        elif 'heart_rate' in data:
            series = np.array(data['heart_rate'], dtype=np.float32)
            metric_name = 'heart_rate'
        elif 'blood_pressure_systolic' in data:
            series = np.array(data['blood_pressure_systolic'], dtype=np.float32)
            metric_name = 'blood_pressure_systolic'
        else:
            raise ValueError("No valid ECG data or features provided for anomaly detection")

        if len(series) < SEQUENCE_LENGTH:
            return {
                "warning": f"Insufficient data points. Need at least {SEQUENCE_LENGTH}, but got {len(series)}.",
                "anomalies_detected": False,
                "anomaly_indices": [],
                "recommendation": "Please provide more data points for reliable anomaly detection."
            }

        if anomaly_model is not None:
            try:
                sequences = create_sequences(series, SEQUENCE_LENGTH)
                normalized_sequences, _, _ = normalize_sequences(sequences)
                reshaped_sequences = normalized_sequences.reshape((-1, MODEL_SEQUENCE_LENGTH, 1))

                predicted_sequences = anomaly_model.predict(reshaped_sequences)
                mse = np.mean(np.square(reshaped_sequences - predicted_sequences), axis=(1, 2))
                threshold = np.mean(mse) + 2 * np.std(mse)
                anomaly_indices = np.where(mse > threshold)[0]
                original_anomaly_indices = [int(i) for i in anomaly_indices]
                anomalies_detected = len(original_anomaly_indices) > 0

                if anomalies_detected:
                    if metric_name == 'heart_rate':
                        recommendation = "Unusual heart rate patterns detected. Monitor and consult a healthcare professional."
                    elif metric_name == 'blood_pressure_systolic':
                        recommendation = "Unusual blood pressure patterns detected. Monitor and consider medical advice."
                    else:
                        recommendation = "Anomalies detected in your health data. Consider consulting with a healthcare professional."
                else:
                    recommendation = "No unusual patterns detected. Continue monitoring regularly."

                return {
                    "anomalies_detected": anomalies_detected,
                    "anomaly_indices": original_anomaly_indices,
                    "threshold": float(threshold),
                    "metric": metric_name,
                    "recommendation": recommendation
                }
            except Exception as model_error:
                logger.error(f"Model prediction failed: {str(model_error)}")
                return fallback_anomaly_detection(data)
        else:
            return fallback_anomaly_detection(data)

    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        return {
            "error": f"An error occurred: {str(e)}",
            "anomalies_detected": False,
            "recommendation": "Unable to process your data. Ensure valid input is provided."
        }


def create_sequences(data, seq_length):
    effective_length = MODEL_SEQUENCE_LENGTH
    sequences = []
    for i in range(len(data) - seq_length + 1):
        seq = data[i:i + seq_length]
        padded_seq = np.zeros((effective_length, 1))
        padded_seq[:len(seq), 0] = seq
        sequences.append(padded_seq)
    return np.array(sequences)


def normalize_sequences(sequences):
    seq_min = np.min(sequences)
    seq_max = np.max(sequences)
    if seq_max == seq_min:
        return sequences, seq_min, seq_max
    normalized = (sequences - seq_min) / (seq_max - seq_min)
    return normalized, seq_min, seq_max


def extract_ecg_features(raw_ecg):
    try:
        import scipy.signal
        r_peaks = scipy.signal.find_peaks(raw_ecg)[0]
        r_amplitudes, t_amplitudes = [], []

        for r_peak in r_peaks:
            end_idx = min(r_peak + 200, len(raw_ecg) - 1)
            t_peak = np.argmin(raw_ecg[r_peak:end_idx]) + r_peak
            r_amplitudes.append(raw_ecg[r_peak])
            t_amplitudes.append(raw_ecg[t_peak])

        sampling_rate = 250
        duration = len(raw_ecg) / sampling_rate
        heart_rate = (len(r_peaks) / duration) * 60 if duration > 0 else 70
        rr_intervals = np.diff(r_peaks) if len(r_peaks) > 1 else np.array([0])

        features = [
            float(r_peaks[0]) if r_peaks.size else 0,
            float(t_amplitudes[0]) if t_amplitudes else 0,
            float(heart_rate),
            float(rr_intervals[0]) if rr_intervals.size else 0,
            float(np.std(r_amplitudes)) if r_amplitudes else 0,
            float(np.mean(r_amplitudes)) if r_amplitudes else 0,
            float(np.median(r_amplitudes)) if r_amplitudes else 0,
            float(np.sum(r_amplitudes)) if r_amplitudes else 0,
            float(np.std(t_amplitudes)) if t_amplitudes else 0,
            float(np.mean(t_amplitudes)) if t_amplitudes else 0,
            float(np.median(t_amplitudes)) if t_amplitudes else 0,
            float(np.sum(t_amplitudes)) if t_amplitudes else 0,
            float(np.std(rr_intervals)) if rr_intervals.size else 0,
            float(np.mean(rr_intervals)) if rr_intervals.size else 0,
            float(np.median(rr_intervals)) if rr_intervals.size else 0,
            float(np.min(raw_ecg)),
            float(np.max(raw_ecg)),
            float(np.mean(raw_ecg)),
            float(np.std(raw_ecg))
        ]
        return np.array(features).reshape((19, 1))
    except Exception as e:
        logger.error(f"Error extracting ECG features: {str(e)}")
        return np.zeros((19, 1))


def fallback_anomaly_detection(data):
    try:
        if 'heart_rate' in data:
            series = np.array(data['heart_rate'], dtype=np.float32)
            metric_name = 'heart_rate'
        elif 'blood_pressure_systolic' in data:
            series = np.array(data['blood_pressure_systolic'], dtype=np.float32)
            metric_name = 'blood_pressure_systolic'
        elif 'blood_glucose' in data:
            series = np.array(data['blood_glucose'], dtype=np.float32)
            metric_name = 'blood_glucose'
        else:
            raise ValueError("No valid time series data provided")

        mean = np.mean(series)
        std = np.std(series)
        z_scores = np.abs((series - mean) / (std if std > 0 else 1))
        threshold = 3
        anomaly_indices = np.where(z_scores > threshold)[0]
        anomalies_detected = len(anomaly_indices) > 0

        if anomalies_detected:
            recommendation = "Unusual values detected. Consider monitoring and discussing with your healthcare provider."
        else:
            recommendation = "No anomalies detected. Continue monitoring regularly."

        return {
            "anomalies_detected": anomalies_detected,
            "anomaly_indices": [int(i) for i in anomaly_indices],
            "threshold": float(threshold),
            "metric": metric_name,
            "recommendation": recommendation,
            "note": "Using statistical Z-score method"
        }

    except Exception as e:
        logger.error(f"Error in fallback anomaly detection: {str(e)}")
        return {
            "error": f"An error occurred: {str(e)}",
            "anomalies_detected": False,
            "recommendation": "Unable to process data. Check input format."
        }
