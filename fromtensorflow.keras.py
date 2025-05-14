from tensorflow.keras.models import load_model
model = load_model("C:\\Users\\hanar\\OneDrive\\Desktop\\CuraHealthAI\\attached_assets\\diabetes_model.keras")  # or .h5 if that's the format
model.summary()
