from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization

# Load the dataset (replace with the appropriate Keras dataset)
(x_train, y_train), (x_test, y_test) = keras.datasets.california_housing.load_data(
    version="large", path="california_housing.npz", test_split=0.2, seed=113
)

# Perform data preprocessing with StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Create the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    BatchNormalization(),  # Batch normalization added
    keras.layers.Dense(64, activation='relu'),
    BatchNormalization(),  # Batch normalization added
    keras.layers.Dense(64, activation='relu'),
    BatchNormalization(),  # Batch normalization added
    keras.layers.Dense(64, activation='relu'),
    BatchNormalization(),  # Batch normalization added
    keras.layers.Dense(64, activation='relu'),
    BatchNormalization(),  # Batch normalization added
    keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Evaluate the model on the test set
loss = model.evaluate(x_test, y_test)
print("Mean Squared Error:", loss)