import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



data = pd.read_csv('Churn_Modelling.csv')

data.head()

data.shape

data.drop(columns = ['RowNumber','CustomerId','Surname'],inplace=True)
print(data)


data['Gender'] = data['Gender'].replace({'Male': 0, 'Female': 1})

# Replace values in the "Geography" column
data['Geography'] = data['Geography'].replace({'France': 0, 'Germany': 1, 'Spain': 2})


# Split the data into training and testing sets
X = data.drop('Exited', axis=1)
y = data['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Build the ANN model
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))



# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Train the model
traing_model = model.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.2)



# Make predictions on the test set
predictions = model.predict(X_test)
# Convert probabilities to binary predictions
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)



import seaborn as sns
import matplotlib.pyplot as plt
# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, binary_predictions)
# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()




# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')




# Extract loss and accuracy from the history object
train_loss = traing_model.history['loss']
val_loss = traing_model.history['val_loss']
train_accuracy = traing_model.history['accuracy']
val_accuracy = traing_model.history['val_accuracy']
epochs = range(1, len(train_loss) + 1)



# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()




# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(train_accuracy, label='Training Accuracy', color='blue')
plt.plot(val_accuracy, label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

