import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# Function to resize images
def resize_images(images, size):
    resized_images = tf.image.resize(images, size)
    return resized_images

# Step 2: Load and preprocess the CIFAR-10 dataset with resizing
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Change the data format from (height, width, channels) to (batch, height, width, channels)
train_images = train_images.reshape(-1, 32, 32, 3)
test_images = test_images.reshape(-1, 32, 32, 3)

train_images = resize_images(train_images, (28, 28))
test_images = resize_images(test_images, (28, 28))

train_images, test_images = train_images / 255.0, test_images / 255.0

# Step 3: Define the CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Step 4: Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Step 5: Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Step 6: Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)