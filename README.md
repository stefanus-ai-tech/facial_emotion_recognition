# facial_emotion_recognition
the dataset is from https://www.kaggle.com/datasets/msambare/fer2013?resource=download

# Facial Expression Recognition with TensorFlow and Keras

This tutorial will guide you through creating a facial expression recognition system using TensorFlow and Keras. The system will be trained on a dataset of facial images, and it will classify the expressions into seven categories: angry, disgust, fear, happy, neutral, sad, and surprise. Additionally, we will use K-Fold Cross-Validation for training and OpenCV to capture live webcam input for real-time facial expression recognition.

## Prerequisites

Before you start, make sure you have the following installed:

- Python 3.x
- TensorFlow
- scikit-learn
- OpenCV
- Matplotlib

You can install the necessary packages using pip:

```bash
pip install tensorflow scikit-learn opencv-python matplotlib
```

## Code Explanation

### Importing Libraries

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.model_selection import KFold
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
```

### Loading the Dataset

We will load the training and testing datasets from the specified directory. Ensure your dataset is organized into subdirectories for each class (e.g., `train/angry`, `train/happy`, etc.).

```python
data_dir = '/path/to/your/dataset'
batch_size = 64
img_height = 48
img_width = 48

# Load training datasets
train_dir = os.path.join(data_dir, 'train')
train_dataset = image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True
)

# Load test datasets
test_dir = os.path.join(data_dir, 'test')
test_dataset = image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height, img_width),
    shuffle=True
)
```

### Normalizing the Data

Normalize the pixel values to the range [0, 1].

```python
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))
```

### Combining and Converting Datasets to Numpy Arrays

Combine the training and test datasets, and convert them to numpy arrays for use with K-Fold Cross-Validation.

```python
combined_dataset = train_dataset.concatenate(test_dataset)

def dataset_to_numpy(dataset):
    images, labels = [], []
    for image_batch, label_batch in dataset:
        images.append(image_batch.numpy())
        labels.append(label_batch.numpy())
    return np.concatenate(images), np.concatenate(labels)

images, labels = dataset_to_numpy(combined_dataset)
```

### Displaying a Batch of Images

Display a batch of images with their labels to ensure the dataset is loaded correctly.

```python
def display_batch(images, labels, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(class_names[labels[i]])
        plt.axis("off")

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

for image_batch, label_batch in train_dataset.take(1):
    display_batch(image_batch.numpy(), label_batch.numpy(), class_names)
```

### Defining the Model

Create a convolutional neural network (CNN) model.

```python
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(7, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model
```

### K-Fold Cross-Validation

Use K-Fold Cross-Validation to train the model.

```python
k = 5
kf = KFold(n_splits=k, shuffle=True)

fold_no = 1
for train_index, val_index in kf.split(images):
    print(f'Training fold {fold_no}...')
    
    x_train, x_val = images[train_index], images[val_index]
    y_train, y_val = labels[train_index], labels[val_index]
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    
    model = create_model()
    
    model.fit(train_dataset, epochs=10, validation_data=val_dataset)
    
    fold_no += 1
```

### Saving the Model

Save the trained model to disk.

```python
model.save('facial_expression_model.keras')
```

### Loading the Model

Load the trained model for inference.

```python
model = load_model('facial_expression_model.keras')
```

### Real-Time Facial Expression Recognition

Use OpenCV to capture video from the webcam and perform real-time facial expression recognition.

```python
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

try:
    cv2.namedWindow("Facial Expression Detection")
    headless = False
except cv2.error as e:
    print("OpenCV GUI support not available, using headless mode.")
    headless = True

cap = cv2.VideoCapture(0)

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_gray = cv2.resize(gray, (48, 48))
    normalized_gray = resized_gray / 255.0
    input_image = np.expand_dims(normalized_gray, axis=0)
    input_image = np.expand_dims(input_image, axis=-1)
    
    predictions = model.predict(input_image)
    predicted_label = emotion_labels[np.argmax(predictions)]
    
    cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    if headless:
        cv2.imwrite('output.jpg', frame)
    else:
        cv2.imshow('Facial Expression Detection', frame)

try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        process_frame(frame)
        
        if not headless:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            for _ in range(10):
                ret, frame = cap.read()
                if not ret:
                    break
                process_frame(frame)

finally:
    cap.release()
    if not headless:
        cv2.destroyAllWindows()
```

### Demonstration video

[![Facial Expression Recognition Demonstration](https://img.youtube.com/vi/28W4qCLxPDw/maxresdefault.jpg)](https://www.youtube.com/watch?v=28W4qCLxPDw)


---
