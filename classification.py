import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import regularizers


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

print("[+] SCRIPT RUNNING")

train_dir = "dataset/train"
test_dir = "dataset/test"

CATEGORIES = ["benign", "malignant"]
IMG_SIZE = 224

training_data = []
testing_data = []

# Configure TensorFlow memory growth
# Configure TensorFlow memory growth (modern approach)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[+] GPU memory growth enabled")
        
        # Limit GPU memory usage (optional)
        # Adjust the 0.8 value based on your system's memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024*8)]  # 8GB limit
        )
    except RuntimeError as e:
        print(f"[-] GPU configuration error: {e}")
else:
    print("[-] No GPU found, using CPU")

def create_data(dir, dataset):
    print("[~] FILTERING IMAGES...")
    for category in CATEGORIES:
        path = os.path.join(dir, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            tmp_path = os.path.join(path, img)
            src = cv2.imread(tmp_path)
            
            if src is None:
                print(f"[-] Error loading image: {tmp_path}")
                continue  # Skip corrupt or unreadable files
            
            # Convert original image to grayscale
            grayScale = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
            
            # Kernel for morphological filtering
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17,17))
        
            # Perform blackHat filtering to find hair contours
            blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
        
            # Thresholding to intensify hair contours
            _, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
            
            # Inpaint the original image based on the mask
            dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)
            new_array = cv2.resize(dst, (IMG_SIZE, IMG_SIZE))
            dataset.append([new_array, class_num])

        print(f'    [+] {category} IMAGES PREPARED FOR ANALYSIS')

create_data(train_dir, training_data)
create_data(test_dir, testing_data)
            
random.shuffle(training_data)
random.shuffle(testing_data)

X_train = []
X_test = []
y_train = []
y_test = []

for features, label in training_data:
    X_train.append(features)
    y_train.append(label)
    
for features, label in testing_data:
    X_test.append(features)
    y_test.append(label)

# Free up memory by deleting original data lists
del training_data
del testing_data

print("[+] LABELLED & FEATURED")
    
X_train = np.array(X_train)
X_test = np.array(X_test)

print(f"[+] DATA SPLIT INTO NP ARRAYS - X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

print("[+] CATEGORISED DATA")

# Normalize and convert to float32 to save memory
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print("[+] NORMALIZED PICTURE RGB VALUES")

# Check shapes before splitting
print(f"Shapes before split - X_train: {X_train.shape}, y_train: {y_train.shape}")

# Splitting data into validation and training set
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train, y_train, test_size=0.2, random_state=2
)

print("[+] DATA SPLIT INTO VALIDATION & TRAINING SET")

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=30, 
    width_shift_range=0.2, 
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    horizontal_flip=True
)

# Remove datagen.fit(X_train) as it's unnecessary without feature-wise normalization

print("[+] TRAINING DATA AUGMENTATION CONFIGURED")

# Model building remains the same
def build(input_shape=(224,224,3), lr=1e-3, num_classes=2, activ='relu', optim='adam'):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

# Rest of the code remains unchanged...
input_shape = (224,224,3)
lr = 1e-5
activ = 'relu'
optim = 'adam'
epochs = 25
batch_size = 64

print("[-] BUILDING MODEL...")

model = build(lr=lr, activ= activ, optim=optim, input_shape= input_shape)

print("[+] MODEL BUILT")

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


#training the model with validation data
history = model.fit(datagen.flow(X_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_validation,y_validation), callbacks=[learning_rate_reduction])


# list all data in history
print("[-] PRINTING HISTORY KEYS")
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Saving the model 
model.save('skin_cancer_classification.h5')

print("[+] MODEL SAVED AS 'skin_cancer_classification.h5'")

# Evaluate with smaller batches (32 instead of default 32)
loss_v, accuracy_v = model.evaluate(X_validation, y_validation, batch_size=32, verbose=1)
loss, accuracy = model.evaluate(X_test, y_test, batch_size=32, verbose=1)
print(f"Validation: accuracy = {accuracy_v:.6f}  ;  loss_v = {loss_v:.6f}")
print(f"Test: accuracy = {accuracy:.6f}  ;  loss = {loss:.6f}")

# Confusion matrix with memory optimizations
def safe_confusion_matrix(model, X_val, y_val, batch_size=32):
    """Calculate confusion matrix in memory-efficient batches"""
    y_true = np.argmax(y_val, axis=1)
    y_pred = []
    
    # Predict in batches
    for i in range(0, len(X_val), batch_size):
        # Create batch and predict
        batch = X_val[i:i+batch_size]
        pred = model.predict(batch, verbose=0)
        y_pred.extend(np.argmax(pred, axis=1))
        
        # Clean up batch-specific variables
        del pred
        if i + batch_size < len(X_val):  # Only delete if not last batch
            del batch
    
    return confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Generate and plot confusion matrix
confusion_mtx = safe_confusion_matrix(model, X_validation, y_validation)
plot_confusion_matrix(confusion_mtx, classes=range(2))

# Explicit cleanup AFTER last use
del X_validation, y_validation
