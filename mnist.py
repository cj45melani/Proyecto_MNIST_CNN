import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train[:80%]', 'test[80%:]'],
    as_supervised=True,
    with_info=True
)

def preprocessing(img, label):
    img = tf.image.resize(img, (28, 28))
    img = tf.cast(img, tf.float32) / 255.0  
    return img, label

train_ds = ds_train.map(preprocessing).shuffle(10000).batch(32).prefetch(1)
test_ds = ds_test.map(preprocessing).batch(32).prefetch(1)

for imgs, labels in train_ds.take(1):
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(imgs[i].numpy().reshape(28, 28), cmap='gray')
        plt.title(f"Número: {labels[i].numpy()}")
        plt.axis('off')
plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=10, validation_data=test_ds)

plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del modelo')
plt.grid()
plt.legend()
plt.show()

for img, label in test_ds.take(1):
    pred = model.predict(img)
    plt.imshow(img[13], cmap='gray')
    plt.title(tf.argmax(pred[13]))  
    plt.axis('off')
    plt.show()
