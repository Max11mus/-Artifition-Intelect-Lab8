# Імпортуємо необхідні бібліотеки
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

# Вхідні дані A,B
a_b_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
# Вихідні дані NOR, NAND
nor_nand_train = [[1, 1], [0, 1], [0, 1], [0, 0]]

# Створюємо модель нейромережі
# Вхідний шар з двома нейронами
input_layer = Input(shape=(2,))
# Прихований шар з 2 нейронами
hidden_layer = Dense(2)(input_layer)
# Вихідний шар з двома нейронами і функцією активації hard_sigmoid
output_layer = Dense(2, activation='hard_sigmoid')(hidden_layer)
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Виводимо опис моделі
model.summary()

# Компіляція моделі з використанням бінарної кросс-ентропії як функції втрати та алгоритму оптимізації Adam
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Навчання мережі
history = model.fit(x = a_b_train, y = nor_nand_train, epochs=3000, batch_size=4)

# Використання навченої моделі для класифікації
predictions = model.predict(a_b_train)

# Виведення результатів класифікації
for i in range(len(a_b_train)):
    print("Вхідні дані: {}, Вихідні дані: {}".format(a_b_train[i], predictions[i]))

# Отримуємо значення функції втрат на кожній епосі тренування
loss = history.history['loss']
accuracy = history.history['accuracy']
plt.plot(loss)
plt.xlabel('Епоха')
plt.ylabel('Значення функції втрат')
plt.grid(True)
plt.show()

# Виводимо значення точності
plt.plot(accuracy)
plt.xlabel('Епоха')
plt.ylabel('Точність')
plt.grid(True)
plt.show()

