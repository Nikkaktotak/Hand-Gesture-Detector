import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from skimage import io, color, transform
from sklearn.preprocessing import label_binarize

# Шлях до кореневої папки бази даних
root_dir = 'D:/Delete/archive/leapGestRecog'

images = []
labels = []

# Зчитування зображень та міток
for directory in os.listdir(root_dir):
    if directory.isdigit():  # Перевірка, що це папка з ідентифікатором суб'єкта
        subject_dir = os.path.join(root_dir, directory)
        for gesture_folder in os.listdir(subject_dir):
            gesture_path = os.path.join(subject_dir, gesture_folder)
            if os.path.isdir(gesture_path):
                for img in os.listdir(gesture_path):
                    img_path = os.path.join(gesture_path, img)
                    image = io.imread(img_path, as_gray=True)  # Зчитування зображення у відтінках сірого
                    images.append(image)
                    labels.append(gesture_folder)

# Створення DataFrame
hand_gesture_data = pd.DataFrame({'Images': images, 'Labels': labels})

# Розділення на тренувальний та тестувальний набори даних
train_set, test_set = train_test_split(hand_gesture_data, test_size=0.2, random_state=42)
train_set, val_set = train_test_split(train_set, test_size=0.3, random_state=42)

# Ініціалізація ваг та зсувів
input_size = 784  # 28*28 пікселів
hidden_size = 30
output_size = 10

weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))

bias_input_to_hidden = np.zeros((hidden_size, 1))
bias_hidden_to_output = np.zeros((output_size, 1))

learning_rate = 0.01
epochs = 10

# Конвертувати текстові мітки у відповідні вектори
labels_one_hot = label_binarize(labels, classes=np.unique(labels))

# Цикл навчання
for epoch in range(epochs):
    total_error = 0
    for idx, row in train_set.iterrows():
        image = row['Images']
        label = labels_one_hot[idx]  # Використовуємо вектор міток у форматі one-hot encoding

        # Приведення зображення до розміру 28x28
        image = transform.resize(image, (28, 28))

        # Розгортання зображення у вектор
        image = np.reshape(image, (-1, 1))

        # Пряме поширення
        hidden_raw = weights_input_to_hidden @ image + bias_input_to_hidden
        hidden = 1 / (1 + np.exp(-hidden_raw))
        output_raw = weights_hidden_to_output @ hidden + bias_hidden_to_output
        output = 1 / (1 + np.exp(-output_raw))

        # Обчислення помилки
        error = label.reshape(-1, 1) - output
        total_error += np.sum(error ** 2)

        # Зворотнє поширення
        output_delta = error * output * (1 - output)
        hidden_delta = hidden * (1 - hidden) * (weights_hidden_to_output.T @ output_delta)

        # Оновлення ваг між прихованим та вихідним шаром
        weights_hidden_to_output += learning_rate * output_delta @ hidden.T
        bias_hidden_to_output += learning_rate * output_delta
        weights_input_to_hidden += learning_rate * hidden_delta @ image.T
        bias_input_to_hidden += learning_rate * hidden_delta

    print(f"Epoch {epoch+1}, Error: {total_error/len(train_set)}")

# Оцінка точності на валідаційному наборі
correct = 0
for idx, row in val_set.iterrows():
    image = row['Images']
    label = labels_one_hot[idx]  # Використовуємо вектор міток у форматі one-hot encoding

    # Приведення зображення до розміру 28x28
    image = transform.resize(image, (28, 28))

    # Розгортання зображення у вектор
    image = np.reshape(image, (-1, 1))

    # Пряме поширення
    hidden = 1 / (1 + np.exp(-(weights_input_to_hidden @ image + bias_input_to_hidden)))
    output = 1 / (1 + np.exp(-(weights_hidden_to_output @ hidden + bias_hidden_to_output)))

    # Порівняння з прогнозованою міткою
    if np.argmax(output) == np.argmax(label):
        correct += 1

accuracy = correct / len(val_set)
print(f"Validation Accuracy: {accuracy}")