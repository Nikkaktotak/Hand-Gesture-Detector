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
hidden_size1 = 50  # Розмір першого прихованого шару
hidden_size2 = 30  # Розмір другого прихованого шару
output_size = 10

weights_input_to_hidden1 = np.random.uniform(-0.5, 0.5, (hidden_size1, input_size))
weights_hidden1_to_hidden2 = np.random.uniform(-0.5, 0.5, (hidden_size2, hidden_size1))
weights_hidden2_to_output = np.random.uniform(-0.5, 0.5, (output_size, hidden_size2))

bias_input_to_hidden1 = np.zeros((hidden_size1, 1))
bias_hidden1_to_hidden2 = np.zeros((hidden_size2, 1))
bias_hidden2_to_output = np.zeros((output_size, 1))

learning_rate = 0.01
epochs = 25

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
        hidden_raw1 = weights_input_to_hidden1 @ image + bias_input_to_hidden1
        hidden1 = 1 / (1 + np.exp(-hidden_raw1))
        hidden_raw2 = weights_hidden1_to_hidden2 @ hidden1 + bias_hidden1_to_hidden2
        hidden2 = 1 / (1 + np.exp(-hidden_raw2))
        output_raw = weights_hidden2_to_output @ hidden2 + bias_hidden2_to_output
        output = 1 / (1 + np.exp(-output_raw))

        # Обчислення помилки
        error = label.reshape(-1, 1) - output
        total_error += np.sum(error ** 2)

        # Зворотнє поширення
        output_delta = error * output * (1 - output)
        hidden2_delta = hidden2 * (1 - hidden2) * (weights_hidden2_to_output.T @ output_delta)
        hidden1_delta = hidden1 * (1 - hidden1) * (weights_hidden1_to_hidden2.T @ hidden2_delta)

        # Оновлення ваг
        weights_hidden2_to_output += learning_rate * output_delta @ hidden2.T
        bias_hidden2_to_output += learning_rate * output_delta
        weights_hidden1_to_hidden2 += learning_rate * hidden2_delta @ hidden1.T
        bias_hidden1_to_hidden2 += learning_rate * hidden2_delta
        weights_input_to_hidden1 += learning_rate * hidden1_delta @ image.T
        bias_input_to_hidden1 += learning_rate * hidden1_delta

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
    hidden1 = 1 / (1 + np.exp(-(weights_input_to_hidden1 @ image + bias_input_to_hidden1)))
    hidden2 = 1 / (1 + np.exp(-(weights_hidden1_to_hidden2 @ hidden1 + bias_hidden1_to_hidden2)))
    output = 1 / (1 + np.exp(-(weights_hidden2_to_output @ hidden2 + bias_hidden2_to_output)))

    # Порівняння з прогнозованою міткою
    if np.argmax(output) == np.argmax(label):
        correct += 1

accuracy = correct / len(val_set)
print(f"Validation Accuracy: {accuracy}")
