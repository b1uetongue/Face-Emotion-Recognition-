import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import to_categorical

def load_and_preprocess_data(file_path):
    Catch_bat = pd.read_csv(file_path)
    
    X_train, Y_train, X_test, Y_test = [], [], [], []
    
    for index, row in Catch_bat.iterrows():
        val = row['pixels'].split(' ')
        try:
            if 'Training' in row['Usage']:
                X_train.append(np.array(val, 'float32'))
                Y_train.append(row['emotion'])
            elif 'PublicTest' in row['Usage']:
                X_test.append(np.array(val, 'float32'))
                Y_test.append(row['emotion'])
        except:
            pass
    
    X_train = np.array(X_train, 'float32')
    Y_train = np.array(Y_train, 'float32')
    X_test = np.array(X_test, 'float32')
    Y_test = np.array(Y_test, 'float32')
    
    # 数据标准化
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
    
    # one-hot 编码
    labels = 7
    Y_train = to_categorical(Y_train, num_classes=labels)
    Y_test = to_categorical(Y_test, num_classes=labels)
    
    # 数据reshape
    width, height = 48, 48
    X_train = X_train.reshape(X_train.shape[0], width, height, 1)
    X_test = X_test.reshape(X_test.shape[0], width, height, 1)
    
    return X_train, Y_train, X_test, Y_test

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def train_model(model, X_train, Y_train, X_test, Y_test, batch_size=64, epochs=1):
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, Y_test), shuffle=True)

def save_model(model, model_file, weights_file):
    model_json = model.to_json()
    with open(model_file, 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(weights_file)

# 加载和预处理数据
X_train, Y_train, X_test, Y_test = load_and_preprocess_data('fer2013.csv')

# 构建模型
input_shape = X_train.shape[1:]
num_classes = 7
model = build_model(input_shape, num_classes)

# 训练模型
train_model(model, X_train, Y_train, X_test, Y_test)

# 保存模型
save_model(model, 'fer-1.json', 'fer-1.h5')
