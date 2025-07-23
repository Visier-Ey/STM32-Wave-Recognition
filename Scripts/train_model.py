# train_model.py
import numpy as np
import h5py
import os
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, InputLayer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 可配置参数
class Config:
    dataset_path = "waveform_dataset.h5"
    model_path = "waveform_cnn_256.h5"
    epochs = 20
    batch_size = 32
    class_names = ["SINE", "TRIANGLE", "FSK", "BPSK"]

def load_data():
    if not os.path.exists(Config.dataset_path):
        raise FileNotFoundError(f"❌ 数据集文件不存在: {Config.dataset_path}")

    with h5py.File(Config.dataset_path, 'r') as f:
        X = f['X'][:]
        y = f['y'][:]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(signal_length, num_classes):
    return Sequential([
        InputLayer(input_shape=(signal_length, 1)),
        Conv1D(8, 5, activation='relu'),  # 减少滤波器数量和核大小
        MaxPooling1D(2),
        Conv1D(16, 5, activation='relu'),  # 减少滤波器数量和核大小
        MaxPooling1D(2),
        Flatten(),
        Dense(16, activation='relu'),      # 减少全连接层单元数
        Dense(num_classes, activation='softmax')
    ])

def train_model():
    X_train, X_test, y_train, y_test = load_data()
    signal_length = X_train.shape[1]
    num_classes = y_train.shape[1]

    model = build_model(signal_length, num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("🚀 开始训练模型...")
    history = model.fit(X_train, y_train,
                        epochs=Config.epochs,
                        batch_size=Config.batch_size,
                        validation_data=(X_test, y_test))

    loss, acc = model.evaluate(X_test, y_test)
    print(f"✅ 测试准确率: {acc:.4f}")

    model.save(Config.model_path)
    print(f"✅ 模型已保存到 {Config.model_path}")
    return history

def evaluate_model():
    """验证已训练模型的准确率和混淆矩阵"""
    if not os.path.exists(Config.model_path):
        raise FileNotFoundError(f"❌ 模型文件不存在: {Config.model_path}")
    
    model = load_model(Config.model_path)
    _, X_test, _, y_test = load_data()

    y_pred = model.predict(X_test)
    y_true_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # 打印分类报告
    print("\n📊 分类报告:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=Config.class_names))

    # 混淆矩阵可视化
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.class_names,
                yticklabels=Config.class_names)
    plt.xlabel("Prediction category")
    plt.ylabel("Real category")
    plt.title("confusion matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 只解开你要运行的部分
    train_model()
    # evaluate_model()  # 如只验证模型，请注释上行并取消此行注释
