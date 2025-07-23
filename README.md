# STM32H7VBT6_Neron 波形识别项目

![STM32H7 AI 波形识别](https://img.shields.io/badge/STM32H7-AI%20Waveform%20Recognition-blue)
![Python](https://img.shields.io/badge/Python-3.9-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

基于STM32H7的嵌入式AI波形识别系统，实现实时信号分类与边缘智能分析。

## 目录
- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [开发指南](#开发指南)
- [常见问题](#常见问题)

## 功能特性

### 核心能力
- 🎛️ 分类自主可调
- ⏱️ 实时推理速度 <5ms @480MHz
- 📡 第一条骗你的,要自己改网络结构和训练模型

### 技术栈
| 模块        | 技术方案                 |
|-------------|--------------------------|
| 神经网络    | 1D CNN (`Keras`/`TensorFlow`)|
| 模型部署    | STM32 `X-CUBE-AI`          |
| 信号采集    | 12-bit ADC       |
| 开发环境    | `Keil` MDK + `STM32CubeMX`   |

## 快速开始

### 1. Python环境配置
```bash
# 创建虚拟环境（推荐Python 3.9）
conda create -n stm32ai python=3.9
conda activate stm32ai

# 安装依赖
pip install matplotlib tensorflow

```



### 2. 训练模型

```bash
# 生成训练数据（默认生成4类波形各1000样本）
python Scripts/dataset_generator.py

# 训练CNN模型
python Scripts/train_model.py

```


模型输出路径：Scripts/Model/waveform_cnn_256.h5

### 3. 嵌入式部署
1. 使用STM32CubeMX打开工程文件
2. 通过X-CUBE-AI导入.h5模型
3. 生成代码并打开Keil工程
4. 编译烧录到STM32H7开发板

## 项目结构
```
├── Scripts/ # Python训练脚本
│ ├── dataset_generator.py # 波形数据生成器
│ ├── train_model.py # CNN训练脚本
│ └── Model/ # 预训练模型
├── STM32H7VBT6_Neron/ # 嵌入式工程
│ ├── Core/ # 用户代码
│ │ ├── Src/ # 数据采集/推理逻辑
│ │ └── Inc/ # AI接口定义
│ ├── MDK-ARM/ # Keil项目文件
│ └── .ai/ # X-CUBE-AI配置
└── docs/ # 开发文档
```

## 开发指南

无

## 常见问题

Q: 模型在PC上尚可,部署后效果不好 \
A: 
1. 检查模型量化是否正确
2. 检查内存是否对齐

Q: 识别准确率低怎么办？ \
A:
1. 增加训练数据量
2. 调整CNN网络结构
3. 没救了

---