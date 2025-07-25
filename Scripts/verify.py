import numpy as np
import os
import h5py
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def generate_waveform(cls, length, fs, freq, amp=1.0, phase_shift=0):
    """生成指定类型的波形"""
    duration = length / fs
    t = np.linspace(0, duration, length, endpoint=False)

    if cls == 0:  # SINE
        return amp * np.sin(2 * np.pi * freq * t + phase_shift)
    elif cls == 1:  # TRIANGLE
        return amp * (2 * np.abs(2 * ((freq * t + phase_shift / (2 * np.pi)) % 1) - 1) - 1)
    elif cls == 2:  # FSK
        half = length // 2
        return amp * np.concatenate([
            np.sin(2 * np.pi * freq * t[:half] + phase_shift),
            np.sin(2 * np.pi * (2 * freq) * t[half:] + phase_shift)
        ])
    elif cls == 3:  # BPSK
        return amp * np.sign(np.sin(2 * np.pi * freq * t + phase_shift)) * np.random.choice([-1, 1])
    return np.zeros(length)

def export_to_c(array, name, header_path, source_path):
    """将波形数组导出为C语言头文件和源文件"""
    array_str = ",    ".join(f"{x:.6f}f" for x in array)

    with open(source_path, 'w') as cfile:
        cfile.write(f'#include "{os.path.basename(header_path)}"\n\n')
        cfile.write(f'float {name}[{len(array)}] = {{\n    {array_str}\n}};\n')

    with open(header_path, 'w') as hfile:
        hfile.write(f'#ifndef WAVEFORM_DATA_H\n#define WAVEFORM_DATA_H\n\n')
        hfile.write(f'#define WAVEFORM_LEN {len(array)}\n')
        hfile.write(f'extern float {name}[{len(array)}];\n\n')
        hfile.write(f'#endif // WAVEFORM_DATA_H\n')

    print(f"✅ 已生成C文件：\n  - {header_path}\n  - {source_path}")

def plot_waveform(waveform, waveform_type, fs, freq):
    """绘制波形图"""
    plt.figure(figsize=(12, 4))
    plt.plot(waveform, label='Waveform')
    plt.title(f"{waveform_type.upper()} Waveform\nFrequency: {freq/1000:.1f}kHz, Sampling Rate: {fs/1000:.1f}kHz")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def validate_with_model(model, waveform, waveform_type, noise_level=0.05, plot=True):
    """用训练好的模型验证波形"""
    # 添加噪声
    noisy_wave = waveform + np.random.normal(0, noise_level, len(waveform))
    
    # 准备模型输入
    input_data = noisy_wave.astype('float32').reshape(1, len(waveform), 1)
    
    # 模型预测
    predictions = model.predict(input_data, verbose=0)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]
    
    # 类别名称
    class_names = ["SINE", "TRIANGLE", "FSK", "BPSK"]
    
    # 打印结果
    print("\n" + "="*50)
    print(f"🔍 模型验证结果 - {waveform_type.upper()}波形")
    print(f"🎯 预测: {class_names[predicted_class]} (置信度: {confidence*100:.1f}%)")
    print("="*50 + "\n")
    
    # 打印所有类别概率
    for i, prob in enumerate(predictions[0]):
        print(f"{class_names[i]}: {prob*100:.2f}%")
    
    # 绘制波形
    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(waveform, label='Clean Signal', alpha=0.7)
        plt.plot(noisy_wave, label=f'Noisy Signal (σ={noise_level})', alpha=0.5)
        plt.title(f"{waveform_type.upper()} Waveform\nPredicted: {class_names[predicted_class]} ({confidence*100:.1f}%)")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return predicted_class, confidence

def main():
    parser = argparse.ArgumentParser(description="波形生成器 (带模型验证)")
    parser.add_argument('--type', type=str, default='sine', 
                       choices=['sine', 'triangle', 'fsk', 'bpsk'], 
                       help='波形类型')
    parser.add_argument('--length', type=int, default=256, 
                       help='采样点数 (默认:256)')
    parser.add_argument('--fs', type=float,   default=1000000, 
                       help='采样率Hz (默认:1MHz)')
    parser.add_argument('--freq', type=float, default=100000.0, 
                       help='基础频率Hz (默认:10kHz)')
    parser.add_argument('--amp', type=float, default=1.0, 
                       help='幅值 (默认:1.0)')
    parser.add_argument('--phase', type=float, default=0.0, 
                       help='相位(弧度) (默认:0.0)')
    parser.add_argument('--noise', type=float, default=0.005, 
                       help='测试噪声水平 (默认:0.05)')
    parser.add_argument('--output', type=str, default='Verify/waveform_data', 
                       help='输出文件名前缀 (默认:waveform_data)')
    parser.add_argument('--model', type=str, default='Model/waveform_cnn_256.h5', 
                       help='模型文件路径 (默认:waveform_cnn_256.h5)')
    parser.add_argument('--plot', default=True , action='store_true', 
                       help='是否显示波形图')

    args = parser.parse_args()

    # 1. 生成波形
    type_map = {'sine': 0, 'triangle': 1, 'fsk': 2, 'bpsk': 3}
    waveform = generate_waveform(
        cls=type_map[args.type.lower()],
        length=args.length,
        fs=args.fs,
        freq=args.freq,
        amp=args.amp,
        phase_shift=args.phase
    )

    # 2. 绘制波形
    if args.plot:
        plot_waveform(waveform, args.type, args.fs, args.freq)

    # 3. 导出到C文件
    export_to_c(
        waveform,
        f"waveform_{args.type.lower()}",
        args.output + ".h",
        args.output + ".c"
    )

    # 4. 用模型验证波形
    if os.path.exists(args.model):
        try:
            model = load_model(args.model)
            print(f"\n🔧 正在使用模型验证: {args.model}")
            validate_with_model(
                model=model,
                waveform=waveform,
                waveform_type=args.type,
                noise_level=args.noise,
                plot=args.plot
            )
        except Exception as e:
            print(f"⚠️ 模型验证失败: {str(e)}")
    else:
        print(f"⚠️ 模型文件不存在，跳过验证: {args.model}")

if __name__ == "__main__":
    main()