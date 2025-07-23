import numpy as np
import h5py
import argparse

def generate_waveform(cls, length, fs, freq, amp=1.0, phase_shift=0):
    duration = length / fs  # 持续时间（秒）
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

def generate_dataset(num_classes, signal_length, fs, samples_per_class,
                     output_file, freq_min, freq_max, amp_min, amp_max):
    X, y = [], []
    for cls in range(num_classes):
        for _ in range(samples_per_class):
            freq = np.random.uniform(freq_min, freq_max)
            amp = np.random.uniform(amp_min, amp_max)
            phase_shift = np.random.uniform(0, 2 * np.pi)
            signal = generate_waveform(cls, signal_length, fs, freq, amp, phase_shift)
            noise = np.random.normal(0, 0.05, signal_length)
            X.append(signal + noise)
            y.append(cls)

    X = np.array(X).astype("float32")[..., np.newaxis]
    y = np.eye(num_classes)[y]

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('X', data=X)
        f.create_dataset('y', data=y)
    print(f"✅ 数据集已保存到 {output_file}，X shape: {X.shape}，y shape: {y.shape}")


def convert_to_cubeai_format(input_file, output_file):
    """
    将自定义格式的HDF5文件转换为X-CUBE-AI兼容格式
    """
    with h5py.File(input_file, 'r') as f_in:
        X = f_in['X'][:]  # 形状 (4000, 256, 1)
        y = f_in['y'][:]  # 形状 (4000, 4) - one-hot编码
        
        # 调整输入维度为 (4000, 256, 1, 1)
        X = X.reshape((X.shape[0], X.shape[1], 1, 1))
        
        # 确保输出是4维one-hot (4000, 1, 1, 4)
        if y.ndim == 2:  # 如果是二维one-hot
            y = y.reshape((y.shape[0], 1, 1, y.shape[1]))
        elif y.ndim == 1:  # 如果是一维类别索引
            y = np.eye(4)[y].reshape((y.shape[0], 1, 1, 4))
        
        with h5py.File(output_file, 'w') as f_out:
            f_out.create_dataset('input', data=X.astype('float32'))
            f_out.create_dataset('output', data=y.astype('int32'))
    
    print(f"✅ 转换完成！X-CUBE-AI格式数据集已保存到 {output_file}")
    print(f"  输入数据维度: {X.shape}, 输出数据维度: {y.shape}")
    
    # 保存NPY文件（保持原始维度）
    np.save(output_file.replace('.h5', '_input.npy'), X.squeeze())  # (4000, 256)
    np.save(output_file.replace('.h5', '_output.npy'), np.argmax(y, axis=3).squeeze())  # (4000,)
    print(f"✅ NPY文件已保存：{output_file.replace('.h5', '_input.npy')} 和 _output.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="波形数据集生成器（固定长度 + 可变频率/幅值/采样率）")
    parser.add_argument('--length', type=int, default=256, help='采样点数，固定长度（默认: 256）')
    parser.add_argument('--fs', type=int, default=1000000, help='采样率 Hz（默认: 256）')
    parser.add_argument('--samples', type=int, default=1000, help='每类波形样本数（默认: 500）')
    parser.add_argument('--freq_min', type=float, default=10000.0, help='最小频率 (Hz)')
    parser.add_argument('--freq_max', type=float, default=300000.0, help='最大频率 (Hz)')
    parser.add_argument('--amp_min', type=float, default=0.5, help='最小幅值')
    parser.add_argument('--amp_max', type=float, default=1.0, help='最大幅值')
    parser.add_argument('--output', type=str, default='waveform_dataset.h5', help='输出 HDF5 文件名')

    args = parser.parse_args()

    generate_dataset(
        num_classes=4,
        signal_length=args.length,
        fs=args.fs,
        samples_per_class=args.samples,
        output_file=args.output,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        amp_min=args.amp_min,
        amp_max=args.amp_max
    )
    convert_to_cubeai_format(args.output, "waveform_dataset_cubeai.h5")
