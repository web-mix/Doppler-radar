import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# 基本定数
C = 3.0e8  # 光速[m/s]
F_CARRIER = 24.0e9  # 搬送波周波数[Hz]
SAMPLE_RATE = 200.0  # サンプリングレート[Hz]

def read_iq_data(file_path):
    """CSVからI/Qデータを読み込み、中心を0に調整する"""
    data = pd.read_csv(file_path, encoding='shift_jis')
    I = data['Idata'].values
    Q = data['Qdata'].values
    
    # 中心を0付近に調整
    I_mean = np.mean(I)
    Q_mean = np.mean(Q)
    I_adjusted = I - I_mean
    Q_adjusted = Q - Q_mean
    
    return I_adjusted, Q_adjusted

def smooth_and_interpolate(I, Q, kernel_size=5):
    """I/Q信号の平滑化と振幅小さい点の補間"""
    # メディアンフィルタで平滑化
    I_smooth = medfilt(I, kernel_size=kernel_size)
    Q_smooth = medfilt(Q, kernel_size=kernel_size)

    # 振幅計算
    amplitude = np.sqrt(I_smooth**2 + Q_smooth**2)
    min_amp = np.percentile(amplitude, 5)  # 下位5%を閾値に
    mask = amplitude > min_amp

    # 欠損点は前後で線形補間
    I_interp = np.copy(I_smooth)
    Q_interp = np.copy(Q_smooth)
    if not np.all(mask):  # 補間が必要な場合のみ
        idx = np.arange(len(I))
        I_interp[~mask] = np.interp(idx[~mask], idx[mask], I_smooth[mask])
        Q_interp[~mask] = np.interp(idx[~mask], idx[mask], Q_smooth[mask])

    return I_interp, Q_interp

def remove_outliers_delta_phase(delta_phase, threshold=3.0):
    """delta_phaseの外れ値（スパイク）を中央値±threshold×標準偏差でクリッピング"""
    median = np.median(delta_phase)
    std = np.std(delta_phase)
    lower = median - threshold * std
    upper = median + threshold * std
    delta_phase_clipped = np.clip(delta_phase, lower, upper)
    return delta_phase_clipped

def calculate_displacement(I, Q):
    """I/Qデータから位相変化、速度、位置変位を計算する"""
    # 1. 平滑化＆補間
    # I_proc, Q_proc = smooth_and_interpolate(I, Q, kernel_size=5)
    I_proc, Q_proc = I, Q

    # 2. 位相の計算とアンラッピング（しきい値調整）
    phase_angles = np.arctan2(Q_proc, I_proc)
    phase_angles_unwrapped = np.unwrap(phase_angles, discont=np.pi/8)  # π/2以上のジャンプを補正

    # 3. 位相の時間微分を計算（差分で近似）
    delta_phase = np.diff(phase_angles_unwrapped)

    # 4. delta_phaseの外れ値除去
    delta_phase_clean = remove_outliers_delta_phase(delta_phase, threshold=3.0)

    # 5. ドップラーシフトによる周波数差に変換
    doppler_freq = delta_phase_clean * SAMPLE_RATE / (2 * np.pi)

    # 6. 速度計算
    velocity = doppler_freq * C / (2 * F_CARRIER)

    # 7. 時間配列準備
    time = np.arange(len(delta_phase_clean)) / SAMPLE_RATE

    # 8. 速度の積分（累積和）で位置変位を求める
    displacement = np.cumsum(velocity) / SAMPLE_RATE

    return time, phase_angles[:-1], phase_angles_unwrapped[:-1], delta_phase_clean, velocity, displacement

def save_results_to_csv(file_path, time, velocity, displacement):
    """計算結果をCSVに保存"""
    results = pd.DataFrame({
        'Time': time,
        'Velocity': velocity,
        'Displacement': displacement
    })
    results.to_csv(file_path, index=False)

def plot_results(time, phase_angles, phase_angles_unwrapped, delta_phase, velocity, displacement):
    """位相（アンラップ前・後）、位相変化、速度、位置変位をプロット"""
    plt.figure(figsize=(12, 12))
    
    # アンラップ前・後の位相角のプロット
    plt.subplot(2, 1, 1)
    t_phase = np.arange(len(phase_angles)) / SAMPLE_RATE
    plt.plot(t_phase, phase_angles, label='Raw Phase Angles', color='blue', alpha=0.5)
    plt.plot(t_phase, phase_angles_unwrapped, label='Unwrapped Phase Angles', color='red', linewidth=1)
    plt.xlabel('Time [s]')
    plt.ylabel('Phase Angle [rad]')
    plt.legend()
    plt.title('Phase Angles (Raw & Unwrapped)')

    # 位相変化のプロット
    # plt.subplot(5, 1, 2)
    # plt.plot(time, delta_phase, label='Delta Phase', color='purple')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Delta Phase [rad]')
    # plt.legend()

    # 速度のプロット
    # plt.subplot(5, 1, 3)
    # plt.plot(time, velocity, label='Velocity', color='green')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Velocity [m/s]')
    # plt.legend()
    
    # 位置変位のプロット
    plt.subplot(5, 1, 4)
    plt.plot(time, displacement, label='Displacement', color='orange')
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.legend()

    # 余白
    plt.tight_layout()
    plt.show()

def main(input_csv, output_csv):
    I, Q = read_iq_data(input_csv)
    time, phase_angles, phase_angles_unwrapped, delta_phase, velocity, displacement = calculate_displacement(I, Q)
    save_results_to_csv(output_csv, time, velocity, displacement)
    plot_results(time, phase_angles, phase_angles_unwrapped, delta_phase, velocity, displacement)

# ファイルパス指定
# input_csv = 'No_250109102033_rawSig.csv'
# output_csv = 'No_250109102033_rawSig_results.csv'
# input_csv = 'abe2_240116141133_rawSig.csv'
# output_csv = 'abe2_240116141133_rawSig_results.csv'

import tkinter as tk
from tkinter import filedialog

# GUIでファイル選択
def choose_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title='CSVファイルを選択してください',
        filetypes=[("CSV Files", "*.csv")]
    )
    return file_path

input_csv = choose_file()
output_csv = input_csv.replace(".csv", "_results.csv")


main(input_csv, output_csv)
