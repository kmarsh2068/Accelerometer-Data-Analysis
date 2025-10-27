import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import tkinter as tk
from tkinter import filedialog
import numpy as np
import csv

def inspect_txt(file_path, n=10):
    df = pd.read_csv(file_path, sep='\t')
    print("\n[TXT INSPECTION]")
    print("First few rows:")
    print(df.head(n))
    print("\nColumn names:")
    print(df.columns.tolist())
    return df

def choose_file():
    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(
        title="Select accelerometer file",
        filetypes=[
            ("CSV and TXT Files", "*.csv *.txt"),
            ("CSV Files", "*.csv"),
            ("Text Files", "*.txt"),
            ("All Files", "*.*"),
        ]
    )
    root.destroy()
    return filename

def parse_wit_ble_txt(file_path):
    df = pd.read_csv(file_path, sep='\t')
    required_cols = ["time", "AccX(g)", "AccY(g)", "AccZ(g)"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print("Missing columns: {}".format(", ".join(missing_cols)))
        return [], pd.DataFrame()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for col in ["AccX(g)", "AccY(g)", "AccZ(g)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols)
    if df.empty:
        return [], pd.DataFrame()

    start_time = df["time"].iloc[0]
    times = (df["time"] - start_time).dt.total_seconds().tolist()
    return times, df[["AccX(g)", "AccY(g)", "AccZ(g)"]]

def read_csv_data(filename):
    df = pd.read_csv(filename)
    if df.shape[1] < 2:
        print("CSV file does not have enough columns.")
        return [], pd.DataFrame()
    
    time_col = df.columns[0]
    df = df.rename(columns={df.columns[1]: "AccX(g)"})
    df["time"] = pd.to_numeric(df[time_col], errors="coerce")
    df["AccX(g)"] = pd.to_numeric(df["AccX(g)"], errors="coerce")
    df = df.dropna()
    times = df["time"].tolist()
    return times, df[["AccX(g)"]]

def find_peaks_filtered(signal, prominence=0.25, distance=25, trim_start=0.05, trim_end=0.20):
    n = len(signal)
    start = int(n * trim_start)
    end = int(n * (1 - trim_end))
    segment = signal[start:end].reset_index(drop=True)
    peaks, _ = find_peaks(segment, prominence=prominence, distance=distance)
    peaks = [p + start for p in peaks]
    return peaks, len(peaks)

def compute_step_times(times, peaks):
    if len(peaks) < 2:
        return [], None
    
    intervals = [times[peaks[i+1]] - times[peaks[i]] for i in range(len(peaks)-1)]
    avg_interval = sum(intervals) / len(intervals)
    return intervals, avg_interval

def plot_three_axes(times, df, filename, trim_start=0.05, trim_end=0.20):
    axes = [col for col in df.columns if "Acc" in col]
    colors = ["blue", "green", "red", "purple"]
    plt.figure(figsize=(10, 6))

    for col, color in zip(axes, colors):
        accels = df[col].reset_index(drop=True)
        mean_val = np.mean(accels)
        std_val = np.std(accels)
        if std_val == 0:
            print(f"Warning: zero standard deviation for {col}, skipping.")
            continue
        z_scores = (accels - mean_val) / std_val

        positive_steps = z_scores[z_scores > 0]
        negative_steps = z_scores[z_scores < 0]

        pos_mean = positive_steps.mean() if len(positive_steps) > 0 else 0
        neg_mean = negative_steps.mean() if len(negative_steps) > 0 else 0
        combined_mean = (pos_mean + neg_mean) / 2

        print(f"{col}: Mean (pos)={pos_mean:.3f}, Mean (neg)={neg_mean:.3f}, Combined avg={combined_mean:.3f}")

        peaks, total_peaks = find_peaks_filtered(z_scores, trim_start=trim_start, trim_end=trim_end)
        intervals, avg_interval = compute_step_times(times, peaks)

        if avg_interval is not None:
            print(f"{col}: {total_peaks} steps, Avg step interval = {avg_interval:.2f} s\n")
        else:
            print(f"{col}: Not enough steps detected.\n")

        plt.plot(times[:len(z_scores)], z_scores, label=f"{col} (steps={total_peaks})", color=color)
        plt.plot([times[i] for i in peaks], [z_scores.iloc[i] for i in peaks], "o", color=color)

        n = len(z_scores)
        start = int(n * trim_start)
        end = int(n * (1 - trim_end))
        plt.axvspan(times[0], times[start], color=color, alpha=0.1)
        plt.axvspan(times[end], times[-1], color=color, alpha=0.1)

    title_name = os.path.splitext(os.path.basename(filename))[0]
    plt.title(f"Z-Score Normalized Accelerometer Data: {title_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Acceleration (z-score)")
    plt.grid(True)
    plt.legend()
    plt.show()

def main_loop():
    while True:
        filename = choose_file()
        if not filename:
            break

        ext = os.path.splitext(filename)[1].lower()
        if ext == ".txt":
            times, df_acc = parse_wit_ble_txt(filename)
            if not df_acc.empty:
                plot_three_axes(times, df_acc, filename)
            else:
                print("No valid data in TXT file.")
        elif ext == ".csv":
            times, df_acc = read_csv_data(filename)
            if not df_acc.empty:
                plot_three_axes(times, df_acc, filename)
            else:
                print("No valid data in CSV file.")
        else:
            print("Unsupported file type. Please select .csv or .txt")

        again = input("Load another file? (y/n): ").strip().lower()
        if again != "y":
            break

if __name__ == "__main__":
    main_loop()
