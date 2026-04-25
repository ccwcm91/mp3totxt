import os
import math
from pydub import AudioSegment

# 1. 環境設定
# 請確保路徑正確指向你的 ffmpeg bin 資料夾
os.environ["PATH"] += os.pathsep + r"D:\tools_exe\ffmpeg-7.0.1-full_build\bin"

input_folder = "input_mp3"
output_folder = "output_mp3_split"
TARGET_SIZE_MB = 20  # 設定目標大小為 20MB

if not os.path.exists(input_folder):
    os.makedirs(input_folder)
    print(f"📁 已建立輸入資料夾：{input_folder}")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. 獲取檔案清單
audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp3")]

for filename in audio_files:
    input_path = os.path.join(input_folder, filename)
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    
    if file_size_mb <= TARGET_SIZE_MB:
        print(f"ℹ️  跳過 {filename} (大小僅 {file_size_mb:.2f}MB，未超過限制)")
        continue

    print(f"📦 正在處理: {filename} ({file_size_mb:.2f}MB)")
    
    # 載入音檔
    audio = AudioSegment.from_file(input_path)
    duration_ms = len(audio)
    
    # 計算需要切成幾份 (稍微留一點餘裕，每份約 19MB)
    num_segments = math.ceil(file_size_mb / 19)
    segment_ms = duration_ms // num_segments
    
    base_name = os.path.splitext(filename)[0]
    
    for i in range(num_segments):
        start_ms = i * segment_ms
        end_ms = min((i + 1) * segment_ms, duration_ms)
        
        # 建立輸出的檔名 (例如: 檔名_part1.mp3)
        output_filename = f"{base_name}_part{i+1}.mp3"
        output_path = os.path.join(output_folder, output_filename)
        
        # 導出切割後的檔案
        print(f"  Exporting: {output_filename}")
        audio[start_ms:end_ms].export(output_path, format="mp3")

    print(f"✅ {filename} 切割完成，共分成 {num_segments} 個檔案。\n")

print("✨ 所有切割任務執行完畢！")
