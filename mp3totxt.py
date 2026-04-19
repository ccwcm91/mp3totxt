import os
from faster_whisper import WhisperModel
import torch

# 1. 環境設定
os.environ["PATH"] += os.pathsep + r"D:\tools_exe\ffmpeg-7.0.1-full_build\bin"

# 設定輸入與輸出資料夾路徑
input_folder = "input_mp3"
output_folder = "output_txt"

# 如果資料夾不存在則建立
if not os.path.exists(input_folder):
    os.makedirs(input_folder)
    print(f"📁 已建立輸入資料夾：{input_folder}，請將 MP3 放入後重新執行。")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. 模型載入 (放在迴圈外，只需執行一次)
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float32" if device == "cuda" else "int8"

print(f"🚀 載入模型中 (Device: {device}, Precision: {compute_type})...")
model = WhisperModel("turbo", device=device, compute_type=compute_type)

# 3. 獲取資料夾內所有 mp3 檔案清單
audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp3")]

if not audio_files:
    print(f"❌ 在 {input_folder} 中找不到任何 MP3 檔案。")
else:
    print(f"🎵 找到 {len(audio_files)} 個檔案，準備開始轉錄...\n")

# 4. 迴圈處理每一個檔案
for filename in audio_files:
    input_path = os.path.join(input_folder, filename)
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_folder, f"{base_name}.txt")
    
    print(f"--- ⏳ 正在處理: {filename} ---")
    
    # 執行轉錄
    segments, info = model.transcribe(
        input_path,
        language="ja",
        task="transcribe",
        initial_prompt="""
這是中文與日文混合的語音學習內容。
日文請使用標準書寫（漢字優先，例如：先生，並且在後標註假名）。
禁止使用羅馬字或錯誤音譯。
""",
        condition_on_previous_text=False, 
        no_speech_threshold=0.8,
        beam_size=5,
        temperature=0,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=1000),
    )

    # 寫入檔案
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            line = f"[{segment.start:.2f} --> {segment.end:.2f}] {segment.text.strip()}"
            print(line) # 如果想在螢幕看即時進度可以取消註解
            f.write(line + "\n")
    
    print(f"✅ 完成！結果儲存至: {output_path}\n")

print("✨ 所有任務執行完畢！")