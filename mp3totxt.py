import os
from faster_whisper import WhisperModel
import torch

# 1. 環境設定
os.environ["PATH"] += os.pathsep + r"D:\tools_exe\ffmpeg-7.0.1-full_build\bin"
audio_file = "audio.mp3"

# 修改後的環境判斷
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    # 如果 float16 報錯，通常改用 float32 絕對沒問題
    # 或者嘗試 "int8_float16" (在某些舊卡上比 float16 相容性更好)
    compute_type = "float32" 
else:
    compute_type = "int8"

print(f"🚀 啟動 GPU 加速轉錄 (Device: {device}, Precision: {compute_type})...")
model = WhisperModel("turbo", device=device, compute_type=compute_type)

# 3. 執行轉錄
# faster-whisper 的 transcribe 會回傳一個 generator，效率更高
segments, info = model.transcribe(
    audio_file,
    language="ja",
    task="transcribe",
    initial_prompt="""
這是中文與日文混合的語音學習內容。
日文請使用標準書寫（漢字優先，例如：先生，並且在後標註假名）。
禁止使用羅馬字或錯誤音譯。
""",
    # --- 對應原始參數的設定 ---
    condition_on_previous_text=False, 
    no_speech_threshold=0.8,
    beam_size=5,        # 相當於穩定度，預設 5
    temperature=0,      # 固定輸出，不亂猜
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=1000),
)

# 4. 輸出檔案
base_name = os.path.splitext(os.path.basename(audio_file))[0]
output_file = base_name + ".txt"

print(f"偵測到語言: {info.language} (信心度: {info.language_probability:.2f})")

with open(output_file, "w", encoding="utf-8") as f:
    for segment in segments:
        # 即時印出進度 (對應 verbose=True)
        line = f"[{segment.start:.2f} --> {segment.end:.2f}] {segment.text.strip()}"
        print(line)
        f.write(line + "\n")

print(f"\n✅ 任務完成！結果已存入：{output_file}")