import os
import whisper
import torch

# 1. 環境設定
os.environ["PATH"] += os.pathsep + r"D:\tools_exe\ffmpeg-7.0.1-full_build\bin"
audio_file = "audio.mp3"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. 載入模型
model = whisper.load_model("turbo", device=device)

print(f"🚀 啟動 GPU 加速轉錄 (Device: {device})...")

# 3. 執行轉錄
result = model.transcribe(
    audio_file,
    language="ja",
    task="transcribe",
    verbose=True,                  # 恢復：讓你看到即時進度
    # 恢復你的原始 Prompt
    initial_prompt="""
這是中文與日文混合的語音學習內容。
日文請使用標準書寫（漢字優先，例如：先生，並且在後標註假名）。
禁止使用羅馬字或錯誤音譯。
""",
    # --- 解決 9999 與重複的關鍵參數 ---
    condition_on_previous_text=False, # 防止錯誤連鎖反應
    no_speech_threshold=0.6,          # 略過雜訊/音樂
    temperature=0                     # 讓模型輸出更穩定，不亂猜
)

# 4. 輸出檔案（解決文字黏在一起的問題）
base_name = os.path.splitext(os.path.basename(audio_file))[0]
output_file = base_name + ".txt"

with open(output_file, "w", encoding="utf-8") as f:
    for segment in result["segments"]:
        line = f"[{segment['start']:.2f} --> {segment['end']:.2f}] {segment['text'].strip()}"
        f.write(line + "\n")

print(f"\n✅ 任務完成！結果已存入：{output_file}")