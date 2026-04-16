print("🔥 程式有執行")

import os
import whisper

os.environ["PATH"] += os.pathsep + r"D:\tools_exe\ffmpeg-7.0.1-full_build\bin"

model = whisper.load_model("large") # tiny / base / small / medium / large

print("🚀 開始轉錄...")

result = model.transcribe(
    "audio.mp3",
    language=ja,
    task="transcribe",
    verbose=True,
    initial_prompt="""
這是中文與日文混合的語音學習內容。
日文請使用標準書寫（漢字優先，例如：先生，並且在後標註假名）。
禁止使用羅馬字或錯誤音譯。
"""
)

print("✅ 完成")
print(result["text"])

# =========================
# 📁 輸出同名 txt
# =========================

base_name = os.path.splitext(os.path.basename(audio_file))[0]
output_file = base_name + ".txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(result["text"])

print(f"💾 已輸出：{output_file}")