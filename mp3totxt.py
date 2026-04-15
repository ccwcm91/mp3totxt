print("🔥 程式有執行")

import os
import whisper

os.environ["PATH"] += os.pathsep + r"D:\tools_exe\ffmpeg-7.0.1-full_build\bin"

model = whisper.load_model("small") # tiny / base / small / medium / large

print("🚀 開始轉錄...")

result = model.transcribe("audio.mp3", verbose=True)

print("✅ 完成")
print(result["text"])