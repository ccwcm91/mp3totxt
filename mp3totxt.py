import os
import whisper
import torch

# 1. 修正路徑與變數定義
os.environ["PATH"] += os.pathsep + r"D:\tools_exe\ffmpeg-7.0.1-full_build\bin"
audio_file = "audio.mp3"  # 先定義變數，後面才不會報錯

print("🔥 程式有執行")

# 檢查 GPU 是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"運行設備: {device}")

print("🚀 載入模型...")
# 使用 turbo 模型（這是 OpenAI 官方 Whisper 剛更新支援的模型名）
model = whisper.load_model("turbo") 

print("🚀 開始轉錄...")

result = model.transcribe(
    audio_file,
    language="ja",             # 修正點：必須是字串 "ja" 或 "japanese"
    task="transcribe",
    verbose=True,
    initial_prompt="""
這是一段日文教學，包含大量日文單字與中文解說。
請保留日文原文，不要將日文翻譯成中文。
日文內容請包含漢字與假名，例如：これは本です。
"""
)

print("✅ 完成")

# =========================
# 📁 輸出同名 txt
# =========================
base_name = os.path.splitext(os.path.basename(audio_file))[0]
output_file = base_name + ".txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(result["text"])

print(f"💾 已輸出：{output_file}")