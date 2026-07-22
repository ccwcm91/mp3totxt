# ============================================================
# 早安日語 高精度逐字稿產生器 v2
# 改善重點：
#   1. language=None → 自動偵測語言（解決中日混合誤判）
#   2. temperature 多段 fallback → 遇到辨識失敗自動重試
#   3. compression_ratio_threshold → 偵測重複幻覺並捨棄重辨
#   4. no_speech_threshold 調低 → 更積極過濾靜音背景音樂段落
# ============================================================

# ── 安裝套件（只需第一次）──────────────────────────────────
!pip install -q faster-whisper tqdm torch

import os
import torch
from tqdm import tqdm
from faster_whisper import WhisperModel
from google.colab import files

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1：上傳 MP3
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("📁 [1/4] 請選擇並上傳 MP3 檔案...")
uploaded = files.upload()

if not uploaded:
    raise ValueError("❌ 未偵測到上傳檔案，請重新執行此格。")

filename = list(uploaded.keys())[0]
print(f"✅ 已接收：{filename}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2：載入模型
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"

print(f"\n🚀 [2/4] 載入模型 'large-v3' (裝置: {device}, 精度: {compute_type})...")
model = WhisperModel("large-v3", device=device, compute_type=compute_type)
print("✅ 模型載入完成！")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3：開始轉錄
#
# 關鍵參數說明：
#   language=None                   → 不鎖定語言，讓模型自動判斷每段
#                                     避免把中文硬解釋成日文
#   temperature=[0, 0.2, 0.4, ...]  → 貪婪解碼失敗時，自動升溫重試
#                                     類似「重新思考」的機制
#   compression_ratio_threshold=2.2 → 壓縮率過高代表文字大量重複
#                                     超過此閾值 → 自動捨棄此段重辨
#                                     直接解決「あそこあそこあそこ...」問題
#   no_speech_threshold=0.6         → 靜音/背景音樂超過此信心值時跳過
#   condition_on_previous_text=False→ 不讓前文影響後文，防止迴圈蔓延
#   log_prob_threshold=-1.0         → 平均對數概率低於此值視為失敗段
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n🎵 [3/4] 開始辨識音訊：{filename} ...\n")

segments_gen, info = model.transcribe(
    filename,
    language=None,                           # 自動偵測語言
    task="transcribe",
    initial_prompt=None,                     # 不給提示，避免幻覺
    condition_on_previous_text=False,        # 防止迴圈蔓延
    no_speech_threshold=0.6,                 # 積極過濾靜音
    beam_size=5,
    temperature=[0, 0.2, 0.4, 0.6, 0.8, 1.0],  # 多溫度 fallback
    compression_ratio_threshold=2.2,         # 偵測重複幻覺
    log_prob_threshold=-1.0,                 # 低信心段落捨棄
    vad_filter=True,
    vad_parameters=dict(
        min_silence_duration_ms=500,         # 500ms 靜音即切段
        threshold=0.5,
    ),
)

total_duration = info.duration
m_total = int(total_duration // 60)
s_total = int(total_duration % 60)
print(f"⏱️  音訊總長度：{m_total:02d}:{s_total:02d}（{total_duration/60:.2f} 分鐘）")
print(f"🌐 偵測語言：{info.language}（信心度：{info.language_probability:.1%}）\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4：寫出 TXT 並即時顯示
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
txt_filename = os.path.splitext(filename)[0] + "_v2.txt"

REPEAT_LIMIT = 5  # 同一詞連續出現超過此次數，視為幻覺並過濾

def is_hallucination(text: str, limit: int = REPEAT_LIMIT) -> bool:
    """偵測單一 segment 內是否有大量重複（備援過濾）"""
    if not text:
        return False
    words = text.split()
    if len(words) < limit:
        return False
    # 若最常出現的詞佔比超過 60% 且重複超過 limit 次，視為幻覺
    from collections import Counter
    counter = Counter(words)
    most_common_word, most_common_count = counter.most_common(1)[0]
    return most_common_count >= limit and (most_common_count / len(words)) > 0.6

skipped_count = 0
written_count = 0

with open(txt_filename, "w", encoding="utf-8") as f, \
     tqdm(total=round(total_duration), unit="s", desc="轉錄進度", ncols=80) as pbar:

    last_pos = 0.0

    for segment in segments_gen:
        text = segment.text.strip()

        # 備援：如果 compression_ratio_threshold 沒擋住，這裡再過一次
        if is_hallucination(text):
            skipped_count += 1
            tqdm.write(f"  ⚠️  [跳過幻覺段落] {int(segment.start//60):02d}:{int(segment.start%60):02d} → {text[:40]}...")
            # 更新進度條但不寫入
            advance = round(segment.end - last_pos)
            if advance > 0:
                pbar.update(min(advance, round(total_duration) - round(last_pos)))
                last_pos = segment.end
            continue

        # 格式：[分:秒 --> 分:秒] 內容
        t_s = f"{int(segment.start//60):02d}:{int(segment.start%60):02d}"
        t_e = f"{int(segment.end//60):02d}:{int(segment.end%60):02d}"
        line = f"[{t_s} --> {t_e}] {text}"

        tqdm.write(line)
        f.write(line + "\n")
        written_count += 1

        advance = round(segment.end - last_pos)
        if advance > 0:
            pbar.update(min(advance, round(total_duration) - round(last_pos)))
            last_pos = segment.end

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 完成報告
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'='*50}")
print(f"✨ [4/4] 轉錄完成！")
print(f"   📝 寫入段落：{written_count} 段")
print(f"   ⚠️  過濾幻覺：{skipped_count} 段")
print(f"   💾 輸出檔案：{txt_filename}")
print(f"{'='*50}\n")

files.download(txt_filename)
print("⬇️  已自動觸發下載！")
