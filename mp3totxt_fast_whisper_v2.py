import os
import torch
from collections import Counter
from faster_whisper import WhisperModel
from tqdm import tqdm

# ============================================================
# 1. 環境與資料夾設定
# ============================================================
# 如果您的電腦 ffmpeg 未加入系統 PATH，請取消下一行的註解並修改為您的 ffmpeg bin 路徑：
# os.environ["PATH"] += os.pathsep + r"D:\tools_exe\ffmpeg-7.0.1-full_build\bin"

input_folder = "input_mp3"
output_folder = "output_txt_v2"

# 建立資料夾
if not os.path.exists(input_folder):
    os.makedirs(input_folder)
    print(f"📁 已建立輸入資料夾：{input_folder}，請將 MP3 放入後重新執行。")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ============================================================
# 2. 載入模型 (針對低 VRAM 顯卡優化)
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

# 【顯卡記憶體優化說明】：
# 1. 模型選用 "turbo"：相比 large-v3，參數量大幅縮減，速度更快且 VRAM 需求大幅降低。
# 2. compute_type 選用 "int8_float16" (或 "int8")：
#    將模型權重進行 8-bit 量化，能將 VRAM 佔用直接壓到 2GB~3GB 左右，避免 8G 以下顯卡爆記憶體 (OOM)。
compute_type = "int8_float16" if device == "cuda" else "int8"

print(f"🚀 載入輕量高精度模型 'turbo' (裝置: {device}, 精度: {compute_type})...")
model = WhisperModel("turbo", device=device, compute_type=compute_type)
print("✅ 模型載入完成！\n")

# ============================================================
# 3. 備援防幻覺過濾函數
# ============================================================
REPEAT_LIMIT = 5  # 同一詞連續出現超過此次數，視為幻覺並過濾

def is_hallucination(text: str, limit: int = REPEAT_LIMIT) -> bool:
    """偵測單一 segment 內是否有大量重複（備援過濾）"""
    if not text:
        return False
    words = text.split()
    if len(words) < limit:
        return False
    # 若最常出現的詞佔比超過 60% 且重複超過 limit 次，視為幻覺
    counter = Counter(words)
    most_common_word, most_common_count = counter.most_common(1)[0]
    return most_common_count >= limit and (most_common_count / len(words)) > 0.6

# ============================================================
# 4. 取得檔案清單與批次處理
# ============================================================
audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp3")]

if not audio_files:
    print(f"❌ 在 {input_folder} 中找不到任何 MP3 檔案，請放入檔案後再執行。")
else:
    print(f"🎵 找到 {len(audio_files)} 個檔案，準備開始轉錄...\n")

for idx, filename in enumerate(audio_files, 1):
    input_path = os.path.join(input_folder, filename)
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_folder, f"{base_name}_v2.txt")

    print(f"{'='*60}")
    print(f"⏳ [{idx}/{len(audio_files)}] 正在處理：{filename}")
    print(f"{'='*60}")

    # 高精度防幻覺 transcribe 參數
    segments_gen, info = model.transcribe(
        input_path,
        language=None,                             # 自動偵測語言（避免中日混合硬翻）
        task="transcribe",
        initial_prompt=None,                       # 不給提示，避免產生固定幻覺
        condition_on_previous_text=False,          # 防止前文錯誤影響後文
        no_speech_threshold=0.6,                   # 積極過濾靜音/背景音樂
        beam_size=5,
        temperature=[0, 0.2, 0.4, 0.6, 0.8, 1.0],  # 多段升溫 fallback
        compression_ratio_threshold=2.2,           # 壓縮率偵測重複幻覺並自動捨棄重辨
        log_prob_threshold=-1.0,                   # 低信心段落捨棄
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,           # 500ms 靜音即切段
            threshold=0.5,
        ),
    )

    total_duration = info.duration
    m_total = int(total_duration // 60)
    s_total = int(total_duration % 60)
    print(f"⏱️  音訊總長度：{m_total:02d}:{s_total:02d}（{total_duration/60:.2f} 分鐘）")
    print(f"🌐 偵測主要語言：{info.language}（信心度：{info.language_probability:.1%}）\n")

    skipped_count = 0
    written_count = 0
    last_pos = 0.0

    # 寫入文字檔與視覺化進度條
    with open(output_path, "w", encoding="utf-8") as f, \
         tqdm(total=round(total_duration), unit="s", desc=f"轉錄進度 ({filename[:15]}...)", ncols=80) as pbar:

        for segment in segments_gen:
            text = segment.text.strip()

            # Python 層級備援過濾：如果模型層級沒擋住重複字，這裡拋棄
            if is_hallucination(text):
                skipped_count += 1
                tqdm.write(f"  ⚠️  [跳過幻覺段落] {int(segment.start//60):02d}:{int(segment.start%60):02d} → {text[:40]}...")
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

    print(f"\n✨ 完成此檔案！")
    print(f"   📝 寫入段落：{written_count} 段")
    print(f"   ⚠️  過濾幻覺：{skipped_count} 段")
    print(f"   💾 輸出位置：{output_path}\n")

print("🎉🎉 所有任務執行完畢！")