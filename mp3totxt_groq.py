import os
from groq import Groq
from pydub import AudioSegment
import math

# 1. 環境與 API 設定
# 請注意：不要在程式碼中公開你的 API Key，建議使用環境變數
# GROQ_API_KEY = "" 
os.environ["PATH"] += os.pathsep + r"D:\tools_exe\ffmpeg-7.0.1-full_build\bin"

client = Groq(api_key=GROQ_API_KEY)

input_folder = "input_mp3"
output_folder = "output_txt_groq"

if not os.path.exists(input_folder):
    os.makedirs(input_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def transcribe_audio_segment(file_path, prompt, start_offset_sec=0):
    """處理單個音檔分段並回傳轉錄結果"""
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(file_path), audio_file.read()),
            model="whisper-large-v3",
            prompt=prompt,
            response_format="verbose_json",
            language="ja",
            temperature=0
        )
    
    results = []
    for segment in transcription.segments:
        start = segment['start'] + start_offset_sec
        end = segment['end'] + start_offset_sec
        text = segment['text'].strip()
        t_s = f"{int(start//60):02}:{int(start%60):02}"
        t_e = f"{int(end//60):02}:{int(end%60):02}"
        results.append(f"[{t_s} --> {t_e}] {text}")
    return results

# 2. 處理檔案
audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp3")]

for filename in audio_files:
    input_path = os.path.join(input_folder, filename)
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_folder, f"{base_name}.txt")
    
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    prompt_text = "這是中文與日文混合的語音學習內容。日文請使用標準書寫（漢字優先，例如：先生，並且在後標註假名）。禁止使用羅馬字或錯誤音譯。"

    all_transcribed_lines = []

    # 判斷是否需要切割 (設定 24MB 為安全門檻)
    if file_size_mb > 24:
        print(f"📦 檔案 {filename} 過大 ({file_size_mb:.2f}MB)，啟動自動切割...")
        audio = AudioSegment.from_file(input_path)
        duration_ms = len(audio)
        
        # 計算需要切成幾份 (每份大約 20MB 以確保安全)
        num_segments = math.ceil(file_size_mb / 20)
        segment_ms = duration_ms // num_segments
        
        for i in range(num_segments):
            start_ms = i * segment_ms
            end_ms = min((i + 1) * segment_ms, duration_ms)
            
            temp_filename = f"temp_{i}.mp3"
            audio[start_ms:end_ms].export(temp_filename, format="mp3")
            
            print(f"  ⏳ 正在轉錄分段 {i+1}/{num_segments}...")
            lines = transcribe_audio_segment(temp_filename, prompt_text, start_offset_sec=start_ms/1000)
            all_transcribed_lines.extend(lines)
            
            os.remove(temp_filename) # 刪除暫存檔
    else:
        print(f"--- ⏳ 正在處理: {filename} ---")
        all_transcribed_lines = transcribe_audio_segment(input_path, prompt_text)

    # 3. 寫入檔案
    with open(output_path, "w", encoding="utf-8") as f:
        for line in all_transcribed_lines:
            f.write(line + "\n")
    
    print(f"✅ 完成！結果儲存至: {output_path}\n")

print("✨ 所有任務執行完畢！")
