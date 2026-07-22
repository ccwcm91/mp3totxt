import sys
import os
import imageio_ffmpeg
import yt_dlp

# 確保 Windows terminal 輸出 UTF-8 編碼
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

BV_URL = "https://www.bilibili.com/video/BV1Qx411D7oA/"

def download_audio_as_mp3(url, playlist_items=None, output_dir="downloads"):
    """
    下載 Bilibili 影片並自動經由 FFmpeg 轉碼為標準 MP3。
    檔名自動命名為：早安日語XXX_Bilibili.mp3
    """
    os.makedirs(output_dir, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # 檔名範例：早安日語001_Bilibili.mp3
    out_template = os.path.join(output_dir, '早安日語%(playlist_index)03d_Bilibili.%(ext)s')

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_template,
        'playlist_items': playlist_items if playlist_items else None,
        'writesubtitles': False,
        'writeautomaticsub': False,
        'ignoreerrors': True,
        'no_warnings': False,
        'retries': 10,
        'fragment_retries': 10,
        'socket_timeout': 30,
        'ffmpeg_location': ffmpeg_exe,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://www.bilibili.com/',
        },
        'progress_hooks': [progress_hook],
    }

    print(f"[*] 正在下載並轉碼為標準 MP3: {url}")
    if playlist_items:
        print(f"[*] 下載範圍: 第 {playlist_items} 集")
    else:
        print("[*] 下載範圍: 全部集數")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def progress_hook(d):
    if d['status'] == 'downloading':
        percent = d.get('_percent_str', '').strip()
        speed = d.get('_speed_str', '').strip()
        eta = d.get('_eta_str', '').strip()
        filename = os.path.basename(d.get('filename', ''))
        print(f"\r[下載中] {filename} | 進度: {percent} | 速度: {speed} | 剩餘時間: {eta}  ", end='', flush=True)
    elif d['status'] == 'finished':
        filename = os.path.basename(d.get('filename', ''))
        print(f"\n[下載完成，正在轉碼為 MP3...] {filename}\n")

if __name__ == '__main__':
    items = "1"
    if len(sys.argv) > 1:
        items = sys.argv[1]
        if items.lower() == 'all':
            items = None

    download_audio_as_mp3(BV_URL, playlist_items=items)
