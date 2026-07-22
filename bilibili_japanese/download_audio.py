import sys
import os
import yt_dlp

# Force UTF-8 output encoding for Windows terminal compatibility
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

BV_URL = "https://www.bilibili.com/video/BV1Qx411D7oA/"

def download_audio(url, playlist_items=None, output_dir="downloads"):
    """
    Downloads audio streams from Bilibili video/playlist.
    :param url: Bilibili video URL
    :param playlist_items: String specifying item ranges e.g. "1" for first episode, "1-5", "1,3,5", or None for all
    :param output_dir: Directory to save downloaded audio files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',  # Download best standalone audio stream (m4a/aac)
        'outtmpl': os.path.join(output_dir, '%(playlist_index)03d_%(title)s.%(ext)s'),
        'playlist_items': playlist_items if playlist_items else None,
        'writesubtitles': False,
        'writeautomaticsub': False,
        'ignoreerrors': True,
        'no_warnings': False,
        'retries': 10,
        'fragment_retries': 10,
        'socket_timeout': 30,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://www.bilibili.com/',
        },
        'progress_hooks': [progress_hook],
    }

    print(f"[*] 正在準備下載: {url}")
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
        print(f"\n[已完成] {filename}\n")

if __name__ == '__main__':
    items = "1"
    if len(sys.argv) > 1:
        items = sys.argv[1]
        if items.lower() == 'all':
            items = None

    download_audio(BV_URL, playlist_items=items)
