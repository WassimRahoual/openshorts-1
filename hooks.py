import os
import re
import textwrap
import subprocess
import urllib.request
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def strip_emojis(text):
    """Remove emoji characters that PIL/NotoSerif can't render."""
    emoji_pattern = re.compile(
        "[\U00010000-\U0010ffff"
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\u2600-\u26FF"
        "\u2700-\u27BF"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub('', text).strip()

FONT_URL = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSerif/NotoSerif-Bold.ttf"
FONT_DIR = "fonts"
FONT_PATH = os.path.join(FONT_DIR, "NotoSerif-Bold.ttf")

def download_font_if_needed():
    """Downloads a serif font for the hook text if not present."""
    if not os.path.exists(FONT_DIR):
        os.makedirs(FONT_DIR)
    if not os.path.exists(FONT_PATH):
        print(f"⬇️ Downloading font from {FONT_URL}...")
        try:
            req = urllib.request.Request(
                FONT_URL,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response, open(FONT_PATH, 'wb') as out_file:
                out_file.write(response.read())
            print("✅ Font downloaded.")
        except Exception as e:
            print(f"❌ Failed to download font: {e}")

def create_hook_image(text, target_width, output_image_path="hook_overlay.png", font_scale=1.0):
    """
    Generates a white box with black serif text using pixel-based wrapping.
    target_width: The max width the box should occupy (e.g. 85% of video)
    """
    download_font_if_needed()
    
    # Configuration
    padding_x = 30 # Balanced padding
    padding_y = 25 
    line_spacing = 20 # Increased spacing
    cornerradius = 20
    shadow_offset = (5, 5) 
    shadow_blur = 10
    
    # Font Size Calculation (approx 5% of width - tuned to match Noto Serif Bold metrics in browser)
    base_font_size = int(target_width * 0.05)
    font_size = int(base_font_size * font_scale)
    
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception as e:
        print(f"⚠️ Warning: Could not load font {FONT_PATH}, using default. Error: {e}")
        font = ImageFont.load_default()

    # Wrap text logic (Pixel-based)
    dummy_img = Image.new('RGBA', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    
    max_text_width = target_width - (2 * padding_x)
    
    # Handle manual newlines first
    paragraphs = text.split('\n')
    lines = []
    
    for p in paragraphs:
        if not p.strip():
            lines.append("") 
            continue
            
        words = p.split()
        current_line = []
        
        for word in words:
            # Test if adding word fits
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            w = bbox[2] - bbox[0]
            
            if w <= max_text_width:
                current_line.append(word)
            else:
                # Line full, push current_line and start new
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    # Single word too long? Force it.
                    lines.append(word)
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
    
    # Recalculate true width/height
    max_line_width = 0
    text_heights = []
    
    for line in lines:
        if not line:
            text_heights.append(font_size) # Use font size for empty line height
            continue
            
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        max_line_width = max(max_line_width, w)
        text_heights.append(h)
    
    # Box dimensions
    # We want the box to fit the text exactly + padding
    # Ensure min width for aesthetic reasons if text is short (at least 30% of target)
    box_width = max(max_line_width + (2 * padding_x), int(target_width * 0.3))
    
    # Total Text Height: sum(heights) + spacing * (n-1)
    if not text_heights:
         total_text_height = font_size
    else:
         total_text_height = sum(text_heights) + (len(text_heights) - 1) * line_spacing
         
    box_height = total_text_height + (2 * padding_y)
    
    # Create Final Image with Rounded Corners and Shadow
    # 1. Canvas for Shadow (larger than box)
    canvas_w = box_width + 40
    canvas_h = box_height + 40
    
    img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 2. Draw Shadow
    shadow_box = [
        (20 + shadow_offset[0], 20 + shadow_offset[1]),
        (20 + box_width + shadow_offset[0], 20 + box_height + shadow_offset[1])
    ]
    draw.rounded_rectangle(shadow_box, radius=cornerradius, fill=(0, 0, 0, 100))
    
    # 3. Blur Shadow
    img = img.filter(ImageFilter.GaussianBlur(5))
    
    # 4. Draw White Box (sharper, on top of blurred shadow)
    draw_final = ImageDraw.Draw(img)
    
    main_box = [
        (20, 20),
        (20 + box_width, 20 + box_height)
    ]
    # Semi-transparent white (240/255 alpha ~ 94% opacity)
    draw_final.rounded_rectangle(main_box, radius=cornerradius, fill=(255, 255, 255, 240))
    
    # 5. Draw Text
    current_y = 20 + padding_y - 2 # Minor visual adjustment
    for i, line in enumerate(lines):
        if not line:
            current_y += font_size + line_spacing 
            continue
            
        bbox = draw_final.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        line_h = text_heights[i] if i < len(text_heights) else bbox[3] - bbox[1]
        
        # Center X
        x = 20 + (box_width - line_w) // 2
        
        # Draw Black Text
        draw_final.text((x, current_y), line, font=font, fill="black")
        
        current_y += line_h + line_spacing
        
    img.save(output_image_path)
    return output_image_path, canvas_w, canvas_h

def add_hook_to_video(video_path, text, output_path, position="top", font_scale=1.0):
    """
    Overlays text hook onto video.
    position: 'top', 'center', 'bottom'
    font_scale: float multiplier (1.0 = default)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video {video_path} not found")

    # 1. Probe video width to scale text properly
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', video_path]
        res = subprocess.check_output(cmd).decode().strip()
        # Takes first stream if multiple
        dims = res.split('\n')[0].split('x')
        video_width = int(dims[0])
        video_height = int(dims[1])
    except Exception as e:
        print(f"⚠️ FFprobe failed: {e}. Assuming 1080x1920")
        video_width = 1080
        video_height = 1920
        
    # 2. Generate Image
    # Box check: Don't let it be wider than 90% of screen
    target_box_width = int(video_width * 0.9)
    
    hook_filename = f"temp_hook_{os.path.basename(video_path)}.png"
    # Ensure unique or temp location if needed, but relative is fine for this app structure
    
    try:
        img_path, box_w, box_h = create_hook_image(text, target_box_width, hook_filename, font_scale=font_scale)
        
        # 3. Calculate Overlay Position
        overlay_x = (video_width - box_w) // 2
        
        if position == "center":
            overlay_y = (video_height - box_h) // 2
        elif position == "bottom":
             # Bottom 20% mark (approx)
             overlay_y = int(video_height * 0.70)
        else:
             # Top 20% mark
             overlay_y = int(video_height * 0.20)
        
        # 4. FFmpeg Command
        print(f"🎬 Overlaying hook: '{text}' at {overlay_x},{overlay_y}")
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', img_path,
            '-filter_complex', f"[0:v][1:v]overlay={overlay_x}:{overlay_y}",
            '-c:a', 'copy',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
            output_path
        ]
        
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ Hook added to {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg Error: {e.stderr.decode() if e.stderr else 'Unknown'}")
        raise e
    except Exception as e:
        print(f"❌ Hook Gen Error: {e}")
        raise e
    finally:
        # Cleanup temp image
        if os.path.exists(hook_filename):
            os.remove(hook_filename)


def add_ranking_with_title_to_video(video_path, number, title, output_path, position="top-left", style="circle"):
    """
    Burns ranking number + title directly onto video using FFmpeg drawtext.
    Uses textfile= instead of text= to avoid shell/escaping issues.
    Style: large bold "TOP N" just above title, title text near bottom.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video {video_path} not found")

    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', video_path]
        res = subprocess.check_output(cmd).decode().strip()
        dims = res.split('\n')[0].split('x')
        video_width = int(dims[0])
        video_height = int(dims[1])
    except Exception:
        video_width = 1080
        video_height = 1920

    out_dir = os.path.dirname(os.path.abspath(output_path))

    # Write text to temp files to avoid ALL escaping issues
    top_txt_file = os.path.join(out_dir, f"temp_top_{number}.txt")
    title_txt_file = os.path.join(out_dir, f"temp_title_{number}.txt")
    with open(top_txt_file, 'w', encoding='utf-8') as f:
        f.write(f"TOP {number}")
    # Clean title: strip emojis, limit length
    clean_title = strip_emojis(title).strip()
    if not clean_title:
        clean_title = f"TOP {number}"
    with open(title_txt_file, 'w', encoding='utf-8') as f:
        f.write(clean_title)

    # Font sizes relative to video width
    top_fontsize = int(video_width * 0.12)
    title_fontsize = int(video_width * 0.045)

    # Use Liberation Sans Bold (installed via fonts-liberation in Docker)
    font_spec = "font=Liberation Sans Bold"

    # Position: title near bottom, TOP N just above title
    title_y = int(video_height * 0.82)
    top_y = title_y - top_fontsize - int(video_height * 0.015)

    # Escape file paths for FFmpeg filter syntax
    top_path_esc = top_txt_file.replace("\\", "/").replace(":", "\\:").replace("'", "'\\''")
    title_path_esc = title_txt_file.replace("\\", "/").replace(":", "\\:").replace("'", "'\\''")

    filter_parts = []

    # Layer 1: TOP N drop shadow
    filter_parts.append(
        f"drawtext=textfile='{top_path_esc}'"
        f":fontsize={top_fontsize}:fontcolor=black@0.5"
        f":borderw=0:x=(w-text_w)/2+4:y={top_y}+4"
        f":{font_spec}"
    )
    # Layer 2: TOP N main (white, thick black outline)
    filter_parts.append(
        f"drawtext=textfile='{top_path_esc}'"
        f":fontsize={top_fontsize}:fontcolor=white"
        f":borderw=7:bordercolor=black"
        f":x=(w-text_w)/2:y={top_y}"
        f":{font_spec}"
    )
    # Layer 3: Title drop shadow
    filter_parts.append(
        f"drawtext=textfile='{title_path_esc}'"
        f":fontsize={title_fontsize}:fontcolor=black@0.6"
        f":borderw=0:x=(w-text_w)/2+3:y={title_y}+3"
        f":{font_spec}"
    )
    # Layer 4: Title main (white, black outline)
    filter_parts.append(
        f"drawtext=textfile='{title_path_esc}'"
        f":fontsize={title_fontsize}:fontcolor=white"
        f":borderw=4:bordercolor=black"
        f":x=(w-text_w)/2:y={title_y}"
        f":{font_spec}"
    )

    filter_chain = ",".join(filter_parts)

    print(f"🏆 Overlaying TOP {number} + title: '{clean_title[:40]}'")

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', filter_chain,
        '-c:a', 'copy',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
        output_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ TOP {number} + title burned into {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg Error: {e.stderr.decode()[:300] if e.stderr else 'Unknown'}")
        raise e
    finally:
        for f in [top_txt_file, title_txt_file]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass


def add_ranking_number_to_video(video_path, number, output_path, position="top-left", style="circle"):
    """
    Overlays a ranking number on the video.
    position: 'top-left', 'top-right'
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video {video_path} not found")

    # Probe video dimensions
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', video_path]
        res = subprocess.check_output(cmd).decode().strip()
        dims = res.split('\n')[0].split('x')
        video_width = int(dims[0])
        video_height = int(dims[1])
    except Exception:
        video_width = 1080
        video_height = 1920

    # Use absolute path in same dir as output to avoid CWD issues in Docker
    safe_basename = os.path.basename(video_path).replace(' ', '_')
    rank_filename = os.path.join(
        os.path.dirname(os.path.abspath(output_path)),
        f"temp_rank_{number}_{safe_basename}.png"
    )

    try:
        img_path, img_w, img_h = create_ranking_number_image(
            number, video_width, rank_filename, style=style
        )

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Ranking PNG was not created: {img_path}")

        # Position: 5% margin from edges
        margin = int(video_width * 0.03)
        if position == "top-right":
            overlay_x = video_width - img_w - margin
        else:
            overlay_x = margin
        overlay_y = margin

        print(f"🏆 Overlaying rank #{number} at {overlay_x},{overlay_y}")

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', img_path,
            '-filter_complex', f"[0:v][1:v]overlay={overlay_x}:{overlay_y}",
            '-c:a', 'copy',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
            output_path
        ]

        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ Rank #{number} added to {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg Error: {e.stderr.decode() if e.stderr else 'Unknown'}")
        raise e
    finally:
        if os.path.exists(rank_filename):
            os.remove(rank_filename)


