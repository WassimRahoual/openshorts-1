"""
Reddit Ranking Pipeline.
Takes multiple downloaded Reddit video files and compiles them into a
ranked vertical short (TOP N → TOP 1) with scoreboard overlay.

Usage:
    python reddit_ranking.py \
        --clips clip1.mp4 clip2.mp4 clip3.mp4 ... \
        --titles "EPIC FALL" "POOL FAIL" "CAR CRASH" ... \
        --output output_dir/

Clips are ordered TOP N (first) → TOP 1 (last = best).
"""

import os
import sys
import json
import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import reframing from main pipeline
from main import process_video_to_vertical, process_short_video, fit_fontsize
from hooks import strip_emojis


def get_video_duration(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path]
        return float(subprocess.check_output(cmd).decode().strip())
    except Exception:
        return 0.0


def get_video_dimensions(path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', path]
        dims = subprocess.check_output(cmd).decode().strip().split('\n')[0].split('x')
        return int(dims[0]), int(dims[1])
    except Exception:
        return 1080, 1920


def reframe_clip(args):
    """Reframe a single clip to vertical. Runs in a thread."""
    idx, clip_path, output_path = args
    print(f"  ⏳ Reframing clip #{idx + 1}: {os.path.basename(clip_path)}")

    w, h = get_video_dimensions(clip_path)
    is_vertical = h > w

    if is_vertical:
        # Already vertical — just re-encode with watermark crop
        success = process_short_video(clip_path, output_path)
    else:
        # Horizontal → vertical with subject tracking / blur background
        success = process_video_to_vertical(clip_path, output_path, force_general=True)

    if success:
        dur = get_video_duration(output_path)
        print(f"  ✅ Clip #{idx + 1} reframed ({dur:.1f}s)")
        return (idx, output_path, dur)
    else:
        print(f"  ❌ Clip #{idx + 1} reframe failed")
        return None


def compile_ranking(clip_paths, titles, output_dir, output_name="reddit_ranking"):
    """
    Compile multiple video clips into a single ranking short.

    Args:
        clip_paths: List of video file paths, ordered TOP N → TOP 1
        titles: List of display titles for each clip
        output_dir: Directory for output files
        output_name: Base name for output file

    Returns:
        Path to the final ranking video, or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_clips = len(clip_paths)

    if n_clips < 2:
        print("❌ Need at least 2 clips for a ranking")
        return None

    print(f"\n🏆 Compiling Reddit Ranking: TOP {n_clips} → TOP 1")
    print(f"   Clips: {n_clips} | Output: {output_dir}")

    # Step 1: Reframe all clips to vertical (parallel)
    reframe_jobs = []
    for i, clip_path in enumerate(clip_paths):
        vert_path = os.path.join(output_dir, f"temp_vert_{i}.mp4")
        reframe_jobs.append((i, clip_path, vert_path))

    workers = min(3, n_clips)
    print(f"\n📐 Reframing {n_clips} clips to vertical ({workers} parallel workers)...")

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(reframe_clip, job): job for job in reframe_jobs}
        for future in as_completed(futures):
            r = future.result()
            if r:
                idx, path, dur = r
                results[idx] = (path, dur)

    # Collect in order
    segment_paths = []
    segment_durations = []
    segment_titles = []
    segment_ranks = []

    for i in range(n_clips):
        if i not in results:
            print(f"  ⚠️ Skipping clip #{i + 1} (reframe failed)")
            continue
        path, dur = results[i]
        rank = n_clips - i  # First clip = highest rank number (TOP N), last = TOP 1
        segment_paths.append(path)
        segment_durations.append(dur)
        segment_titles.append(titles[i] if i < len(titles) else f"FAIL #{rank}")
        segment_ranks.append(rank)

    if len(segment_paths) < 2:
        print("❌ Not enough clips survived reframing")
        return None

    # Step 2: Concatenate all segments
    print(f"\n🔗 Concatenating {len(segment_paths)} segments...")
    concat_raw = os.path.join(output_dir, f"{output_name}_raw.mp4")
    concat_list = os.path.join(output_dir, "temp_concat.txt")

    with open(concat_list, 'w') as f:
        for sp in segment_paths:
            f.write(f"file '{os.path.abspath(sp)}'\n")

    concat_cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', concat_list,
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
        '-c:a', 'aac', '-b:a', '192k', '-pix_fmt', 'yuv420p',
        concat_raw
    ]
    result = subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    os.remove(concat_list)

    # Clean temp segments
    for sp in segment_paths:
        if os.path.exists(sp):
            os.remove(sp)

    if result.returncode != 0:
        print(f"❌ Concat failed: {result.stderr.decode()[:300]}")
        return None

    # Step 3: Burn scoreboard overlay (same logic as main.py ranking)
    print(f"📊 Applying scoreboard overlay...")

    cumulative_time = 0.0
    seg_time_ranges = []
    for dur in segment_durations:
        seg_time_ranges.append((cumulative_time, cumulative_time + dur))
        cumulative_time += dur

    vw, vh = get_video_dimensions(concat_raw)
    font_spec = "font=Liberation Sans Bold"
    max_active_fontsize = int(vw * 0.046)
    max_past_fontsize = int(vw * 0.036)
    line_height = int(vh * 0.035)
    base_y = int(vh * 0.55)
    margin_x = int(vw * 0.05)
    max_text_width = vw - 2 * margin_x

    title_files = []
    for i, (rank, title) in enumerate(zip(segment_ranks, segment_titles)):
        txt_path = os.path.join(output_dir, f"temp_sb_{i}.txt")
        clean = strip_emojis(title).strip() or f'TOP {rank}'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"TOP {rank}  {clean}")
        title_files.append(txt_path)

    filter_parts = []
    total_segments = len(segment_ranks)

    # Read title texts for font size calculation
    title_texts = []
    for tf in title_files:
        with open(tf, 'r', encoding='utf-8') as fh:
            title_texts.append(fh.read().strip())

    for seg_idx in range(total_segments):
        seg_start, seg_end = seg_time_ranges[seg_idx]
        txt_esc = title_files[seg_idx].replace(":", "\\:").replace("'", "'\\''")
        line_y = base_y + seg_idx * line_height
        text = title_texts[seg_idx] if seg_idx < len(title_texts) else ''
        active_fontsize = fit_fontsize(text, max_active_fontsize, max_text_width)
        past_fontsize = fit_fontsize(text, max_past_fontsize, max_text_width)

        # Active state (bold, during current segment)
        filter_parts.append(
            f"drawtext=textfile='{txt_esc}':fontsize={active_fontsize}:fontcolor=black@0.5"
            f":x={margin_x}+2:y={line_y}+2:{font_spec}:enable='between(t,{seg_start:.3f},{seg_end:.3f})'"
        )
        filter_parts.append(
            f"drawtext=textfile='{txt_esc}':fontsize={active_fontsize}:fontcolor=white"
            f":borderw=4:bordercolor=black:x={margin_x}:y={line_y}:{font_spec}:enable='between(t,{seg_start:.3f},{seg_end:.3f})'"
        )
        # Past state (dimmed, after segment)
        if seg_idx < total_segments - 1:
            next_start = seg_time_ranges[seg_idx + 1][0]
            filter_parts.append(
                f"drawtext=textfile='{txt_esc}':fontsize={past_fontsize}:fontcolor=black@0.3"
                f":x={margin_x}+1:y={line_y}+1:{font_spec}:enable='gte(t,{next_start:.3f})'"
            )
            filter_parts.append(
                f"drawtext=textfile='{txt_esc}':fontsize={past_fontsize}:fontcolor=white@0.7"
                f":borderw=3:bordercolor=black@0.5:x={margin_x}:y={line_y}:{font_spec}:enable='gte(t,{next_start:.3f})'"
            )

    filter_chain = ",".join(filter_parts)
    final_output = os.path.join(output_dir, f"{output_name}.mp4")

    overlay_cmd = [
        'ffmpeg', '-y', '-i', concat_raw,
        '-vf', filter_chain,
        '-c:a', 'copy', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
        final_output
    ]

    overlay_result = subprocess.run(overlay_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Cleanup
    if os.path.exists(concat_raw):
        os.remove(concat_raw)
    for tf in title_files:
        if os.path.exists(tf):
            os.remove(tf)

    if overlay_result.returncode != 0:
        print(f"❌ Scoreboard overlay failed: {overlay_result.stderr.decode()[:300]}")
        # Fall back to raw concat if overlay fails
        if os.path.exists(concat_raw):
            os.rename(concat_raw, final_output)
        else:
            return None

    total_dur = get_video_duration(final_output)
    print(f"\n🏆 Ranking Short ready! ({total_dur:.1f}s): {final_output}")

    # Write metadata
    metadata = {
        "type": "reddit_ranking",
        "shorts": [{
            "video_title_for_youtube_short": f"TOP {len(segment_ranks)} Fails That Will Make You Cry Laughing",
            "video_description_for_tiktok": "Wait for number 1... 💀",
            "video_description_for_instagram": "Which one is the worst? Comment below! 👇",
            "viral_hook_text": f"TOP {len(segment_ranks)} BIGGEST FAILS",
            "hashtags_tiktok": "#fail #fails #funny #top5 #ranking #fyp #viral",
            "hashtags_instagram": "#fail #fails #funny #ranking #viral #comedy #epic",
        }],
        "clips": [
            {"rank": r, "title": t, "duration": d}
            for r, t, d in zip(segment_ranks, segment_titles, segment_durations)
        ],
        "total_duration": total_dur,
    }
    meta_path = os.path.join(output_dir, f"{output_name}_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Metadata: {meta_path}")

    return final_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reddit Ranking Compiler")
    parser.add_argument('--clips', nargs='+', required=True, help="Video files ordered TOP N → TOP 1")
    parser.add_argument('--titles', nargs='+', default=[], help="Display titles for each clip")
    parser.add_argument('--output', required=True, help="Output directory")
    parser.add_argument('--name', default="reddit_ranking", help="Output file base name")

    args = parser.parse_args()

    start = time.time()
    result = compile_ranking(args.clips, args.titles, args.output, args.name)
    elapsed = time.time() - start

    if result:
        print(f"\n✅ Done in {elapsed:.1f}s: {result}")
    else:
        print(f"\n❌ Failed after {elapsed:.1f}s")
        sys.exit(1)
