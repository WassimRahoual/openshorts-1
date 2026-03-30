import os
import subprocess
import tempfile
from hooks import add_ranking_number_to_video


def _probe_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.check_output(cmd).decode().strip()
    return float(result)


def _probe_has_audio(video_path):
    """Check if video has an audio stream."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a',
        '-show_entries', 'stream=codec_type',
        '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.check_output(cmd).decode().strip()
    return len(result) > 0


def compile_clips(
    clip_paths,
    output_path,
    ranking_numbers=None,
    ranking_style="circle",
    ranking_position="top-left",
    transition_type="fade",
    transition_duration=0.5,
    bg_music_path=None,
    bg_music_volume=0.15,
):
    """
    Combines multiple clips into a single compilation video.

    Args:
        clip_paths: List of video file paths in order
        output_path: Output file path
        ranking_numbers: List of ranking numbers (e.g. [5, 4, 3, 2, 1]) or None
        ranking_style: 'circle', 'bold', or 'none'
        ranking_position: 'top-left' or 'top-right'
        transition_type: 'fade', 'wipeleft', 'wiperight', 'slidedown', 'slideup', 'none'
        transition_duration: Duration of transitions in seconds
        bg_music_path: Optional path to background music file
        bg_music_volume: Volume level for background music (0.0 to 1.0)

    Returns:
        Output file path
    """
    if not clip_paths:
        raise ValueError("No clips provided")

    if len(clip_paths) == 1 and not ranking_numbers and not bg_music_path:
        # Single clip, nothing to compile
        import shutil
        shutil.copy2(clip_paths[0], output_path)
        return output_path

    temp_files = []

    try:
        # Stage 1: Add ranking numbers if requested
        working_clips = []
        if ranking_numbers and ranking_style != "none":
            for i, (clip_path, rank_num) in enumerate(zip(clip_paths, ranking_numbers)):
                ranked_path = os.path.join(
                    os.path.dirname(output_path),
                    f"temp_ranked_{i}_{os.path.basename(clip_path)}"
                )
                temp_files.append(ranked_path)
                add_ranking_number_to_video(
                    clip_path, rank_num, ranked_path,
                    position=ranking_position, style=ranking_style
                )
                working_clips.append(ranked_path)
        else:
            working_clips = list(clip_paths)

        # Stage 2: Probe durations
        durations = [_probe_duration(clip) for clip in working_clips]
        print(f"📏 Clip durations: {[f'{d:.1f}s' for d in durations]}")

        # Stage 3: Concatenate with transitions
        if transition_type == "none" or len(working_clips) == 1:
            # Simple concat without transitions
            compiled_video = _concat_simple(working_clips, output_path if not bg_music_path else None, temp_files)
            if bg_music_path:
                compiled_video = _concat_simple(working_clips, None, temp_files)
        else:
            compiled_video = _concat_xfade(
                working_clips, durations, transition_type, transition_duration,
                output_path if not bg_music_path else None, temp_files
            )

        # Stage 4: Mix background music if provided
        if bg_music_path:
            if compiled_video is None:
                # Simple concat was used, need temp file
                compiled_video = os.path.join(
                    os.path.dirname(output_path), f"temp_compiled_{os.path.basename(output_path)}"
                )
                temp_files.append(compiled_video)
                _concat_simple(working_clips, compiled_video, temp_files)

            _mix_background_music(compiled_video, bg_music_path, bg_music_volume, output_path)
        elif compiled_video and compiled_video != output_path:
            import shutil
            shutil.move(compiled_video, output_path)

        print(f"✅ Compilation complete: {output_path}")
        return output_path

    finally:
        # Cleanup temp files
        for f in temp_files:
            if os.path.exists(f) and f != output_path:
                try:
                    os.remove(f)
                except OSError:
                    pass


def _concat_simple(clip_paths, output_path, temp_files):
    """Concatenate clips without transitions using concat demuxer."""
    # Write concat file
    concat_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.txt', delete=False, dir=os.path.dirname(output_path or clip_paths[0])
    )
    temp_files.append(concat_file.name)

    for clip in clip_paths:
        concat_file.write(f"file '{os.path.abspath(clip)}'\n")
    concat_file.close()

    if output_path is None:
        output_path = concat_file.name.replace('.txt', '_concat.mp4')
        temp_files.append(output_path)

    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0',
        '-i', concat_file.name,
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k',
        output_path
    ]

    print(f"🔗 Concatenating {len(clip_paths)} clips (no transitions)...")
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path


def _concat_xfade(clip_paths, durations, transition_type, transition_duration, output_path, temp_files):
    """Concatenate clips with xfade transitions."""
    n = len(clip_paths)

    # Build input args
    input_args = []
    for clip in clip_paths:
        input_args.extend(['-i', clip])

    # Build xfade filter chain
    video_filters = []
    audio_filters = []

    # Calculate offsets: each transition starts at (cumulative_duration - transition_duration)
    cumulative = 0.0

    for i in range(n - 1):
        # Video xfade
        if i == 0:
            vin1 = "[0:v]"
        else:
            vin1 = f"[v{i - 1}]"
        vin2 = f"[{i + 1}:v]"

        offset = cumulative + durations[i] - transition_duration
        if i < n - 2:
            vout = f"[v{i}]"
        else:
            vout = "[vout]"

        video_filters.append(
            f"{vin1}{vin2}xfade=transition={transition_type}:duration={transition_duration}:offset={offset:.3f}{vout}"
        )

        # Audio crossfade
        if i == 0:
            ain1 = "[0:a]"
        else:
            ain1 = f"[a{i - 1}]"
        ain2 = f"[{i + 1}:a]"

        if i < n - 2:
            aout = f"[a{i}]"
        else:
            aout = "[aout]"

        audio_filters.append(
            f"{ain1}{ain2}acrossfade=d={transition_duration}{aout}"
        )

        # After transition, effective duration contribution is reduced
        cumulative += durations[i] - transition_duration

    filter_complex = ";\n".join(video_filters + audio_filters)

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(clip_paths[0]),
            f"temp_xfade_compilation.mp4"
        )
        temp_files.append(output_path)

    cmd = [
        'ffmpeg', '-y',
        *input_args,
        '-filter_complex', filter_complex,
        '-map', '[vout]', '-map', '[aout]',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k',
        output_path
    ]

    print(f"🎬 Compiling {n} clips with {transition_type} transitions ({transition_duration}s)...")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ xfade failed, falling back to simple concat: {e.stderr.decode()[:200] if e.stderr else ''}")
        return _concat_simple(clip_paths, output_path, temp_files)

    return output_path


def _mix_background_music(video_path, music_path, volume, output_path):
    """Mix background music under the video's audio."""
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', music_path,
        '-filter_complex',
        f"[0:a]volume=1.0[main];"
        f"[1:a]aloop=loop=-1:size=2e9,volume={volume}[bgm];"
        f"[main][bgm]amix=inputs=2:duration=first[afinal]",
        '-map', '0:v', '-map', '[afinal]',
        '-c:v', 'copy',
        '-c:a', 'aac', '-b:a', '192k',
        output_path
    ]

    print(f"🎵 Mixing background music (volume: {volume})...")
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_path
