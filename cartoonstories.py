"""
CartoonStories: AI-powered animated cartoon story generator.

Generates short animated stories (30-90s, 9:16) with:
  1. Story generation & scene breakdown (Gemini)
  2. Cartoon image generation per scene (Flux 2 Pro)
  3. Image-to-video animation per scene (Hailuo / Kling)
  4. Narration voiceover (ElevenLabs TTS)
  5. TikTok-style subtitles (Whisper)
  6. Final assembly with music (FFmpeg)
"""

import os
import re
import json
import time
import random
import subprocess
import httpx
from typing import Optional, List, Dict, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from saasshorts import (
    _fal_run,
    _fal_upload_file,
    _get_media_duration,
    generate_voiceover,
    generate_tiktok_subs,
    ELEVENLABS_API_BASE,
    FAL_QUEUE_BASE,
)

GEMINI_MODEL = "gemini-3-flash-preview"

# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Story Generation (Gemini)
# ═══════════════════════════════════════════════════════════════════════

STORY_PROMPT_TEMPLATE = """You are a master storyteller who creates viral animated short stories for TikTok and Instagram Reels.

Generate a SHORT animated story (30-60 seconds when narrated) that is {genre} in tone.

LANGUAGE: {language}
THEME/TOPIC: {topic}
NUMBER OF SCENES: {num_scenes}

RULES:
- The story MUST be emotionally engaging — it should make viewers feel something (surprise, fear, joy, sadness, wonder).
- Start with a HOOK that grabs attention in the first 2 seconds.
- Each scene MUST have a distinct, vivid visual that can be illustrated as a single cartoon image.
- The narration should be conversational and natural, as if telling a bedtime story.
- Keep the total narration under 150 words for a ~45 second video.
- ALL narration MUST be in {language}.
- End with a twist, moral, or emotional payoff.

IMAGE PROMPT RULES (CRITICAL FOR VISUAL CONSISTENCY):
- Every image_prompt MUST start with the EXACT style prefix: "{style_prefix}"
- Describe the SAME characters with IDENTICAL physical descriptions across ALL scenes (e.g. "a small purple dragon with big green eyes" — use this EXACT description every time that character appears).
- Each image_prompt should describe a SINGLE clear moment/composition.
- Include lighting, mood, camera angle.
- NO text, NO watermarks, NO UI elements in the image descriptions.

MOTION PROMPT RULES:
- Describe subtle animation: character movement, environmental effects (wind, rain, light changes).
- Keep motion descriptions SHORT (1 sentence).
- Focus on the primary action in the scene.

OUTPUT — RETURN ONLY VALID JSON (no markdown, no comments):
{{
  "title": "<catchy story title in {language}>",
  "style_prefix": "{style_prefix}",
  "language": "{language}",
  "character_descriptions": {{
    "<character_name>": "<EXACT physical description to reuse in every scene>"
  }},
  "scenes": [
    {{
      "scene_number": 1,
      "narration": "<narration text for this scene in {language}>",
      "image_prompt": "<MUST start with style prefix. Full image description in English>",
      "motion_prompt": "<short animation description in English>",
      "emotion": "<dominant emotion: happy, sad, scared, surprised, angry, peaceful, tense>"
    }}
  ],
  "full_narration": "<all narration concatenated with natural pauses, in {language}>",
  "hook_text": "<2-5 word hook overlay for the first scene, in {language}>"
}}
"""

STYLE_PRESETS = {
    "pixar_3d": "3D Pixar-style cartoon, vibrant colors, soft lighting, cinematic composition, high quality render,",
    "anime": "Japanese anime style, detailed character design, vibrant colors, dynamic composition, Studio Ghibli inspired,",
    "watercolor": "Soft watercolor illustration style, pastel colors, dreamy atmosphere, storybook aesthetic,",
    "comic": "Bold comic book style, strong outlines, vivid colors, dramatic angles, graphic novel aesthetic,",
    "chibi": "Cute chibi cartoon style, big eyes, small body, adorable expressions, kawaii aesthetic, pastel background,",
    "dark_fantasy": "Dark fantasy illustration, moody lighting, rich textures, atmospheric, gothic fairytale style,",
    "flat_modern": "Modern flat illustration style, clean lines, bold colors, minimalist background, digital art,",
}

GENRE_PRESETS = {
    "fairy_tale": "a magical fairy tale / conte de fées",
    "horror": "a creepy horror story with suspense and a dark twist",
    "comedy": "a funny comedy with unexpected humor",
    "moral": "an inspiring story with a life lesson / moral",
    "mystery": "a mysterious story with suspense and a surprising reveal",
    "romance": "a sweet romantic story",
    "adventure": "an exciting adventure story",
    "scary_kids": "a mildly scary story for kids (not too dark, with a happy ending)",
}


def generate_story(
    topic: str,
    gemini_key: str,
    genre: str = "fairy_tale",
    style: str = "pixar_3d",
    language: str = "fr",
    num_scenes: int = 6,
) -> dict:
    """Generate an animated story script using Gemini."""
    from google import genai

    print(f"[CartoonStories] Generating story ({genre}, {style}, {num_scenes} scenes)...")

    client = genai.Client(api_key=gemini_key)
    style_prefix = STYLE_PRESETS.get(style, STYLE_PRESETS["pixar_3d"])
    genre_desc = GENRE_PRESETS.get(genre, genre)

    prompt = STORY_PROMPT_TEMPLATE.format(
        genre=genre_desc,
        language=language,
        topic=topic,
        num_scenes=num_scenes,
        style_prefix=style_prefix,
    )

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    text = response.text
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    result = json.loads(text)

    # Validate
    scenes = result.get("scenes", [])
    if not scenes:
        raise Exception("No scenes generated")

    print(f"[CartoonStories] Generated story: '{result.get('title')}' with {len(scenes)} scenes")
    return result


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Scene Image Generation (Flux 2 Pro)
# ═══════════════════════════════════════════════════════════════════════

def generate_scene_image(
    image_prompt: str,
    fal_key: str,
    output_path: str,
) -> str:
    """Generate a single scene image using Flux 2 Pro."""
    result = _fal_run(
        "fal-ai/flux-2-pro",
        {
            "prompt": image_prompt,
            "image_size": "portrait_4_3",
            "safety_tolerance": 5,
            "seed": random.randint(0, 999999),
        },
        fal_key,
        timeout=300,
    )

    images = result.get("images") or result.get("output", [])
    if not images:
        raise Exception(f"No images in scene result: {list(result.keys())}")
    img_url = images[0]["url"] if isinstance(images[0], dict) else images[0]

    with httpx.Client(timeout=60.0) as client:
        img_resp = client.get(img_url)
        with open(output_path, "wb") as f:
            f.write(img_resp.content)

    print(f"[CartoonStories] Scene image: {output_path}")
    return output_path


def generate_all_scene_images(
    scenes: List[dict],
    fal_key: str,
    output_dir: str,
    slug: str,
    log: Callable = print,
) -> List[str]:
    """Generate all scene images in parallel."""
    log(f"[2/6] Generating {len(scenes)} scene images (Flux 2 Pro, parallel)...")

    paths = [None] * len(scenes)

    def _gen(i, scene):
        path = os.path.join(output_dir, f"{slug}_scene_{i+1}.png")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            log(f"  Scene {i+1} image cached, skipping.")
            return i, path
        generate_scene_image(scene["image_prompt"], fal_key, path)
        log(f"  Scene {i+1}/{len(scenes)} image ready.")
        return i, path

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_gen, i, s) for i, s in enumerate(scenes)]
        for future in as_completed(futures):
            idx, path = future.result()
            paths[idx] = path

    return paths


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Image-to-Video Animation (Hailuo / Kling)
# ═══════════════════════════════════════════════════════════════════════

def animate_scene(
    image_path: str,
    motion_prompt: str,
    fal_key: str,
    output_path: str,
    mode: str = "lowcost",
) -> str:
    """Animate a scene image into a short video clip."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"[CartoonStories] Animation cached: {output_path}")
        return output_path

    image_url = _fal_upload_file(image_path, fal_key)

    prompt = (
        f"{motion_prompt}. "
        "Smooth cinematic animation, subtle camera movement, "
        "high quality cartoon animation, consistent art style."
    )

    if mode == "premium":
        # Kling v2 Standard — higher quality, ~$0.35/5s
        result = _fal_run(
            "fal-ai/kling-video/v2/standard/image-to-video",
            {
                "image_url": image_url,
                "prompt": prompt,
                "duration": "5",
                "aspect_ratio": "9:16",
            },
            fal_key,
            timeout=600,
        )
    else:
        # Hailuo 2.3 Fast — cheaper, ~$0.19/5s
        result = _fal_run(
            "fal-ai/minimax/hailuo-2.3-fast/standard/image-to-video",
            {
                "image_url": image_url,
                "prompt": prompt,
            },
            fal_key,
            timeout=300,
        )

    # Extract video URL
    if "video" in result:
        video_url = result["video"]["url"] if isinstance(result["video"], dict) else result["video"]
    elif "video_url" in result:
        video_url = result["video_url"]
    elif "output" in result:
        video_url = result["output"]["url"] if isinstance(result["output"], dict) else result["output"]
    else:
        raise Exception(f"No video in animation result: {result}")

    with httpx.Client(timeout=180.0) as client:
        vid_resp = client.get(video_url)
        with open(output_path, "wb") as f:
            f.write(vid_resp.content)

    print(f"[CartoonStories] Animation: {output_path}")
    return output_path


def animate_all_scenes(
    image_paths: List[str],
    scenes: List[dict],
    fal_key: str,
    output_dir: str,
    slug: str,
    mode: str = "lowcost",
    log: Callable = print,
) -> List[str]:
    """Animate all scene images into video clips. Sequential to avoid API limits."""
    log(f"[3/6] Animating {len(scenes)} scenes ({'Kling v2' if mode == 'premium' else 'Hailuo 2.3'})...")

    paths = []
    for i, (img_path, scene) in enumerate(zip(image_paths, scenes)):
        out_path = os.path.join(output_dir, f"{slug}_anim_{i+1}.mp4")
        log(f"  Animating scene {i+1}/{len(scenes)}... (this takes 1-3 min)")
        animate_scene(img_path, scene.get("motion_prompt", ""), fal_key, out_path, mode)
        log(f"  Scene {i+1}/{len(scenes)} animated.")
        paths.append(out_path)

    return paths


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Assembly (FFmpeg)
# ═══════════════════════════════════════════════════════════════════════

def assemble_cartoon_video(
    animation_paths: List[str],
    audio_path: str,
    subs_path: str,
    output_path: str,
    log: Callable = print,
) -> str:
    """Assemble animated scenes + narration + subtitles into final video."""
    log("[6/6] Assembling final video with FFmpeg...")

    audio_duration = _get_media_duration(audio_path)
    n_clips = len(animation_paths)

    # Get each clip's natural duration
    clip_durations = [_get_media_duration(p) for p in animation_paths]

    # Target duration per scene based on audio
    target_per_scene = audio_duration / n_clips

    # Build concat list with speed adjustment per clip
    norm = "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1"

    filter_parts = []
    concat_inputs = []

    inputs_cmd = []
    for i, path in enumerate(animation_paths):
        inputs_cmd.extend(["-i", path])

        clip_dur = clip_durations[i]
        # Calculate speed factor to match target duration
        if clip_dur > 0 and target_per_scene > 0:
            speed = clip_dur / target_per_scene
            # Clamp speed to reasonable range (0.5x to 2.0x)
            speed = max(0.5, min(2.0, speed))
        else:
            speed = 1.0

        # Video: setpts to adjust speed, then normalize
        filter_parts.append(
            f"[{i}:v]setpts=PTS/{speed:.3f},{norm}[v{i}]"
        )
        concat_inputs.append(f"[v{i}]")

    # Concat all video streams
    concat_filter = f"{''.join(concat_inputs)}concat=n={n_clips}:v=1:a=0[outv]"

    # Add subtitles
    safe_sub = subs_path.replace("\\", "/").replace(":", "\\:")
    if subs_path.endswith(".ass"):
        sub_filter = f"[outv]ass='{safe_sub}'[finalv]"
    else:
        sub_style = (
            "Alignment=2,Fontname=Arial Black,Fontsize=24,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,BorderStyle=1,Outline=4,Shadow=0,MarginV=120,Bold=-1"
        )
        sub_filter = f"[outv]subtitles='{safe_sub}':force_style='{sub_style}'[finalv]"

    full_filter = ";".join(filter_parts) + ";" + concat_filter + ";" + sub_filter

    cmd = [
        "ffmpeg", "-y",
        *inputs_cmd,
        "-i", audio_path,
        "-filter_complex", full_filter,
        "-map", "[finalv]",
        "-map", f"{n_clips}:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode()[:500]
        print(f"[CartoonStories] FFmpeg error: {stderr}")
        raise Exception(f"FFmpeg assembly failed: {stderr}")

    log(f"[CartoonStories] Final video: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════

def generate_cartoon_video(
    story: dict,
    config: dict,
    output_dir: str,
    log: Callable[[str], None] = print,
) -> dict:
    """
    Full cartoon story video generation pipeline.

    Args:
        story: Story object from generate_story()
        config: {
            "fal_key": str,
            "elevenlabs_key": str,
            "voice_id": str,
            "video_mode": "lowcost" | "premium",
        }
        output_dir: Directory to write output files
        log: Callback for progress logging

    Returns:
        {"video_path": str, "duration": float, "cost_estimate": dict}
    """
    os.makedirs(output_dir, exist_ok=True)

    fal_key = config["fal_key"]
    elevenlabs_key = config["elevenlabs_key"]
    voice_id = config.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
    video_mode = config.get("video_mode", "lowcost")

    title = story.get("title", "cartoon")
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower())[:30]
    scenes = story.get("scenes", [])
    full_narration = story.get("full_narration", "")

    if not full_narration:
        full_narration = " ".join(s.get("narration", "") for s in scenes)

    # Paths
    audio_path = os.path.join(output_dir, f"{slug}_voice.mp3")
    subs_path = os.path.join(output_dir, f"{slug}_subs.ass")
    final_path = os.path.join(output_dir, f"{slug}_final.mp4")

    def _exists(path):
        return os.path.exists(path) and os.path.getsize(path) > 0

    # ── Step 1: Generate voiceover ──
    if not _exists(audio_path):
        log("[1/6] Generating narration voiceover (ElevenLabs)...")
        generate_voiceover(full_narration, elevenlabs_key, audio_path, voice_id)
    else:
        log("[1/6] Voiceover cached, skipping.")

    # ── Step 2: Generate scene images (parallel) ──
    image_paths = generate_all_scene_images(scenes, fal_key, output_dir, slug, log)

    # ── Step 3: Animate scenes (sequential — API rate limits) ──
    animation_paths = animate_all_scenes(
        image_paths, scenes, fal_key, output_dir, slug, video_mode, log
    )

    # ── Step 4: Generate subtitles ──
    log("[4/6] Generating TikTok-style subtitles...")
    generate_tiktok_subs(audio_path, subs_path, max_words=3)

    # ── Step 5: Skip (reserved) ──
    log("[5/6] Preparing assembly...")

    # ── Step 6: Assemble final video ──
    assemble_cartoon_video(animation_paths, audio_path, subs_path, final_path, log)

    log("Video generation complete!")

    # Cost estimate
    n = len(scenes)
    narration_len = len(full_narration)
    if video_mode == "lowcost":
        cost = {
            "scene_images_flux": round(n * 0.05, 2),
            "animations_hailuo": round(n * 0.19, 2),
            "voiceover_elevenlabs": round(narration_len * 0.00003, 3),
            "ffmpeg_assembly": 0.00,
        }
    else:
        cost = {
            "scene_images_flux": round(n * 0.05, 2),
            "animations_kling": round(n * 0.35, 2),
            "voiceover_elevenlabs": round(narration_len * 0.00003, 3),
            "ffmpeg_assembly": 0.00,
        }
    cost["total"] = round(sum(cost.values()), 2)

    duration = _get_media_duration(final_path)

    return {
        "video_path": final_path,
        "video_filename": os.path.basename(final_path),
        "duration": duration,
        "cost_estimate": cost,
        "story": story,
    }
