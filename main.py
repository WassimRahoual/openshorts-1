import time
import cv2
import scenedetect
import subprocess
import argparse
import re
import sys
from scenedetect import VideoManager, SceneManager, open_video
from scenedetect.detectors import ContentDetector
from ultralytics import YOLO
import torch
import os
import numpy as np
from tqdm import tqdm
import yt_dlp
import mediapipe as mp
# import whisper (replaced by faster_whisper inside function)
from google import genai
from dotenv import load_dotenv
import json

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Load environment variables
load_dotenv()

# --- Constants ---
ASPECT_RATIO = 9 / 16

GEMINI_PROMPT_TEMPLATE = """
You are a senior short-form video editor. Read the ENTIRE transcript and word-level timestamps to choose the 3–15 MOST VIRAL moments for TikTok/IG Reels/YouTube Shorts. Each clip must be between 15 and 60 seconds long.

⚠️ FFMPEG TIME CONTRACT — STRICT REQUIREMENTS:
- Return timestamps in ABSOLUTE SECONDS from the start of the video (usable in: ffmpeg -ss <start> -to <end> -i <input> ...).
- Only NUMBERS with decimal point, up to 3 decimals (examples: 0, 1.250, 17.350).
- Ensure 0 ≤ start < end ≤ VIDEO_DURATION_SECONDS.
- Each clip between 15 and 60 s (inclusive).
- Prefer starting 0.2–0.4 s BEFORE the hook and ending 0.2–0.4 s AFTER the payoff.
- Use silence moments for natural cuts; never cut in the middle of a word or phrase.
- STRICTLY FORBIDDEN to use time formats other than absolute seconds.

VIDEO_DURATION_SECONDS: {video_duration}

TRANSCRIPT_TEXT (raw):
{transcript_text}

WORDS_JSON (array of {{w, s, e}} where s/e are seconds):
{words_json}

STRICT EXCLUSIONS:
- No generic intros/outros or purely sponsorship segments unless they contain the hook.
- No transition scenes (animated transitions, logo bumpers, "subscribe" screens, countdowns, black/white frames between segments).
- No end screens, credit rolls, or "thanks for watching" segments.
- Only select moments with REAL CONTENT (someone talking, demonstrating, reacting, or performing).
- No clips < 15 s or > 60 s.

OUTPUT — RETURN ONLY VALID JSON (no markdown, no comments). Order clips by predicted performance (best to worst). In the descriptions, ALWAYS include a CTA like "Follow me and comment X and I'll send you the workflow" (especially if discussing an n8n workflow):
{{
  "shorts": [
    {{
      "start": <number in seconds, e.g., 12.340>,
      "end": <number in seconds, e.g., 37.900>,
      "video_description_for_tiktok": "<description for TikTok oriented to get views>",
      "video_description_for_instagram": "<description for Instagram oriented to get views>",
      "video_title_for_youtube_short": "<title for YouTube Short oriented to get views 100 chars max>",
      "viral_hook_text": "<SHORT punchy text overlay (max 10 words). MUST BE IN THE SAME LANGUAGE AS THE VIDEO TRANSCRIPT. Examples: 'POV: You realized...', 'Did you know?', 'Stop doing this!'>",
      "hashtags_tiktok": "<8-12 hashtags for TikTok separated by spaces. Mix of: 2-3 high-volume trending hashtags (e.g. #fyp #viral #pourtoi), 3-4 niche topic hashtags, 2-3 mid-volume discovery hashtags. MUST start with #. MUST BE IN THE SAME LANGUAGE AS THE VIDEO (except universal ones like #fyp). Example: #fyp #viral #cybersecurity #hacking #telegram #darkweb #pourtoi #tech>",
      "hashtags_instagram": "<8-12 hashtags for Instagram Reels separated by spaces. Mix of: 2-3 high-volume hashtags (e.g. #reels #explore #trending), 3-4 niche topic hashtags, 2-3 community hashtags. MUST start with #. MUST BE IN THE SAME LANGUAGE AS THE VIDEO (except universal ones like #reels). Example: #reels #explore #cybersecurite #hacking #telegram #darkweb #tech #infosec>"
    }}
  ]
}}
"""

GEMINI_RANKING_PROMPT_TEMPLATE = """
WATCH THE VIDEO CAREFULLY. You are selecting the {num_clips} FUNNIEST clips from this fail compilation for a "TOP {num_clips}" ranking Short.

You can SEE the video. Use your EYES to find the funniest moments — people falling, crashing, animals doing stupid things, unexpected fails. DO NOT rely on the transcript (it's mostly noise/reactions).

WHAT TO PICK:
- Moments where something VISUALLY FUNNY happens: falls, crashes, fails, surprises, animals being chaotic
- Moments with a clear BUILDUP then PAYOFF (setup → fail)
- Pick clips that would make someone LAUGH OUT LOUD
- STRONGLY PREFER longer scenes (6-15 seconds) that show the full setup before the fail
- If a scene is short (under 5s), only pick it if it's EXTREMELY funny
- SKIP: boring/dark/static footage, talking heads, intros/outros, mildly amusing moments
- SKIP: transition scenes (animated transitions, logo bumpers, "subscribe" screens, countdowns, black/white frames between segments), end screens, credit rolls, "thanks for watching" segments

VIDEO_DURATION_SECONDS: {video_duration}

===== SCENE LIST =====
{scene_list}
=====

RULES:
- Select scenes by SCENE ID. Do NOT invent timestamps.
- Each TOP = ONE scene only.
- ⚠️ EVERY scene_id MUST BE UNIQUE — NEVER select the same scene twice. Each clip must come from a DIFFERENT scene. If you run out of good scenes, return fewer clips rather than duplicating.
- trim_start = scene start time, trim_end = scene end time. USE THE FULL SCENE. Do NOT trim shorter — every second of buildup matters.
- PREFER longer scenes. Short scenes (under 5s) usually lack buildup and are less funny.
- Position {num_clips} = least impressive. Position 1 = BEST clip.

TITLE RULES (VERY IMPORTANT):
- ranking_title describes the PHYSICAL ACTION you would SEE in the clip.
- 3-4 words, ALL CAPS, no emoji.
- GOOD examples: "FALLS OFF SKATEBOARD", "CAR HITS WALL", "DOG STEALS FOOD", "SLIPS ON ICE"
- BAD examples (NEVER USE): anything from the transcript/audio, names, song lyrics, "JENNY JENNY", "ENDLESS LOOP", "MANTRA BEGINS", reactions like "OH MY GOD"
- If you cannot determine the visual action, use a GENERIC FAIL description like "EPIC FAIL MOMENT"

OUTPUT — RETURN ONLY VALID JSON (no markdown, no comments):
{{
  "video_title_for_youtube_short": "<YouTube title 100 chars max>",
  "video_description_for_tiktok": "<TikTok description with CTA>",
  "video_description_for_instagram": "<Instagram description with CTA>",
  "hashtags_tiktok": "<8-12 hashtags>",
  "hashtags_instagram": "<8-12 hashtags>",
  "shorts": [
    {{
      "rank": {num_clips},
      "scene_id": <integer scene ID>,
      "trim_start": <number>,
      "trim_end": <number>,
      "ranking_title": "<PHYSICAL ACTION, 3-4 words, ALL CAPS>"
    }},
    ... (continue for each rank down to rank 1)
  ]
}}
"""

# Fallback prompt when no scene detection is available (e.g. very short videos)
GEMINI_RANKING_PROMPT_TEMPLATE_NO_SCENES = """
You are selecting the {num_clips} best clips from a FAIL COMPILATION for a "TOP {num_clips}" ranking Short.

⚠️ This is a COMPILATION of SEPARATE short clips. The transcript is mostly reactions and background noise — DO NOT choose moments based on what is SAID. Choose based on where PHYSICAL ACTIONS happen (falls, crashes, fails, surprises).

SKIP THESE — they are NOT valid clips:
- Intro/outro sequences, transition scenes, animated logos, "subscribe" screens, countdowns
- Black/white frames between segments, end screens, credit rolls
- Only select moments with REAL VISUAL ACTION happening.

TIMESTAMP RULES:
- Absolute seconds, up to 3 decimals. 0 ≤ start < end ≤ VIDEO_DURATION_SECONDS.
- Each clip: 5-10 seconds.
- NON-OVERLAPPING clips covering distinct moments. NEVER select overlapping or duplicate time ranges — each clip must be from a DIFFERENT moment.

VIDEO_DURATION_SECONDS: {video_duration}

TRANSCRIPT_TEXT (raw):
{transcript_text}

WORDS_JSON (array of {{w, s, e}} where s/e are seconds):
{words_json}

RANKING:
- Position {num_clips} = least impressive. Position 1 = BEST clip.

TITLE RULES:
- Describe the PHYSICAL ACTION visible in the clip. 3-4 words, ALL CAPS, no emoji.
- GOOD: "FALLS OFF SKATEBOARD", "CAR HITS WALL", "SLIPS ON ICE"
- BAD (NEVER USE): transcript words, names, lyrics, reactions, "ENDLESS LOOP", "JENNY JENNY"

OUTPUT — RETURN ONLY VALID JSON:
{{
  "video_title_for_youtube_short": "<YouTube title 100 chars max>",
  "video_description_for_tiktok": "<TikTok description with CTA>",
  "video_description_for_instagram": "<Instagram description with CTA>",
  "hashtags_tiktok": "<8-12 hashtags>",
  "hashtags_instagram": "<8-12 hashtags>",
  "shorts": [
    {{
      "rank": {num_clips},
      "start": <number>,
      "end": <number>,
      "ranking_title": "<PHYSICAL ACTION, 3-4 words, ALL CAPS>"
    }},
    ... (continue for each rank down to rank 1)
  ]
}}
"""

GEMINI_SHORT_PROMPT_TEMPLATE = """
You are a senior short-form video editor. Read the transcript of this SHORT video (already a clip, under 90 seconds) and generate optimized metadata for reposting on TikTok, Instagram Reels, and YouTube Shorts.

VIDEO_DURATION_SECONDS: {video_duration}

TRANSCRIPT_TEXT (raw):
{transcript_text}

Generate engaging metadata that will maximize views and engagement. MUST BE IN THE SAME LANGUAGE AS THE VIDEO TRANSCRIPT (except universal hashtags like #fyp, #reels).

OUTPUT — RETURN ONLY VALID JSON (no markdown, no comments):
{{
  "shorts": [
    {{
      "start": 0,
      "end": {video_duration},
      "video_description_for_tiktok": "<engaging TikTok description with CTA, oriented to get views>",
      "video_description_for_instagram": "<engaging Instagram description with CTA, oriented to get views>",
      "video_title_for_youtube_short": "<catchy title for YouTube Short, 100 chars max>",
      "viral_hook_text": "<SHORT punchy text overlay (max 10 words). Examples: 'POV: You realized...', 'Did you know?', 'Stop doing this!'>",
      "hashtags_tiktok": "<8-12 hashtags for TikTok. Mix of trending + niche + discovery. MUST start with #. Example: #fyp #viral #pourtoi #tech>",
      "hashtags_instagram": "<8-12 hashtags for Instagram Reels. Mix of trending + niche + community. MUST start with #. Example: #reels #explore #trending #tech>"
    }}
  ]
}}
"""

# Load the YOLO model once (Keep for backup or scene analysis if needed)
model = YOLO('yolov8n.pt')

# --- MediaPipe Setup ---
# Use standard Face Detection (BlazeFace) for speed
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

class SmoothedCameraman:
    """
    Handles smooth camera movement.
    Simplified Logic: "Heavy Tripod"
    Only moves if the subject leaves the center safe zone.
    Moves slowly and linearly.
    """
    def __init__(self, output_width, output_height, video_width, video_height):
        self.output_width = output_width
        self.output_height = output_height
        self.video_width = video_width
        self.video_height = video_height
        
        # Initial State
        self.current_center_x = video_width / 2
        self.target_center_x = video_width / 2
        
        # Calculate crop dimensions once
        self.crop_height = video_height
        self.crop_width = int(self.crop_height * ASPECT_RATIO)
        if self.crop_width > video_width:
             self.crop_width = video_width
             self.crop_height = int(self.crop_width / ASPECT_RATIO)
             
        # Safe Zone: 20% of the video width
        # As long as the target is within this zone relative to current center, DO NOT MOVE.
        self.safe_zone_radius = self.crop_width * 0.25

    def update_target(self, face_box):
        """
        Updates the target center based on detected face/person.
        """
        if face_box:
            x, y, w, h = face_box
            self.target_center_x = x + w / 2
    
    def get_crop_box(self, force_snap=False):
        """
        Returns the (x1, y1, x2, y2) for the current frame.
        """
        if force_snap:
            self.current_center_x = self.target_center_x
        else:
            diff = self.target_center_x - self.current_center_x
            
            # SIMPLIFIED LOGIC:
            # 1. Is the target outside the safe zone?
            if abs(diff) > self.safe_zone_radius:
                # 2. If yes, move towards it slowly (Linear Speed)
                # Determine direction
                direction = 1 if diff > 0 else -1
                
                # Speed: 2 pixels per frame (Slow pan)
                # If the distance is HUGE (scene change or fast movement), speed up slightly
                if abs(diff) > self.crop_width * 0.5:
                    speed = 15.0 # Fast re-frame
                else:
                    speed = 3.0  # Slow, steady pan
                
                self.current_center_x += direction * speed
                
                # Check if we overshot (prevent oscillation)
                new_diff = self.target_center_x - self.current_center_x
                if (direction == 1 and new_diff < 0) or (direction == -1 and new_diff > 0):
                    self.current_center_x = self.target_center_x
            
            # If inside safe zone, DO NOTHING (Stationary Camera)
                
        # Clamp center
        half_crop = self.crop_width / 2
        
        if self.current_center_x - half_crop < 0:
            self.current_center_x = half_crop
        if self.current_center_x + half_crop > self.video_width:
            self.current_center_x = self.video_width - half_crop
            
        x1 = int(self.current_center_x - half_crop)
        x2 = int(self.current_center_x + half_crop)
        
        x1 = max(0, x1)
        x2 = min(self.video_width, x2)
        
        y1 = 0
        y2 = self.video_height
        
        return x1, y1, x2, y2

class SpeakerTracker:
    """
    Tracks speakers over time to prevent rapid switching and handle temporary obstructions.
    """
    def __init__(self, stabilization_frames=15, cooldown_frames=30):
        self.active_speaker_id = None
        self.speaker_scores = {}  # {id: score}
        self.last_seen = {}       # {id: frame_number}
        self.locked_counter = 0   # How long we've been locked on current speaker
        
        # Hyperparameters
        self.stabilization_threshold = stabilization_frames # Frames needed to confirm a new speaker
        self.switch_cooldown = cooldown_frames              # Minimum frames before switching again
        self.last_switch_frame = -1000
        
        # ID tracking
        self.next_id = 0
        self.known_faces = [] # [{'id': 0, 'center': x, 'last_frame': 123}]

    def get_target(self, face_candidates, frame_number, width):
        """
        Decides which face to focus on.
        face_candidates: list of {'box': [x,y,w,h], 'score': float}
        """
        current_candidates = []
        
        # 1. Match faces to known IDs (simple distance tracking)
        for face in face_candidates:
            x, y, w, h = face['box']
            center_x = x + w / 2
            
            best_match_id = -1
            min_dist = width * 0.15 # Reduced matching radius to avoid jumping in groups
            
            # Try to match with known faces seen recently
            for kf in self.known_faces:
                if frame_number - kf['last_frame'] > 30: # Forgot faces older than 1s (was 2s)
                    continue
                    
                dist = abs(center_x - kf['center'])
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = kf['id']
            
            # If no match, assign new ID
            if best_match_id == -1:
                best_match_id = self.next_id
                self.next_id += 1
            
            # Update known face
            self.known_faces = [kf for kf in self.known_faces if kf['id'] != best_match_id]
            self.known_faces.append({'id': best_match_id, 'center': center_x, 'last_frame': frame_number})
            
            current_candidates.append({
                'id': best_match_id,
                'box': face['box'],
                'score': face['score']
            })

        # 2. Update Scores with decay
        for pid in list(self.speaker_scores.keys()):
             self.speaker_scores[pid] *= 0.85 # Faster decay (was 0.9)
             if self.speaker_scores[pid] < 0.1:
                 del self.speaker_scores[pid]

        # Add new scores
        for cand in current_candidates:
            pid = cand['id']
            # Score is purely based on size (proximity) now that we don't have mouth
            raw_score = cand['score'] / (width * width * 0.05)
            self.speaker_scores[pid] = self.speaker_scores.get(pid, 0) + raw_score

        # 3. Determine Best Speaker
        if not current_candidates:
            # If no one found, maintain last active speaker if cooldown allows
            # to avoid black screen or jump to 0,0
            return None 
            
        best_candidate = None
        max_score = -1
        
        for cand in current_candidates:
            pid = cand['id']
            total_score = self.speaker_scores.get(pid, 0)
            
            # Hysteresis: HUGE Bonus for current active speaker
            if pid == self.active_speaker_id:
                total_score *= 3.0 # Sticky factor
                
            if total_score > max_score:
                max_score = total_score
                best_candidate = cand

        # 4. Decide Switch
        if best_candidate:
            target_id = best_candidate['id']
            
            if target_id == self.active_speaker_id:
                self.locked_counter += 1
                return best_candidate['box']
            
            # New person
            if frame_number - self.last_switch_frame < self.switch_cooldown:
                old_cand = next((c for c in current_candidates if c['id'] == self.active_speaker_id), None)
                if old_cand:
                    return old_cand['box']
            
            self.active_speaker_id = target_id
            self.last_switch_frame = frame_number
            self.locked_counter = 0
            return best_candidate['box']
            
        return None

def detect_face_candidates(frame):
    """
    Returns list of all detected faces using lightweight FaceDetection.
    """
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    candidates = []
    
    if not results.detections:
        return []
        
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * width)
        y = int(bboxC.ymin * height)
        w = int(bboxC.width * width)
        h = int(bboxC.height * height)
        
        candidates.append({
            'box': [x, y, w, h],
            'score': w * h # Area as score
        })
            
    return candidates

def detect_person_yolo(frame):
    """
    Fallback: Detect largest person using YOLO when face detection fails.
    Returns [x, y, w, h] of the person's 'upper body' approximation.
    """
    # Use the globally loaded model
    results = model(frame, verbose=False, classes=[0]) # class 0 is person
    
    if not results:
        return None
        
    best_box = None
    max_area = 0
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            if area > max_area:
                max_area = area
                # Focus on the top 40% of the person (head/chest) for framing
                # This approximates where the face is if we can't detect it directly
                face_h = int(h * 0.4)
                best_box = [x1, y1, w, face_h]
                
    return best_box

def remove_watermark_crop(frame, crop_percent=4):
    """
    Crops a small percentage from all edges of the frame to remove watermarks/logos.
    Default 4% crop removes corner watermarks (ARTE, CNN, BBC, etc.) while preserving content.
    """
    h, w = frame.shape[:2]
    crop_x = int(w * crop_percent / 100)
    crop_y = int(h * crop_percent / 100)
    return frame[crop_y:h-crop_y, crop_x:w-crop_x]


def create_general_frame(frame, output_width, output_height):
    """
    Creates a 'General Shot' frame: 
    - Background: Blurred zoom of original
    - Foreground: Original video scaled to fit width, centered vertically.
    """
    orig_h, orig_w = frame.shape[:2]
    
    # 1. Background (Fill Height)
    # Crop center to aspect ratio
    bg_scale = output_height / orig_h
    bg_w = int(orig_w * bg_scale)
    bg_resized = cv2.resize(frame, (bg_w, output_height))
    
    # Crop center of background
    start_x = (bg_w - output_width) // 2
    if start_x < 0: start_x = 0
    background = bg_resized[:, start_x:start_x+output_width]
    if background.shape[1] != output_width:
        background = cv2.resize(background, (output_width, output_height))
        
    # Blur background
    background = cv2.GaussianBlur(background, (51, 51), 0)
    
    # 2. Foreground (Fit inside output — scale to fit both width AND height)
    scale = min(output_width / orig_w, output_height / orig_h)
    fg_w = int(orig_w * scale)
    fg_h = int(orig_h * scale)
    foreground = cv2.resize(frame, (fg_w, fg_h))
    
    # 3. Overlay (center both horizontally and vertically)
    y_offset = (output_height - fg_h) // 2
    x_offset = (output_width - fg_w) // 2

    # Clone background to avoid modifying it
    final_frame = background.copy()
    final_frame[y_offset:y_offset+fg_h, x_offset:x_offset+fg_w] = foreground

    return final_frame

def analyze_scenes_strategy(video_path, scenes):
    """
    Analyzes each scene to determine if it should be TRACK (Single person) or GENERAL (Group/Wide).
    Returns list of strategies corresponding to scenes.
    """
    cap = cv2.VideoCapture(video_path)
    strategies = []
    
    if not cap.isOpened():
        return ['TRACK'] * len(scenes)
        
    for start, end in tqdm(scenes, desc="   Analyzing Scenes"):
        # Sample 3 frames (start, middle, end)
        frames_to_check = [
            start.get_frames() + 5,
            int((start.get_frames() + end.get_frames()) / 2),
            end.get_frames() - 5
        ]
        
        face_counts = []
        for f_idx in frames_to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            # Detect faces
            candidates = detect_face_candidates(frame)
            face_counts.append(len(candidates))
            
        # Decision Logic
        if not face_counts:
            avg_faces = 0
        else:
            avg_faces = sum(face_counts) / len(face_counts)
            
        # Strategy:
        # 0 faces -> GENERAL (Landscape/B-roll)
        # 1 face -> TRACK
        # > 1.2 faces -> GENERAL (Group)
        
        if avg_faces > 1.2 or avg_faces < 0.5:
            strategies.append('GENERAL')
        else:
            strategies.append('TRACK')
            
    cap.release()
    return strategies

def detect_scenes(video_path, threshold=27.0):
    """Detect scenes using the modern open_video() API (VideoManager is deprecated and broken)."""
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    fps = video.frame_rate
    return scene_list, fps

def is_junk_scene(video_path, start_sec, end_sec, video_duration):
    """Detect if a scene is a junk scene (intro, outro, transition, end screen, text-only).

    Uses visual analysis: checks if frames are near-black/white, low complexity,
    or mostly static text screens. Also filters scenes in the last 5% of video
    that are likely end screens.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = end_sec - start_sec

    # Sample 3 frames from the scene
    sample_times = [
        start_sec + duration * 0.25,
        start_sec + duration * 0.50,
        start_sec + duration * 0.75,
    ]

    dark_count = 0
    low_complexity_count = 0

    for t in sample_times:
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = gray.mean()
        std_val = gray.std()

        # Near-black or near-white frame (end screens, transitions)
        if mean_val < 25 or mean_val > 240:
            dark_count += 1

        # Low visual complexity (text-only screens, solid colors, simple graphics)
        # Real content (people, objects, outdoor) has high std deviation
        if std_val < 35:
            low_complexity_count += 1

    cap.release()

    # If majority of frames are dark/white → junk
    if dark_count >= 2:
        return True

    # If majority of frames are low complexity → likely text screen or transition
    if low_complexity_count >= 2:
        return True

    # Scenes in the last 5% of video that are short (< 8s) are likely end screens
    if start_sec > video_duration * 0.95 and duration < 8.0:
        return True

    # Scenes in the first 3% of video that are short (< 5s) are likely intros
    if end_sec < video_duration * 0.03 and duration < 5.0:
        return True

    return False


def filter_junk_scenes(video_path, scene_bounds, video_duration):
    """Filter out junk scenes (intros, outros, transitions, end screens) from scene list.
    Returns filtered list and prints what was removed."""
    filtered = []
    removed = 0
    for i, (start, end) in enumerate(scene_bounds):
        if is_junk_scene(video_path, start, end, video_duration):
            print(f"   🗑️  Filtering junk scene {i+1}: {start:.1f}s–{end:.1f}s ({end-start:.1f}s)")
            removed += 1
        else:
            filtered.append((start, end))
    if removed:
        print(f"   🧹 Removed {removed} junk scene(s), {len(filtered)} valid scenes remaining")
    return filtered


def fit_fontsize(text, max_fontsize, max_width, char_width_ratio=0.6):
    """Calculate font size that fits text within max_width.
    char_width_ratio: approximate ratio of char width to font size for monospace-like fonts."""
    text_len = len(text)
    if text_len == 0:
        return max_fontsize
    # Estimate: each char is roughly char_width_ratio * fontsize pixels wide
    estimated_width = text_len * char_width_ratio * max_fontsize
    if estimated_width <= max_width:
        return max_fontsize
    # Scale down proportionally
    return max(int(max_fontsize * max_width / estimated_width), int(max_fontsize * 0.5))


def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def sanitize_filename(filename):
    """Remove invalid characters from filename. Also remove URL-unsafe chars like # and unicode quotes."""
    # Remove common problematic characters for filesystems AND URLs
    filename = re.sub(r'[<>:"/\\|?*#%&{}$!@`=+\[\]]', '', filename)
    # Replace unicode quotes/apostrophes with ASCII equivalent
    filename = filename.replace('\u2019', "'").replace('\u2018', "'")
    filename = filename.replace('\u201c', '').replace('\u201d', '')
    # Remove any remaining non-ASCII characters that could cause URL issues
    filename = filename.encode('ascii', 'ignore').decode('ascii')
    filename = filename.replace(' ', '_')
    # Remove consecutive underscores
    filename = re.sub(r'_+', '_', filename)
    return filename.strip('_')[:100]


def download_youtube_video(url, output_dir="."):
    """
    Downloads a YouTube video using yt-dlp.
    Returns the path to the downloaded video and the video title.
    """
    print(f"🔍 Debug: yt-dlp version: {yt_dlp.version.__version__}")
    print("📥 Downloading video from YouTube...")
    step_start_time = time.time()

    cookies_path = '/app/cookies.txt'
    cookies_env = os.environ.get("YOUTUBE_COOKIES")
    if cookies_env:
        print("🍪 Found YOUTUBE_COOKIES env var, creating cookies file inside container...")
        try:
            with open(cookies_path, 'w') as f:
                f.write(cookies_env)
            if os.path.exists(cookies_path):
                 print(f"   Debug: Cookies file created. Size: {os.path.getsize(cookies_path)} bytes")
                 with open(cookies_path, 'r') as f:
                     content = f.read(100)
                     print(f"   Debug: First 100 chars of cookie file: {content}")
        except Exception as e:
            print(f"⚠️ Failed to write cookies file: {e}")
            cookies_path = None
    else:
        cookies_path = None
        print("⚠️ YOUTUBE_COOKIES env var not found.")
    
    # Common yt-dlp options to work around YouTube bot detection.
    # extractor_args tries multiple player clients in order; tv_embed / android
    # avoid the OAuth/PO-token checks that block server IPs.
    _COMMON_YDL_OPTS = {
        'quiet': False,
        'verbose': True,
        'no_warnings': False,
        'cookiefile': cookies_path if cookies_path else None,
        'socket_timeout': 30,
        'retries': 10,
        'fragment_retries': 10,
        'nocheckcertificate': True,
        'cachedir': False,
        'js_runtimes': {'node': {}},
        'extractor_args': {
            'youtube': {
                'player_client': ['web', 'mweb', 'android', 'tv_embed'],
            }
        },
        'http_headers': {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
        },
    }

    with yt_dlp.YoutubeDL(_COMMON_YDL_OPTS) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'youtube_video')
            sanitized_title = sanitize_filename(video_title)
        except Exception as e:
            # Force print to stderr/stdout immediately so it's captured before crash
            import sys
            import traceback
            
            # Print minimal error first to ensure something gets out
            print("🚨 YOUTUBE DOWNLOAD ERROR 🚨", file=sys.stderr)
            
            error_msg = f"""
            
❌ ================================================================= ❌
❌ FATAL ERROR: YOUTUBE DOWNLOAD FAILED
❌ ================================================================= ❌
            
REASON: YouTube has blocked the download request (Error 429/Unavailable).
        This is likely a temporary IP ban on this server.

👇 SOLUTION FOR USER 👇
---------------------------------------------------------------------
1. Download the video manually to your computer.
2. Use the 'Upload Video' tab in this app to process it.
---------------------------------------------------------------------

Technical Details: {str(e)}
            """
            # Print to both streams to ensure capture
            print(error_msg, file=sys.stdout)
            print(error_msg, file=sys.stderr)
            
            # Force flush
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Wait a split second to allow buffer to drain before raising
            time.sleep(0.5)
            
            raise e
    
    output_template = os.path.join(output_dir, f'{sanitized_title}.%(ext)s')
    expected_file = os.path.join(output_dir, f'{sanitized_title}.mp4')
    if os.path.exists(expected_file):
        os.remove(expected_file)
        print(f"🗑️  Removed existing file to re-download with H.264 codec")

    # Use yt-dlp CLI via subprocess for reliable JS challenge solving
    # The Python API doesn't properly pass remote_components for EJS
    cmd = [
        'yt-dlp',
        '--remote-components', 'ejs:github',
        '--js-runtimes', 'node',
        '-f', 'bestvideo[height>=1080][vcodec!~="av01"]+bestaudio/bestvideo[vcodec!~="av01"]+bestaudio/bestvideo+bestaudio/best',
        '--merge-output-format', 'mp4',
        '-o', output_template,
        '--no-check-certificates',
        '--socket-timeout', '30',
        '--retries', '10',
        '--fragment-retries', '10',
    ]
    if cookies_path:
        cmd.extend(['--cookies', cookies_path])
    cmd.append(url)

    print(f"📥 Running: {' '.join(cmd[:8])}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️ yt-dlp stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"yt-dlp download failed: {result.stderr[-200:]}")
    print(result.stdout[-300:] if result.stdout else "")
    
    downloaded_file = os.path.join(output_dir, f'{sanitized_title}.mp4')
    
    if not os.path.exists(downloaded_file):
        for f in os.listdir(output_dir):
            if f.startswith(sanitized_title) and f.endswith('.mp4'):
                downloaded_file = os.path.join(output_dir, f)
                break
    
    step_end_time = time.time()
    print(f"✅ Video downloaded in {step_end_time - step_start_time:.2f}s: {downloaded_file}")
    
    return downloaded_file, sanitized_title

def process_video_to_vertical(input_video, final_output_video, force_general=False):
    """
    Core logic to convert horizontal video to vertical using scene detection and Active Speaker Tracking (MediaPipe).
    force_general: if True, always use blur-background GENERAL layout (skip face-tracking TRACK mode).
    """
    script_start_time = time.time()
    
    # Define temporary file paths based on the output name
    base_name = os.path.splitext(final_output_video)[0]
    temp_video_output = f"{base_name}_temp_video.mp4"
    temp_audio_output = f"{base_name}_temp_audio.aac"
    
    # Clean up previous temp files if they exist
    if os.path.exists(temp_video_output): os.remove(temp_video_output)
    if os.path.exists(temp_audio_output): os.remove(temp_audio_output)
    if os.path.exists(final_output_video): os.remove(final_output_video)

    print(f"🎬 Processing clip: {input_video}")
    print("   Step 1: Detecting scenes...")
    scenes, fps = detect_scenes(input_video)
    
    if not scenes:
        print("   ❌ No scenes were detected. Using full video as one scene.")
        # If scene detection fails or finds nothing, treat whole video as one scene
        cap = cv2.VideoCapture(input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        from scenedetect import FrameTimecode
        scenes = [(FrameTimecode(0, fps), FrameTimecode(total_frames, fps))]

    print(f"   ✅ Found {len(scenes)} scenes.")

    print("\n   🧠 Step 2: Preparing Active Tracking...")
    raw_width, raw_height = get_video_resolution(input_video)

    # Account for watermark crop (4% from each side)
    crop_pct = 4
    original_width = raw_width - 2 * int(raw_width * crop_pct / 100)
    original_height = raw_height - 2 * int(raw_height * crop_pct / 100)
    print(f"   ✂️  Watermark crop: {raw_width}x{raw_height} → {original_width}x{original_height}")

    # Target 1080x1920 (Full HD vertical) as minimum output resolution
    # For higher-res sources (1440p, 4K), scale up proportionally
    if original_height >= 1080:
        OUTPUT_WIDTH = max(1080, int(original_height * ASPECT_RATIO))
        OUTPUT_HEIGHT = int(OUTPUT_WIDTH / ASPECT_RATIO)
    else:
        # For lower-res sources, upscale to 1080x1920
        OUTPUT_WIDTH = 1080
        OUTPUT_HEIGHT = 1920
    if OUTPUT_WIDTH % 2 != 0:
        OUTPUT_WIDTH += 1
    if OUTPUT_HEIGHT % 2 != 0:
        OUTPUT_HEIGHT += 1
    print(f"   📐 Output resolution: {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} (source: {original_width}x{original_height})")

    # Initialize Cameraman
    cameraman = SmoothedCameraman(OUTPUT_WIDTH, OUTPUT_HEIGHT, original_width, original_height)
    
    # --- New Strategy: Per-Scene Analysis ---
    print("\n   🤖 Step 3: Analyzing Scenes for Strategy (Single vs Group)...")
    scene_strategies = analyze_scenes_strategy(input_video, scenes)
    # scene_strategies is a list of 'TRACK' or 'General' corresponding to scenes
    
    print("\n   ✂️ Step 4: Processing video frames...")
    
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-', '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-an', temp_video_output
    ]

    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_number = 0
    current_scene_index = 0
    
    # Pre-calculate scene boundaries
    scene_boundaries = []
    for s_start, s_end in scenes:
        scene_boundaries.append((s_start.get_frames(), s_end.get_frames()))

    # Global tracker for single-person shots
    speaker_tracker = SpeakerTracker(cooldown_frames=30)

    with tqdm(total=total_frames, desc="   Processing", file=sys.stdout) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Remove watermarks/logos by cropping edges (4% from each side)
            frame = remove_watermark_crop(frame, crop_percent=4)

            # Update Scene Index
            if current_scene_index < len(scene_boundaries):
                start_f, end_f = scene_boundaries[current_scene_index]
                if frame_number >= end_f and current_scene_index < len(scene_boundaries) - 1:
                    current_scene_index += 1
            
            # Determine Strategy for current frame based on scene
            current_strategy = 'GENERAL' if force_general else (
                scene_strategies[current_scene_index] if current_scene_index < len(scene_strategies) else 'TRACK'
            )

            # Apply Strategy
            if current_strategy == 'GENERAL':
                # "Plano General" -> Blur Background + Fit Width
                output_frame = create_general_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                
                # Reset cameraman/tracker so they don't drift while inactive
                cameraman.current_center_x = original_width / 2
                cameraman.target_center_x = original_width / 2
                
            else:
                # "Single Speaker" -> Track & Crop
                
                # Detect every 2nd frame for performance
                if frame_number % 2 == 0:
                    candidates = detect_face_candidates(frame)
                    target_box = speaker_tracker.get_target(candidates, frame_number, original_width)
                    if target_box:
                        cameraman.update_target(target_box)
                    else:
                        person_box = detect_person_yolo(frame)
                        if person_box:
                            cameraman.update_target(person_box)

                # Snap camera on scene change to avoid panning from previous scene position
                is_scene_start = (frame_number == scene_boundaries[current_scene_index][0])
                
                x1, y1, x2, y2 = cameraman.get_crop_box(force_snap=is_scene_start)
                
                # Crop
                if y2 > y1 and x2 > x1:
                    cropped = frame[y1:y2, x1:x2]
                    output_frame = cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                else:
                    output_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

            ffmpeg_process.stdin.write(output_frame.tobytes())
            frame_number += 1
            pbar.update(1)
    
    ffmpeg_process.stdin.close()
    stderr_output = ffmpeg_process.stderr.read().decode()
    ffmpeg_process.wait()
    cap.release()

    if ffmpeg_process.returncode != 0:
        print("\n   ❌ FFmpeg frame processing failed.")
        print("   Stderr:", stderr_output)
        return False

    print("\n   🔊 Step 5: Extracting audio...")
    audio_extract_command = [
        'ffmpeg', '-y', '-i', input_video, '-vn', '-acodec', 'copy', temp_audio_output
    ]
    try:
        subprocess.run(audio_extract_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("\n   ❌ Audio extraction failed (maybe no audio?). Proceeding without audio.")
        pass

    print("\n   ✨ Step 6: Merging...")
    if os.path.exists(temp_audio_output):
        merge_command = [
            'ffmpeg', '-y', '-i', temp_video_output, '-i', temp_audio_output,
            '-c:v', 'copy', '-c:a', 'copy', final_output_video
        ]
    else:
         merge_command = [
            'ffmpeg', '-y', '-i', temp_video_output,
            '-c:v', 'copy', final_output_video
        ]
        
    try:
        subprocess.run(merge_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"   ✅ Clip saved to {final_output_video}")
    except subprocess.CalledProcessError as e:
        print("\n   ❌ Final merge failed.")
        print("   Stderr:", e.stderr.decode())
        return False

    # Clean up temp files
    if os.path.exists(temp_video_output): os.remove(temp_video_output)
    if os.path.exists(temp_audio_output): os.remove(temp_audio_output)
    
    return True

def transcribe_video(video_path):
    print("🎙️  Transcribing video with Faster-Whisper (CPU Optimized)...")
    from faster_whisper import WhisperModel
    
    # Run on CPU with INT8 quantization for speed
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
    segments, info = model.transcribe(video_path, word_timestamps=True)
    
    print(f"   Detected language '{info.language}' with probability {info.language_probability:.2f}")
    
    # Convert to openai-whisper compatible format
    transcript_segments = []
    full_text = ""
    
    for segment in segments:
        # Print progress to keep user informed (and prevent timeouts feeling)
        print(f"   [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        
        seg_dict = {
            'text': segment.text,
            'start': segment.start,
            'end': segment.end,
            'words': []
        }
        
        if segment.words:
            for word in segment.words:
                seg_dict['words'].append({
                    'word': word.word,
                    'start': word.start,
                    'end': word.end,
                    'probability': word.probability
                })
        
        transcript_segments.append(seg_dict)
        full_text += segment.text + " "
        
    return {
        'text': full_text.strip(),
        'segments': transcript_segments,
        'language': info.language
    }

def get_viral_clips(transcript_result, video_duration):
    print("🤖  Analyzing with Gemini...")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in environment variables.")
        return None


    client = genai.Client(api_key=api_key)
    
    # We use gemini-2.5-flash as requested.
    model_name = 'gemini-2.5-flash' 
    
    print(f"🤖  Initializing Gemini with model: {model_name}")

    # Extract words
    words = []
    for segment in transcript_result['segments']:
        for word in segment.get('words', []):
            words.append({
                'w': word['word'],
                's': word['start'],
                'e': word['end']
            })

    prompt = GEMINI_PROMPT_TEMPLATE.format(
        video_duration=video_duration,
        transcript_text=json.dumps(transcript_result['text']),
        words_json=json.dumps(words)
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        
        # --- Cost Calculation ---
        try:
            usage = response.usage_metadata
            if usage:
                # Gemini 2.5 Flash Pricing (Dec 2025)
                # Input: $0.10 per 1M tokens
                # Output: $0.40 per 1M tokens
                
                input_price_per_million = 0.10
                output_price_per_million = 0.40
                
                prompt_tokens = usage.prompt_token_count
                output_tokens = usage.candidates_token_count
                
                input_cost = (prompt_tokens / 1_000_000) * input_price_per_million
                output_cost = (output_tokens / 1_000_000) * output_price_per_million
                total_cost = input_cost + output_cost
                
                cost_analysis = {
                    "input_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": total_cost,
                    "model": model_name
                }

                print(f"💰 Token Usage ({model_name}):")
                print(f"   - Input Tokens: {prompt_tokens} (${input_cost:.6f})")
                print(f"   - Output Tokens: {output_tokens} (${output_cost:.6f})")
                print(f"   - Total Estimated Cost: ${total_cost:.6f}")
                
        except Exception as e:
            print(f"⚠️ Could not calculate cost: {e}")
            cost_analysis = None
        # ------------------------

        # Clean response if it contains markdown code blocks
        text = response.text
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        result_json = json.loads(text)
        if cost_analysis:
            result_json['cost_analysis'] = cost_analysis
            
        return result_json
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return None

def get_ranking_clips(transcript_result, video_duration, num_clips=6, scene_boundaries=None, video_path=None):
    """Use Gemini with VIDEO INPUT to identify ranked moments (TOP N → TOP 1).

    Uploads the actual video file to Gemini so the AI can SEE the content,
    not just read the transcript. This is critical for compilation videos
    where the transcript is mostly reactions/noise.
    """
    print(f"🏆 Analyzing with Gemini Vision (Ranking mode, TOP {num_clips})...")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found.")
        return None

    client = genai.Client(api_key=api_key)
    model_name = 'gemini-2.5-flash'

    words = []
    for segment in transcript_result['segments']:
        for word in segment.get('words', []):
            words.append({'w': word['word'], 's': word['start'], 'e': word['end']})

    use_scene_mode = scene_boundaries and len(scene_boundaries) >= num_clips

    # Upload video file to Gemini for visual analysis
    uploaded_file = None
    if video_path and os.path.exists(video_path):
        try:
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            print(f"   📤 Uploading video to Gemini ({file_size_mb:.1f} MB)...")
            uploaded_file = client.files.upload(file=video_path)
            print(f"   ✅ Video uploaded: {uploaded_file.name}")

            # Wait for file to be processed (must reach ACTIVE state)
            import time
            max_wait = 300  # 5 minutes max
            waited = 0
            while waited < max_wait:
                uploaded_file = client.files.get(name=uploaded_file.name)
                state_str = str(uploaded_file.state).upper() if uploaded_file.state else ""
                if "ACTIVE" in state_str:
                    print(f"   ✅ Video processed and ACTIVE (waited {waited}s)")
                    break
                if "FAILED" in state_str:
                    print(f"   ❌ Video processing FAILED")
                    uploaded_file = None
                    break
                print(f"   ⏳ Waiting for video processing... ({waited}s, state: {state_str})")
                time.sleep(5)
                waited += 5
            else:
                print(f"   ❌ Video processing timeout after {max_wait}s")
                uploaded_file = None
        except Exception as e:
            print(f"   ⚠️ Video upload failed: {e}, falling back to text-only mode")
            uploaded_file = None

    if use_scene_mode:
        print(f"   🎯 Scene-based mode: {len(scene_boundaries)} scenes detected")

        scene_lines = []
        for i, (s, e) in enumerate(scene_boundaries):
            duration = e - s
            scene_lines.append(f"[Scene {i+1}] {s:.1f}s–{e:.1f}s ({duration:.1f}s)")

        scene_list_text = "\n".join(scene_lines)

        prompt = GEMINI_RANKING_PROMPT_TEMPLATE.format(
            num_clips=num_clips,
            video_duration=video_duration,
            scene_list=scene_list_text
        )
    else:
        print(f"   ⚠️ No scene boundaries (or too few), using free-form timestamp mode")
        prompt = GEMINI_RANKING_PROMPT_TEMPLATE_NO_SCENES.format(
            num_clips=num_clips,
            video_duration=video_duration,
            transcript_text=json.dumps(transcript_result['text']),
            words_json=json.dumps(words)
        )

    try:
        # Send video + prompt to Gemini (video first for best results)
        if uploaded_file:
            print(f"   🎬 Sending video + prompt to Gemini Vision...")
            contents = [uploaded_file, prompt]
        else:
            contents = prompt
        response = client.models.generate_content(model=model_name, contents=contents)

        try:
            usage = response.usage_metadata
            if usage:
                input_price_per_million = 0.10
                output_price_per_million = 0.40
                prompt_tokens = usage.prompt_token_count
                output_tokens = usage.candidates_token_count
                input_cost = (prompt_tokens / 1_000_000) * input_price_per_million
                output_cost = (output_tokens / 1_000_000) * output_price_per_million
                cost_analysis = {
                    "input_tokens": prompt_tokens, "output_tokens": output_tokens,
                    "input_cost": input_cost, "output_cost": output_cost,
                    "total_cost": input_cost + output_cost, "model": model_name
                }
                print(f"💰 Total cost: ${cost_analysis['total_cost']:.6f}")
        except Exception:
            cost_analysis = None

        text = response.text
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        result_json = json.loads(text)
        if cost_analysis:
            result_json['cost_analysis'] = cost_analysis

        # === MAP SCENE IDs TO TIMESTAMPS (scene-based mode) ===
        if use_scene_mode and 'shorts' in result_json:
            for clip in result_json['shorts']:
                scene_id = clip.get('scene_id')
                if scene_id is not None:
                    idx = max(0, min(scene_id - 1, len(scene_boundaries) - 1))
                    scene_start, scene_end = scene_boundaries[idx]
                    # ALWAYS use full scene — Gemini's trim points cut too aggressively
                    clip['start'] = scene_start
                    clip['end'] = scene_end
                    print(f"   ✅ Rank {clip.get('rank')}: Scene {idx+1} = {clip['start']:.1f}s–{clip['end']:.1f}s (full scene)")
                else:
                    print(f"   ⚠️ Rank {clip.get('rank')}: missing scene_id, using raw start/end")

        # === DEDUPLICATE CLIPS ===
        if 'shorts' in result_json:
            original_count = len(result_json['shorts'])
            deduped = []
            seen_scene_ids = set()
            used_ranges = []  # list of (start, end) tuples

            for clip in result_json['shorts']:
                # Check duplicate scene_id
                scene_id = clip.get('scene_id')
                if scene_id is not None and scene_id in seen_scene_ids:
                    print(f"   🔄 Removing duplicate: Rank {clip.get('rank')} uses same scene {scene_id}")
                    continue

                # Check overlapping time ranges (>50% overlap = duplicate)
                clip_start = clip.get('start', 0)
                clip_end = clip.get('end', 0)
                clip_dur = clip_end - clip_start
                is_overlap = False
                if clip_dur > 0:
                    for used_s, used_e in used_ranges:
                        overlap_start = max(clip_start, used_s)
                        overlap_end = min(clip_end, used_e)
                        overlap = max(0, overlap_end - overlap_start)
                        if overlap > clip_dur * 0.5:
                            print(f"   🔄 Removing overlapping clip: Rank {clip.get('rank')} ({clip_start:.1f}–{clip_end:.1f}s)")
                            is_overlap = True
                            break
                if is_overlap:
                    continue

                if scene_id is not None:
                    seen_scene_ids.add(scene_id)
                used_ranges.append((clip_start, clip_end))
                deduped.append(clip)

            if len(deduped) < original_count:
                print(f"   🧹 Deduplicated: {original_count} → {len(deduped)} clips")
            result_json['shorts'] = deduped

        # === POST-GEMINI JUNK FILTER (programmatic validation) ===
        if 'shorts' in result_json and video_path and os.path.exists(video_path):
            validated = []
            for clip in result_json['shorts']:
                clip_start = clip.get('start', 0)
                clip_end = clip.get('end', 0)
                if is_junk_scene(video_path, clip_start, clip_end, video_duration):
                    print(f"   🗑️ Post-filter: rejecting junk clip Rank {clip.get('rank', '?')} ({clip_start:.1f}s–{clip_end:.1f}s)")
                else:
                    validated.append(clip)
            if len(validated) < len(result_json['shorts']):
                print(f"   🧹 Post-filter: {len(result_json['shorts'])} → {len(validated)} clips")
                # Re-assign ranks
                for i, clip in enumerate(validated):
                    clip['rank'] = len(validated) - i
            result_json['shorts'] = validated

        # === REFINE TITLES WITH FRAME EXTRACTION (second pass) ===
        if 'shorts' in result_json and video_path and os.path.exists(video_path):
            result_json['shorts'] = refine_ranking_titles_with_frames(
                video_path, result_json['shorts'], api_key
            )

        # === REJECT BAD TITLES (text/screen/watching indicators) ===
        JUNK_TITLE_WORDS = {'SCREEN', 'TEXT', 'DISPLAYS', 'WATCHING', 'SUBSCRIBE', 'THANKS', 'CREDITS', 'INTRO', 'OUTRO', 'LOGO'}
        if 'shorts' in result_json:
            for clip in result_json['shorts']:
                title = clip.get('ranking_title', '')
                title_words = set(title.upper().split())
                if title_words & JUNK_TITLE_WORDS:
                    print(f"   ⚠️ Bad title detected: '{title}' → replacing with 'EPIC FAIL MOMENT'")
                    clip['ranking_title'] = 'EPIC FAIL MOMENT'

        # Ensure clips are ordered rank N → 1 (ascending rank = descending position)
        if 'shorts' in result_json:
            result_json['shorts'].sort(key=lambda x: x.get('rank', 999), reverse=True)

        return result_json
    except Exception as e:
        print(f"❌ Gemini Ranking Error: {e}")
        return None
    finally:
        # Cleanup uploaded video file from Gemini storage
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
                print(f"   🗑️ Cleaned up uploaded video from Gemini")
            except Exception:
                pass


def refine_ranking_titles_with_frames(video_path, clips, api_key):
    """Second pass: extract mid-point frames from each clip and ask Gemini to describe the visual action.
    This fixes the problem of Gemini assigning wrong titles when watching the full video."""
    print(f"   🎯 Refining {len(clips)} ranking titles with frame extraction...")

    client = genai.Client(api_key=api_key)
    model_name = 'gemini-2.5-flash'

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("   ⚠️ Cannot open video for frame extraction, keeping original titles")
        return clips

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_data = []
    temp_frame_paths = []

    for i, clip in enumerate(clips):
        clip_start = clip.get('start', 0)
        clip_end = clip.get('end', 0)
        clip_dur = clip_end - clip_start
        # Extract 3 frames (25%, 50%, 75%) for better context
        sample_times = [
            clip_start + clip_dur * 0.25,
            clip_start + clip_dur * 0.50,
            clip_start + clip_dur * 0.75,
        ]
        clip_frame_paths = []
        for j, t in enumerate(sample_times):
            frame_num = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                temp_path = f"/tmp/ranking_frame_{i}_{j}.jpg"
                cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                clip_frame_paths.append(temp_path)
                temp_frame_paths.append(temp_path)
        if clip_frame_paths:
            frames_data.append({
                'rank': clip.get('rank', i + 1),
                'paths': clip_frame_paths,
                'time': f"{clip_start:.1f}s–{clip_end:.1f}s"
            })
        else:
            frames_data.append(None)

    cap.release()

    valid_frames = [f for f in frames_data if f is not None]
    if not valid_frames:
        print("   ⚠️ No frames extracted, keeping original titles")
        return clips

    # Build prompt with all frames (3 per clip for better context)
    prompt_lines = [
        "Below you will see GROUPS of 3 images. Each group of 3 images shows frames from the SAME video clip (beginning, middle, end) in a fail compilation.",
        "For EACH GROUP, describe the PHYSICAL ACTION happening in that clip in 3-4 words, ALL CAPS, no emoji.",
        "Look at ALL 3 frames together to understand the motion/action — what is moving, falling, crashing, etc.",
        "GOOD examples: 'FALLS OFF SKATEBOARD', 'CAR HITS WALL', 'DOG STEALS FOOD', 'SLIPS ON ICE'",
        "BAD examples: anything generic like 'FUNNY MOMENT', 'EPIC FAIL', or text/audio based descriptions.",
        "",
        f"Return exactly {len(valid_frames)} titles (one per clip group).",
        "RETURN ONLY VALID JSON (no markdown):",
        '{"titles": ["TITLE FOR CLIP 1", "TITLE FOR CLIP 2", ...]}'
    ]
    prompt = "\n".join(prompt_lines)

    try:
        from PIL import Image
        contents = []
        for f in valid_frames:
            contents.append(f"--- Clip for Rank {f['rank']} ({f['time']}) ---")
            for path in f['paths']:
                img = Image.open(path)
                contents.append(img)
        contents.append(prompt)

        response = client.models.generate_content(model=model_name, contents=contents)
        text = response.text
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        result = json.loads(text)
        new_titles = result.get('titles', [])

        if len(new_titles) == len(valid_frames):
            valid_idx = 0
            for i, clip in enumerate(clips):
                if frames_data[i] is not None:
                    old_title = clip.get('ranking_title', '')
                    new_title = new_titles[valid_idx].strip().upper()
                    if new_title and len(new_title.split()) <= 6:
                        clip['ranking_title'] = new_title
                        if old_title != new_title:
                            print(f"   ✅ Rank {clip.get('rank')}: '{old_title}' → '{new_title}'")
                    valid_idx += 1
        else:
            print(f"   ⚠️ Title count mismatch ({len(new_titles)} vs {len(valid_frames)}), keeping originals")

    except Exception as e:
        print(f"   ⚠️ Title refinement failed: {e}, keeping original titles")
    finally:
        for p in temp_frame_paths:
            if os.path.exists(p):
                os.remove(p)

    return clips


def get_short_metadata(transcript_result, video_duration):
    """Generate metadata (titles, descriptions, hashtags) for a short video without clip detection."""
    print("🤖  Analyzing short video with Gemini (metadata only)...")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in environment variables.")
        return None

    client = genai.Client(api_key=api_key)
    model_name = 'gemini-2.5-flash'

    prompt = GEMINI_SHORT_PROMPT_TEMPLATE.format(
        video_duration=round(video_duration, 3),
        transcript_text=json.dumps(transcript_result['text']),
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )

        # Cost calculation
        cost_analysis = None
        try:
            usage = response.usage_metadata
            if usage:
                input_price_per_million = 0.10
                output_price_per_million = 0.40
                prompt_tokens = usage.prompt_token_count
                output_tokens = usage.candidates_token_count
                input_cost = (prompt_tokens / 1_000_000) * input_price_per_million
                output_cost = (output_tokens / 1_000_000) * output_price_per_million
                total_cost = input_cost + output_cost
                cost_analysis = {
                    "input_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": total_cost,
                    "model": model_name
                }
                print(f"💰 Token Usage ({model_name}): ${total_cost:.6f}")
        except Exception as e:
            print(f"⚠️ Could not calculate cost: {e}")

        text = response.text
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        result_json = json.loads(text)
        if cost_analysis:
            result_json['cost_analysis'] = cost_analysis
        return result_json
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return None

def process_short_video(input_video, output_video):
    """Process an already-short video: remove watermark, re-encode. Skip reframing if already vertical."""
    width, height = get_video_resolution(input_video)
    is_vertical = height > width

    if is_vertical:
        print(f"   📱 Video is already vertical ({width}x{height}), applying watermark removal + re-encode...")

        # Calculate crop for watermark removal (4% from each edge)
        crop_pct = 4
        crop_x = int(width * crop_pct / 100)
        crop_y = int(height * crop_pct / 100)
        cropped_w = width - 2 * crop_x
        cropped_h = height - 2 * crop_y
        # Ensure even dimensions
        if cropped_w % 2 != 0:
            cropped_w -= 1
        if cropped_h % 2 != 0:
            cropped_h -= 1

        cmd = [
            'ffmpeg', '-y', '-i', input_video,
            '-vf', f'crop={cropped_w}:{cropped_h}:{crop_x}:{crop_y}',
            '-c:v', 'libx264', '-crf', '18', '-preset', 'medium',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '192k',
            output_video
        ]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"   ⚠️ FFmpeg error: {result.stderr.decode()[-300:]}")
            return False
        print(f"   ✅ Short video processed: {output_video}")
        return True
    else:
        print(f"   🔄 Video is horizontal ({width}x{height}), converting to vertical...")
        return process_video_to_vertical(input_video, output_video)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoCrop-Vertical with Viral Clip Detection.")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', type=str, help="Path to the input video file.")
    input_group.add_argument('-u', '--url', type=str, help="YouTube URL to download and process.")
    
    parser.add_argument('-o', '--output', type=str, help="Output directory or file (if processing whole video).")
    parser.add_argument('--keep-original', action='store_true', help="Keep the downloaded YouTube video.")
    parser.add_argument('--skip-analysis', action='store_true', help="Skip AI analysis and convert the whole video.")
    parser.add_argument('--mode', type=str, default='ranking', choices=['normal', 'ranking'],
                        help="'normal' = extract only the single best viral clip. 'ranking' = extract all clips (for auto-compilation).")
    parser.add_argument('--force-general', action='store_true',
                        help="Force GENERAL (blur background) layout for all scenes, skipping face-tracking mode.")
    
    args = parser.parse_args()

    script_start_time = time.time()
    
    def _ensure_dir(path: str) -> str:
        """Create directory if missing and return the same path."""
        if path:
            os.makedirs(path, exist_ok=True)
        return path
    
    # 1. Get Input Video
    if args.url:
        # For multi-clip runs, treat --output as an OUTPUT DIRECTORY (create it if needed).
        # For whole-video runs (--skip-analysis), --output can be a file path.
        if args.output and not args.skip_analysis:
            output_dir = _ensure_dir(args.output)
        else:
            # If output is a directory, use it; if it's a filename, use its directory; else default "."
            if args.output and os.path.isdir(args.output):
                output_dir = args.output
            elif args.output and not os.path.isdir(args.output):
                output_dir = os.path.dirname(args.output) or "."
            else:
                output_dir = "."
        
        input_video, video_title = download_youtube_video(args.url, output_dir)
    else:
        input_video = args.input
        video_title = os.path.splitext(os.path.basename(input_video))[0]
        
        if args.output and not args.skip_analysis:
            # For multi-clip runs, treat --output as an OUTPUT DIRECTORY (create it if needed).
            output_dir = _ensure_dir(args.output)
        else:
            # If output is a directory, use it; if it's a filename, use its directory; else default to input dir.
            if args.output and os.path.isdir(args.output):
                output_dir = args.output
            elif args.output and not os.path.isdir(args.output):
                output_dir = os.path.dirname(args.output) or os.path.dirname(input_video)
            else:
                output_dir = os.path.dirname(input_video)

    if not os.path.exists(input_video):
        print(f"❌ Input file not found: {input_video}")
        exit(1)

    # 2. Get video duration and decide processing mode
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()

    SHORT_VIDEO_THRESHOLD = 90  # seconds

    if args.skip_analysis:
        print("⏩ Skipping analysis, processing entire video...")
        output_file = args.output if args.output else os.path.join(output_dir, f"{video_title}_vertical.mp4")
        process_video_to_vertical(input_video, output_file)
    elif duration <= SHORT_VIDEO_THRESHOLD:
        # --- SHORT REPOST MODE ---
        print(f"\n📱 Short video detected ({duration:.1f}s <= {SHORT_VIDEO_THRESHOLD}s) — Short Repost mode")

        # Transcribe
        transcript = transcribe_video(input_video)

        # Get metadata from Gemini (titles, descriptions, hashtags — no clip detection)
        clips_data = get_short_metadata(transcript, duration)

        if not clips_data or 'shorts' not in clips_data:
            print("⚠️ Gemini metadata generation failed. Processing with default metadata.")
            clips_data = {
                "shorts": [{
                    "start": 0,
                    "end": round(duration, 3),
                    "video_description_for_tiktok": "",
                    "video_description_for_instagram": "",
                    "video_title_for_youtube_short": video_title,
                    "viral_hook_text": "",
                    "hashtags_tiktok": "",
                    "hashtags_instagram": ""
                }]
            }

        # Save metadata
        clips_data['transcript'] = transcript
        metadata_file = os.path.join(output_dir, f"{video_title}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(clips_data, f, indent=2)
        print(f"   Saved metadata to {metadata_file}")

        # Process the single clip (remove watermark, skip reframing if already vertical)
        clip_filename = f"{video_title}_clip_1.mp4"
        clip_final_path = os.path.join(output_dir, clip_filename)

        print(f"\n🎬 Processing Short: {clip_filename}")
        success = process_short_video(input_video, clip_final_path)
        if success:
            print(f"   ✅ Short ready: {clip_final_path}")
    else:
        # --- LONG VIDEO MODE (existing pipeline) ---
        # Transcribe
        transcript = transcribe_video(input_video)

        # Gemini Analysis
        RANKING_CLIPS_PER_SHORT = 5  # 5 clips per ranking Short (TOP 5 → TOP 1)
        RANKING_NUM_SHORTS = 3  # generate 3 ranking shorts
        RANKING_MAX_CLIP_DURATION = 60  # no real limit — trust Gemini's scene selection
        if args.mode == 'ranking':
            # Run scene detection BEFORE Gemini to identify visual cuts
            # Moderate threshold (18.0) — catches real scene changes but keeps clips long enough
            print("🎬 Pre-detecting scenes for ranking alignment (threshold=18.0)...")
            pre_scenes, _ = detect_scenes(input_video, threshold=18.0)
            scene_bounds = [(s.get_seconds(), e.get_seconds()) for s, e in pre_scenes] if pre_scenes else None
            if scene_bounds:
                # Filter out short scenes (< 4s) — too short for a good ranking clip
                scene_bounds = [(s, e) for s, e in scene_bounds if e - s >= 4.0]
                # Filter out junk scenes (intros, outros, transitions, end screens)
                print(f"   🔍 Filtering junk scenes (intros, outros, transitions, end screens)...")
                scene_bounds = filter_junk_scenes(input_video, scene_bounds, duration)
                print(f"   ✅ {len(scene_bounds)} valid scenes remaining")
                for i, (s, e) in enumerate(scene_bounds):
                    print(f"      Scene {i+1}: {s:.1f}s – {e:.1f}s ({e-s:.1f}s)")
            else:
                print("   ⚠️ No scenes detected, Gemini will use free-form timestamps")
            total_clips_needed = RANKING_CLIPS_PER_SHORT * RANKING_NUM_SHORTS
            clips_data = get_ranking_clips(transcript, duration, num_clips=total_clips_needed, scene_boundaries=scene_bounds, video_path=input_video)
        else:
            clips_data = get_viral_clips(transcript, duration)

        if not clips_data or 'shorts' not in clips_data:
            print("❌ Failed to identify clips. Converting whole video as fallback.")
            output_file = os.path.join(output_dir, f"{video_title}_vertical.mp4")
            process_video_to_vertical(input_video, output_file)
        else:
            print(f"🔥 Found {len(clips_data['shorts'])} viral clips!")

            # In Normal mode, keep only the single best (first) clip
            if args.mode == 'normal':
                clips_data['shorts'] = clips_data['shorts'][:1]
                print(f"🎯 Normal mode: keeping only the best clip.")

            # Save metadata
            clips_data['transcript'] = transcript
            metadata_file = os.path.join(output_dir, f"{video_title}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(clips_data, f, indent=2)
            print(f"   Saved metadata to {metadata_file}")

            if args.mode == 'ranking':
                # --- RANKING MODE: 5 ranked clips (5-8s each) = ~25-40s ---
                from hooks import strip_emojis

                # Snap each clip to nearest scene boundary (only for fallback/no-scene mode)
                already_scene_mapped = scene_bounds and len(scene_bounds) >= RANKING_CLIPS_PER_SHORT
                if scene_bounds and len(scene_bounds) > 1 and not already_scene_mapped:
                    print(f"\n🔒 Snapping {len(clips_data['shorts'])} clips to scene boundaries...")
                    used_scenes = set()
                    for clip in clips_data['shorts']:
                        clip_mid = (clip['start'] + clip['end']) / 2
                        best_scene = None
                        best_dist = float('inf')
                        for si, (s_start, s_end) in enumerate(scene_bounds):
                            if si in used_scenes:
                                continue
                            scene_mid = (s_start + s_end) / 2
                            dist = abs(clip_mid - scene_mid)
                            if dist < best_dist:
                                best_dist = dist
                                best_scene = (si, s_start, s_end)
                        if best_scene:
                            used_scenes.add(best_scene[0])
                            old_start, old_end = clip['start'], clip['end']
                            clip['start'] = best_scene[1]
                            clip['end'] = best_scene[2]
                            print(f"   Rank {clip.get('rank')}: {old_start:.1f}–{old_end:.1f}s → snapped to {clip['start']:.1f}–{clip['end']:.1f}s ({clip['end']-clip['start']:.1f}s)")

                # Enforce max clip duration: trim to best RANKING_MAX_CLIP_DURATION seconds
                for clip in clips_data['shorts']:
                    clip_dur = clip['end'] - clip['start']
                    if clip_dur > RANKING_MAX_CLIP_DURATION:
                        excess = clip_dur - RANKING_MAX_CLIP_DURATION
                        new_start = clip['start'] + excess / 2
                        clip['start'] = new_start
                        clip['end'] = new_start + RANKING_MAX_CLIP_DURATION
                        print(f"   ✂️ Rank {clip.get('rank')}: trimmed to {RANKING_MAX_CLIP_DURATION}s ({clip['start']:.1f}–{clip['end']:.1f}s)")

                # Validate and sanitize ranking titles
                for clip in clips_data['shorts']:
                    title = clip.get('ranking_title', '')
                    # Strip emojis and excessive whitespace
                    title = strip_emojis(title).strip()
                    # Reject titles that are too long (>6 words), empty, or look like transcript noise
                    words_in_title = title.split()
                    if not title or len(words_in_title) > 6 or len(title) > 50:
                        rank = clip.get('rank', '?')
                        clip['ranking_title'] = f"EPIC FAIL #{rank}"
                        print(f"   ⚠️ Rank {rank}: bad title '{title}' → fallback '{clip['ranking_title']}'")
                    else:
                        clip['ranking_title'] = title.upper()

                # Sort all clips by rank descending
                all_shorts = clips_data['shorts']
                all_shorts.sort(key=lambda x: x.get('rank', 999), reverse=True)

                # Split into batches of RANKING_CLIPS_PER_SHORT
                batches = []
                for i in range(0, len(all_shorts), RANKING_CLIPS_PER_SHORT):
                    batch = all_shorts[i:i + RANKING_CLIPS_PER_SHORT]
                    if len(batch) >= 3:  # need at least 3 clips for a ranking
                        batches.append(batch)

                print(f"\n🎬 === Generating {len(batches)} Ranking Short(s) from {len(all_shorts)} clips ===")

                # --- PARALLEL CLIP PROCESSING ---
                from concurrent.futures import ThreadPoolExecutor, as_completed

                # Collect ALL clips across all batches for parallel processing
                all_clip_jobs = []  # (batch_idx, local_i, clip, seg_temp, seg_vertical)
                for batch_idx, batch in enumerate(batches):
                    for i, clip in enumerate(batch):
                        clip['_local_rank'] = len(batch) - i
                    for local_i, clip in enumerate(batch):
                        seg_filename = f"temp_seg_{batch_idx}_{local_i}.mp4"
                        seg_temp = os.path.join(output_dir, f"temp_raw_{seg_filename}")
                        seg_vertical = os.path.join(output_dir, f"temp_vert_{seg_filename}")
                        all_clip_jobs.append((batch_idx, local_i, clip, seg_temp, seg_vertical))

                def _process_one_clip(job):
                    """Extract + vertical reframe a single clip. Runs in a thread."""
                    b_idx, l_i, clip, seg_temp, seg_vertical = job
                    local_rank = clip['_local_rank']
                    start, end = clip['start'], clip['end']
                    clip_dur = end - start
                    ranking_title = clip.get('ranking_title', f'TOP {local_rank}')
                    print(f"  ⏳ Processing TOP {local_rank} (batch {b_idx+1}): {start:.1f}s–{end:.1f}s ({clip_dur:.1f}s) — {ranking_title}")

                    cut_cmd = [
                        'ffmpeg', '-y', '-ss', str(start),
                        '-i', input_video,
                        '-t', str(clip_dur),
                        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-r', '30',
                        '-c:a', 'aac', seg_temp
                    ]
                    subprocess.run(cut_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

                    success = process_video_to_vertical(seg_temp, seg_vertical, force_general=True)
                    if os.path.exists(seg_temp):
                        os.remove(seg_temp)
                    if not success:
                        print(f"  ⚠️ Reframe failed for batch {b_idx+1} clip {l_i+1}, skipping")
                        return None

                    try:
                        dur_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', seg_vertical]
                        seg_dur = float(subprocess.check_output(dur_cmd).decode().strip())
                    except Exception:
                        seg_dur = clip_dur

                    print(f"  ✅ TOP {local_rank} ready ({seg_dur:.1f}s)")
                    return (b_idx, l_i, local_rank, ranking_title, seg_vertical, seg_dur)

                PARALLEL_WORKERS = min(3, len(all_clip_jobs))
                print(f"   🚀 Processing {len(all_clip_jobs)} clips with {PARALLEL_WORKERS} parallel workers...")
                clip_results = {}  # (batch_idx, local_i) → result
                with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                    futures = {executor.submit(_process_one_clip, job): job for job in all_clip_jobs}
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            b_idx, l_i, _, _, _, _ = result
                            clip_results[(b_idx, l_i)] = result

                # Reassemble results per batch (in original order)
                for batch_idx, batch in enumerate(batches):
                    print(f"\n🎬 --- Ranking Short #{batch_idx + 1}: TOP {len(batch)} → TOP 1 ---")

                    segment_paths = []
                    segment_durations = []
                    segment_ranks = []
                    segment_titles = []

                    for local_i in range(len(batch)):
                        r = clip_results.get((batch_idx, local_i))
                        if r:
                            _, _, local_rank, ranking_title, seg_vertical, seg_dur = r
                            segment_paths.append(seg_vertical)
                            segment_durations.append(seg_dur)
                            segment_ranks.append(local_rank)
                            segment_titles.append(ranking_title)

                    if len(segment_paths) < 2:
                        print(f"  ❌ Not enough segments for Ranking #{batch_idx+1}, skipping")
                        continue

                    # Concatenate segments
                    concat_raw = os.path.join(output_dir, f"{video_title}_ranking_raw_{batch_idx}.mp4")
                    concat_list = os.path.join(output_dir, f"temp_concat_{batch_idx}.txt")
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
                    for sp in segment_paths:
                        if os.path.exists(sp):
                            os.remove(sp)

                    if result.returncode != 0:
                        print(f"❌ Concat failed for Ranking #{batch_idx+1}")
                        continue

                    # Burn cumulative scoreboard overlay
                    print(f"  📊 Applying scoreboard overlay...")

                    cumulative_time = 0.0
                    seg_time_ranges = []
                    for dur in segment_durations:
                        seg_time_ranges.append((cumulative_time, cumulative_time + dur))
                        cumulative_time += dur

                    title_files = []
                    for i, (rank, title) in enumerate(zip(segment_ranks, segment_titles)):
                        txt_path = os.path.join(output_dir, f"temp_sb_{batch_idx}_{i}.txt")
                        clean = strip_emojis(title).strip() or f'TOP {rank}'
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(f"TOP {rank}  {clean}")
                        title_files.append(txt_path)

                    try:
                        dim_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', concat_raw]
                        dims = subprocess.check_output(dim_cmd).decode().strip().split('\n')[0].split('x')
                        vw, vh = int(dims[0]), int(dims[1])
                    except Exception:
                        vw, vh = 1080, 1920

                    font_spec = "font=Liberation Sans Bold"
                    max_active_fontsize = int(vw * 0.046)
                    max_past_fontsize = int(vw * 0.036)
                    line_height = int(vh * 0.035)
                    base_y = int(vh * 0.55)
                    margin_x = int(vw * 0.05)
                    max_text_width = vw - 2 * margin_x

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

                        # Active state
                        filter_parts.append(
                            f"drawtext=textfile='{txt_esc}':fontsize={active_fontsize}:fontcolor=black@0.5"
                            f":x={margin_x}+2:y={line_y}+2:{font_spec}:enable='between(t,{seg_start:.3f},{seg_end:.3f})'"
                        )
                        filter_parts.append(
                            f"drawtext=textfile='{txt_esc}':fontsize={active_fontsize}:fontcolor=white"
                            f":borderw=4:bordercolor=black:x={margin_x}:y={line_y}:{font_spec}:enable='between(t,{seg_start:.3f},{seg_end:.3f})'"
                        )
                        # Past state
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

                    short_output = os.path.join(output_dir, f"{video_title}_ranking_{batch_idx + 1}.mp4")
                    overlay_cmd = [
                        'ffmpeg', '-y', '-i', concat_raw,
                        '-vf', filter_chain,
                        '-c:a', 'copy', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
                        short_output
                    ]

                    overlay_result = subprocess.run(overlay_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    if os.path.exists(concat_raw):
                        os.remove(concat_raw)
                    for tf in title_files:
                        if os.path.exists(tf):
                            os.remove(tf)

                    if overlay_result.returncode == 0:
                        print(f"🏆 Ranking Short #{batch_idx + 1} ready: {short_output}")
                    else:
                        print(f"❌ Overlay failed for #{batch_idx + 1}: {overlay_result.stderr.decode()[:200]}")

            else:
                # --- NORMAL/VIRAL MODE: process each clip individually ---
                for i, clip in enumerate(clips_data['shorts']):
                    start = clip['start']
                    end = clip['end']
                    print(f"\n🎬 Processing Clip {i+1}: {start}s - {end}s")
                    print(f"   Title: {clip.get('video_title_for_youtube_short', 'No Title')}")

                    clip_filename = f"{video_title}_clip_{i+1}.mp4"
                    clip_temp_path = os.path.join(output_dir, f"temp_{clip_filename}")
                    clip_final_path = os.path.join(output_dir, clip_filename)

                    cut_command = [
                        'ffmpeg', '-y',
                        '-ss', str(start),
                        '-i', input_video,
                        '-t', str(end - start),
                        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-c:a', 'aac',
                        clip_temp_path
                    ]
                    subprocess.run(cut_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

                    force_gen = getattr(args, 'force_general', False)
                    success = process_video_to_vertical(clip_temp_path, clip_final_path, force_general=force_gen)

                    if success:
                        print(f"   ✅ Clip {i+1} ready: {clip_final_path}")

                    if os.path.exists(clip_temp_path):
                        os.remove(clip_temp_path)

    # Clean up original if requested
    if args.url and not args.keep_original and os.path.exists(input_video):
        os.remove(input_video)
        print(f"🗑️  Cleaned up downloaded video.")

    total_time = time.time() - script_start_time
    print(f"\n⏱️  Total execution time: {total_time:.2f}s")
