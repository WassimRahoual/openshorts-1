"""
Reddit video fetcher for OpenShorts.
Uses Reddit's public JSON API (no auth required) to browse viral fail videos
and yt-dlp to download them.
"""

import os
import json
import subprocess
import tempfile
import logging
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

DEFAULT_SUBREDDITS = [
    "fail",
    "Whatcouldgowrong",
    "holdmybeer",
    "IdiotsInCars",
    "therewasanattempt",
    "instant_regret",
]

REDDIT_HEADERS = {
    "User-Agent": "OpenShorts/1.0 (video clipper)"
}


async def fetch_reddit_feed(
    subreddit: str = "fail",
    sort: str = "hot",
    time_filter: str = "week",
    after: Optional[str] = None,
    limit: int = 25,
) -> dict:
    """
    Fetch video posts from a subreddit using Reddit's public JSON API.
    Returns posts that contain playable video (v.redd.it, imgur, streamable, etc).
    """
    if subreddit not in DEFAULT_SUBREDDITS:
        raise ValueError(f"Subreddit r/{subreddit} not in allowed list")

    url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
    params = {"limit": min(limit, 100), "raw_json": 1}
    if sort == "top":
        params["t"] = time_filter
    if after:
        params["after"] = after

    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        resp = await client.get(url, params=params, headers=REDDIT_HEADERS)
        resp.raise_for_status()
        data = resp.json()

    posts = []
    for child in data.get("data", {}).get("children", []):
        post = child.get("data", {})
        if not post:
            continue

        # Filter for video content
        is_video = post.get("is_video", False)
        has_reddit_video = "reddit_video" in post.get("media", {}) if post.get("media") else False
        post_url = post.get("url", "")
        # Dead domains to exclude
        DEAD_DOMAINS = ["gfycat.com", "gifv.com", "redgifs.com"]
        if any(d in post_url for d in DEAD_DOMAINS):
            continue

        is_external_video = any(
            domain in post_url
            for domain in ["imgur.com/", "streamable.com/", "v.redd.it"]
        )

        if not (is_video or has_reddit_video or is_external_video):
            continue

        # Extract video URL and thumbnail
        video_url = post_url
        preview_url = None
        duration = None

        if has_reddit_video:
            rv = post["media"]["reddit_video"]
            video_url = rv.get("fallback_url", video_url)
            duration = rv.get("duration")

        # Get preview image
        if post.get("preview", {}).get("images"):
            preview_url = post["preview"]["images"][0].get("source", {}).get("url")

        # Use thumbnail as fallback
        if not preview_url and post.get("thumbnail", "").startswith("http"):
            preview_url = post["thumbnail"]

        posts.append({
            "id": post.get("id"),
            "title": post.get("title", ""),
            "subreddit": post.get("subreddit", subreddit),
            "url": f"https://www.reddit.com{post.get('permalink', '')}",
            "video_url": video_url,
            "preview_url": preview_url,
            "score": post.get("score", 0),
            "upvote_ratio": post.get("upvote_ratio", 0),
            "num_comments": post.get("num_comments", 0),
            "created_utc": post.get("created_utc", 0),
            "duration": duration,
            "is_nsfw": post.get("over_18", False),
        })

    return {
        "posts": posts,
        "after": data.get("data", {}).get("after"),
        "subreddit": subreddit,
        "sort": sort,
    }


def download_reddit_video(video_url: str, reddit_post_url: str, output_dir: str) -> str:
    """
    Download a Reddit video using yt-dlp.
    yt-dlp handles v.redd.it (merges video+audio), imgur, streamable, etc.
    Returns the path to the downloaded file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "reddit_%(id)s.%(ext)s")

    # Use the reddit post URL — yt-dlp extracts the best video from it
    download_url = reddit_post_url if "reddit.com" in reddit_post_url else video_url

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-warnings",
        "--socket-timeout", "30",
        "--retries", "5",
        download_url,
    ]

    logger.info(f"Downloading Reddit video: {download_url}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        logger.error(f"yt-dlp failed: {result.stderr}")
        raise RuntimeError(f"Download failed: {result.stderr[:500]}")

    # Find the downloaded file
    for f in os.listdir(output_dir):
        if f.startswith("reddit_") and f.endswith(".mp4"):
            return os.path.join(output_dir, f)

    # Fallback: any mp4 file
    for f in os.listdir(output_dir):
        if f.endswith(".mp4"):
            return os.path.join(output_dir, f)

    raise RuntimeError("No video file found after download")
