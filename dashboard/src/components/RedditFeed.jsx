import React, { useState, useEffect, useCallback } from 'react';
import { ArrowUp, MessageCircle, Clock, Play, Loader2, ChevronDown, RefreshCw, AlertTriangle, Check, X, GripVertical, Trophy } from 'lucide-react';
import { getApiUrl } from '../config';

const SORT_OPTIONS = [
  { value: 'hot', label: 'Hot' },
  { value: 'top', label: 'Top' },
  { value: 'new', label: 'New' },
];

const TIME_OPTIONS = [
  { value: 'day', label: '24h' },
  { value: 'week', label: 'Week' },
  { value: 'month', label: 'Month' },
  { value: 'year', label: 'Year' },
  { value: 'all', label: 'All Time' },
];

function formatScore(n) {
  if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
  return n;
}

function timeAgo(utc) {
  const diff = Date.now() / 1000 - utc;
  if (diff < 3600) return Math.floor(diff / 60) + 'm';
  if (diff < 86400) return Math.floor(diff / 3600) + 'h';
  return Math.floor(diff / 86400) + 'd';
}

export default function RedditFeed({ geminiApiKey, onProcessStart }) {
  const [subreddits, setSubreddits] = useState([]);
  const [activeSubreddit, setActiveSubreddit] = useState('fail');
  const [sort, setSort] = useState('hot');
  const [timeFilter, setTimeFilter] = useState('week');
  const [posts, setPosts] = useState([]);
  const [after, setAfter] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  // Selection state: ordered list of selected post IDs (TOP N first → TOP 1 last)
  const [selected, setSelected] = useState([]);
  const [dragIdx, setDragIdx] = useState(null);

  // Fetch subreddit list
  useEffect(() => {
    fetch(getApiUrl('/api/reddit/subreddits'))
      .then(r => r.json())
      .then(d => setSubreddits(d.subreddits))
      .catch(() => {});
  }, []);

  const fetchPosts = useCallback(async (reset = true) => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({
        subreddit: activeSubreddit,
        sort,
        time_filter: timeFilter,
        limit: '25',
      });
      if (!reset && after) params.set('after', after);

      const resp = await fetch(getApiUrl(`/api/reddit/feed?${params}`));
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      setPosts(prev => reset ? data.posts : [...prev, ...data.posts]);
      setAfter(data.after);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [activeSubreddit, sort, timeFilter, after]);

  // Fetch on filter change
  useEffect(() => {
    fetchPosts(true);
  }, [activeSubreddit, sort, timeFilter]);

  const toggleSelect = (post) => {
    setSelected(prev => {
      if (prev.find(p => p.id === post.id)) {
        return prev.filter(p => p.id !== post.id);
      }
      if (prev.length >= 10) return prev; // max 10
      return [...prev, post];
    });
  };

  const removeFromSelection = (postId) => {
    setSelected(prev => prev.filter(p => p.id !== postId));
  };

  const moveInSelection = (fromIdx, toIdx) => {
    setSelected(prev => {
      const next = [...prev];
      const [item] = next.splice(fromIdx, 1);
      next.splice(toIdx, 0, item);
      return next;
    });
  };

  const handleSubmitRanking = async () => {
    if (!geminiApiKey) {
      setError('Gemini API key required. Set it in Settings.');
      return;
    }
    if (selected.length < 2) {
      setError('Select at least 2 clips for a ranking.');
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const resp = await fetch(getApiUrl('/api/reddit/process'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Gemini-Key': geminiApiKey,
        },
        body: JSON.stringify({
          clips: selected.map(post => ({
            video_url: post.video_url,
            reddit_post_url: post.url,
            title: post.title,
          })),
        }),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }

      const data = await resp.json();
      if (onProcessStart) {
        onProcessStart(data.job_id, `Reddit Ranking (${selected.length} clips)`, selected[selected.length - 1]?.preview_url);
      }
      setSelected([]);
    } catch (e) {
      setError(`Failed: ${e.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  const isSelected = (postId) => selected.some(p => p.id === postId);
  const selectionIndex = (postId) => selected.findIndex(p => p.id === postId);

  return (
    <div className="h-full flex flex-col p-6 overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold text-white">Reddit Fails</h1>
          <p className="text-zinc-500 text-sm mt-1">Select clips to compile into a ranked short (TOP N to TOP 1)</p>
        </div>
        <button
          onClick={() => fetchPosts(true)}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-white transition-colors text-sm"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Selection bar */}
      {selected.length > 0 && (
        <div className="mb-4 p-3 rounded-xl bg-orange-500/10 border border-orange-500/20">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-orange-400">
              <Trophy size={14} className="inline mr-1.5 -mt-0.5" />
              Ranking: {selected.length} clip{selected.length > 1 ? 's' : ''} selected
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setSelected([])}
                className="text-xs text-zinc-500 hover:text-white transition-colors"
              >
                Clear all
              </button>
              <button
                onClick={handleSubmitRanking}
                disabled={submitting || selected.length < 2}
                className="flex items-center gap-2 px-4 py-1.5 rounded-lg bg-orange-500 hover:bg-orange-600 text-white text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {submitting ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <Trophy size={14} />
                )}
                Generate TOP {selected.length}
              </button>
            </div>
          </div>

          {/* Ordered selection list (drag to reorder) */}
          <div className="flex flex-wrap gap-2 mt-2">
            {selected.map((post, idx) => {
              const rank = selected.length - idx;
              return (
                <div
                  key={post.id}
                  draggable
                  onDragStart={() => setDragIdx(idx)}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={() => { if (dragIdx !== null && dragIdx !== idx) moveInSelection(dragIdx, idx); setDragIdx(null); }}
                  onDragEnd={() => setDragIdx(null)}
                  className={`flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs cursor-grab active:cursor-grabbing transition-all ${
                    dragIdx === idx ? 'opacity-50 scale-95' : 'bg-white/5 border border-white/10'
                  }`}
                >
                  <GripVertical size={10} className="text-zinc-600" />
                  <span className="text-orange-400 font-bold">#{rank}</span>
                  <span className="text-zinc-300 max-w-[120px] truncate">{post.title}</span>
                  <button
                    onClick={() => removeFromSelection(post.id)}
                    className="text-zinc-600 hover:text-red-400 transition-colors ml-1"
                  >
                    <X size={12} />
                  </button>
                </div>
              );
            })}
          </div>
          <p className="text-[10px] text-zinc-600 mt-2">Drag to reorder. First = TOP {selected.length} (shown first), Last = TOP 1 (best, shown last).</p>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3 mb-5">
        <div className="flex flex-wrap gap-2">
          {subreddits.map(sub => (
            <button
              key={sub}
              onClick={() => setActiveSubreddit(sub)}
              className={`px-3 py-1.5 rounded-full text-xs font-medium transition-colors ${
                activeSubreddit === sub
                  ? 'bg-orange-500/20 text-orange-400 border border-orange-500/40'
                  : 'bg-white/5 text-zinc-400 border border-white/10 hover:bg-white/10 hover:text-white'
              }`}
            >
              r/{sub}
            </button>
          ))}
        </div>

        <div className="h-5 w-px bg-white/10 hidden sm:block" />

        <div className="flex gap-1.5">
          {SORT_OPTIONS.map(opt => (
            <button
              key={opt.value}
              onClick={() => setSort(opt.value)}
              className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
                sort === opt.value
                  ? 'bg-white/15 text-white'
                  : 'text-zinc-500 hover:text-white hover:bg-white/5'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>

        {sort === 'top' && (
          <div className="flex gap-1.5">
            {TIME_OPTIONS.map(opt => (
              <button
                key={opt.value}
                onClick={() => setTimeFilter(opt.value)}
                className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
                  timeFilter === opt.value
                    ? 'bg-white/15 text-white'
                    : 'text-zinc-500 hover:text-white hover:bg-white/5'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-2 p-3 mb-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
          <AlertTriangle size={14} />
          {error}
          <button onClick={() => setError(null)} className="ml-auto text-red-400/60 hover:text-red-400">&times;</button>
        </div>
      )}

      {/* Grid */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {posts.map(post => {
            const sel = isSelected(post.id);
            const idx = selectionIndex(post.id);
            const rank = sel ? selected.length - idx : null;

            return (
              <div
                key={post.id}
                onClick={() => toggleSelect(post)}
                className={`group cursor-pointer rounded-xl overflow-hidden transition-all ${
                  sel
                    ? 'bg-orange-500/10 border-2 border-orange-500/50 ring-1 ring-orange-500/20'
                    : 'bg-white/[0.03] border border-white/[0.06] hover:border-white/15 hover:bg-white/[0.05]'
                }`}
              >
                {/* Thumbnail */}
                <div className="relative aspect-video bg-black/40">
                  {post.preview_url ? (
                    <img
                      src={post.preview_url}
                      alt=""
                      className="w-full h-full object-cover"
                      loading="lazy"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-zinc-600">
                      <Play size={32} />
                    </div>
                  )}
                  {post.duration && (
                    <span className="absolute bottom-2 right-2 px-1.5 py-0.5 rounded bg-black/70 text-[10px] text-white font-mono">
                      {Math.floor(post.duration / 60)}:{String(post.duration % 60).padStart(2, '0')}
                    </span>
                  )}
                  {post.is_nsfw && (
                    <span className="absolute top-2 left-2 px-1.5 py-0.5 rounded bg-red-600/80 text-[10px] text-white font-bold">
                      NSFW
                    </span>
                  )}

                  {/* Selection indicator */}
                  {sel ? (
                    <div className="absolute top-2 right-2 w-7 h-7 rounded-full bg-orange-500 flex items-center justify-center text-white text-xs font-bold shadow-lg">
                      #{rank}
                    </div>
                  ) : (
                    <div className="absolute top-2 right-2 w-7 h-7 rounded-full bg-black/50 border border-white/20 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <span className="text-white/60 text-[10px] font-medium">+</span>
                    </div>
                  )}
                </div>

                {/* Info */}
                <div className="p-3">
                  <h3 className="text-sm font-medium text-white/90 line-clamp-2 leading-snug mb-2">
                    {post.title}
                  </h3>
                  <div className="flex items-center gap-3 text-[11px] text-zinc-500">
                    <span className="text-orange-400/70 font-medium">r/{post.subreddit}</span>
                    <span className="flex items-center gap-1">
                      <ArrowUp size={11} />
                      {formatScore(post.score)}
                    </span>
                    <span className="flex items-center gap-1">
                      <MessageCircle size={11} />
                      {formatScore(post.num_comments)}
                    </span>
                    <span className="flex items-center gap-1 ml-auto">
                      <Clock size={11} />
                      {timeAgo(post.created_utc)}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Load more */}
        {after && (
          <div className="flex justify-center py-6">
            <button
              onClick={() => fetchPosts(false)}
              disabled={loading}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 text-zinc-400 hover:text-white transition-colors text-sm font-medium"
            >
              {loading ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <ChevronDown size={14} />
              )}
              Load More
            </button>
          </div>
        )}

        {/* Empty state */}
        {!loading && posts.length === 0 && !error && (
          <div className="flex flex-col items-center justify-center py-20 text-zinc-500">
            <Play size={40} className="mb-3 opacity-30" />
            <p className="text-sm">No video posts found</p>
            <p className="text-xs mt-1">Try a different subreddit or sort</p>
          </div>
        )}

        {/* Initial loading */}
        {loading && posts.length === 0 && (
          <div className="flex flex-col items-center justify-center py-20 text-zinc-500">
            <Loader2 size={24} className="animate-spin mb-3" />
            <p className="text-sm">Loading r/{activeSubreddit}...</p>
          </div>
        )}
      </div>
    </div>
  );
}
