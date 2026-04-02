import React, { useState, useEffect, useCallback } from 'react';
import { Heart, Trash2, Search, Film, Clock, Star, ChevronDown, ExternalLink, Play, Pause, BarChart3, FileText, Layers } from 'lucide-react';
import { getApiUrl } from '../config';

export default function ClipLibrary() {
  const [clips, setClips] = useState([]);
  const [videos, setVideos] = useState([]);
  const [stats, setStats] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeView, setActiveView] = useState('clips'); // clips, videos, jobs
  const [selectedVideoId, setSelectedVideoId] = useState(null);
  const [favoriteOnly, setFavoriteOnly] = useState(false);
  const [search, setSearch] = useState('');
  const [playingClip, setPlayingClip] = useState(null);
  const [videoDetail, setVideoDetail] = useState(null); // {scenes, analyses, transcript}

  const fetchClips = useCallback(async () => {
    try {
      const params = new URLSearchParams();
      if (selectedVideoId) params.set('video_id', selectedVideoId);
      if (favoriteOnly) params.set('favorite', 'true');
      if (search) params.set('search', search);
      const res = await fetch(getApiUrl(`/api/library/clips?${params}`));
      if (res.ok) setClips(await res.json());
    } catch (e) {
      console.error('Failed to fetch clips:', e);
    }
  }, [selectedVideoId, favoriteOnly, search]);

  const fetchVideos = async () => {
    try {
      const res = await fetch(getApiUrl('/api/library/videos'));
      if (res.ok) setVideos(await res.json());
    } catch (e) {
      console.error('Failed to fetch videos:', e);
    }
  };

  const fetchStats = async () => {
    try {
      const res = await fetch(getApiUrl('/api/library/stats'));
      if (res.ok) setStats(await res.json());
    } catch (e) {
      console.error('Failed to fetch stats:', e);
    }
  };

  const fetchJobs = async () => {
    try {
      const res = await fetch(getApiUrl('/api/library/jobs'));
      if (res.ok) setJobs(await res.json());
    } catch (e) {
      console.error('Failed to fetch jobs:', e);
    }
  };

  useEffect(() => {
    Promise.all([fetchClips(), fetchVideos(), fetchStats(), fetchJobs()]).then(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetchClips();
  }, [fetchClips]);

  const loadVideoDetail = async (videoId) => {
    setVideoDetail(null);
    try {
      const [scenesRes, analysesRes, transcriptRes] = await Promise.allSettled([
        fetch(getApiUrl(`/api/library/videos/${videoId}/scenes`)),
        fetch(getApiUrl(`/api/library/videos/${videoId}/analyses`)),
        fetch(getApiUrl(`/api/library/videos/${videoId}/transcript`)),
      ]);
      setVideoDetail({
        scenes: scenesRes.status === 'fulfilled' && scenesRes.value.ok ? await scenesRes.value.json() : [],
        analyses: analysesRes.status === 'fulfilled' && analysesRes.value.ok ? await analysesRes.value.json() : [],
        transcript: transcriptRes.status === 'fulfilled' && transcriptRes.value.ok ? await transcriptRes.value.json() : null,
      });
    } catch (e) {
      console.error('Failed to load video detail:', e);
    }
  };

  useEffect(() => {
    if (selectedVideoId) loadVideoDetail(selectedVideoId);
    else setVideoDetail(null);
  }, [selectedVideoId]);

  const toggleFavorite = async (clipId) => {
    const res = await fetch(getApiUrl(`/api/library/clips/${clipId}/favorite`), { method: 'POST' });
    if (res.ok) {
      const { favorite } = await res.json();
      setClips(prev => prev.map(c => c.id === clipId ? { ...c, favorite } : c));
    }
  };

  const deleteClip = async (clipId) => {
    if (!confirm('Delete this clip permanently?')) return;
    const res = await fetch(getApiUrl(`/api/library/clips/${clipId}`), { method: 'DELETE' });
    if (res.ok) {
      setClips(prev => prev.filter(c => c.id !== clipId));
      fetchStats();
    }
  };

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-zinc-500 animate-pulse">Loading library...</div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-6 animate-[fadeIn_0.3s_ease-out]">
      {/* Header + Stats */}
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold">Clip Library</h1>
          {stats && (
            <div className="flex items-center gap-4 text-xs text-zinc-500">
              <span><strong className="text-white">{stats.total_clips}</strong> clips</span>
              <span><strong className="text-white">{stats.total_videos}</strong> videos</span>
              <span><strong className="text-white">{stats.total_favorites}</strong> favorites</span>
              <span><strong className="text-white">{stats.completed_jobs}</strong> jobs</span>
              {stats.total_cost > 0 && (
                <span className="text-emerald-400">${stats.total_cost.toFixed(4)} spent</span>
              )}
            </div>
          )}
        </div>

        {/* View Tabs */}
        <div className="flex items-center gap-2 mb-6">
          {[
            { id: 'clips', label: 'All Clips', icon: Film },
            { id: 'videos', label: 'By Video', icon: Play },
            { id: 'jobs', label: 'Job History', icon: BarChart3 },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => { setActiveView(tab.id); setSelectedVideoId(null); }}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-colors ${
                activeView === tab.id
                  ? 'bg-blue-500/10 text-blue-400 border border-blue-500/30'
                  : 'text-zinc-400 hover:text-white hover:bg-white/5 border border-transparent'
              }`}
            >
              <tab.icon size={14} />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Clips View */}
        {(activeView === 'clips' || activeView === 'videos') && (
          <>
            {/* Filters */}
            <div className="flex items-center gap-3 mb-6">
              <div className="relative flex-1 max-w-md">
                <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" />
                <input
                  type="text"
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  placeholder="Search clips..."
                  className="w-full pl-9 pr-3 py-2 bg-surface border border-white/10 rounded-lg text-sm text-white placeholder-zinc-600 focus:border-blue-500/50 focus:outline-none"
                />
              </div>
              <button
                onClick={() => setFavoriteOnly(!favoriteOnly)}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm border transition-colors ${
                  favoriteOnly
                    ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30'
                    : 'text-zinc-400 border-white/10 hover:bg-white/5'
                }`}
              >
                <Star size={14} fill={favoriteOnly ? 'currentColor' : 'none'} />
                Favorites
              </button>
              {selectedVideoId && (
                <button
                  onClick={() => setSelectedVideoId(null)}
                  className="text-xs text-zinc-500 hover:text-white px-2 py-1 rounded border border-white/10"
                >
                  Clear filter
                </button>
              )}
            </div>

            {/* Video filter bar (in videos view) */}
            {activeView === 'videos' && !selectedVideoId && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
                {videos.map(v => (
                  <button
                    key={v.id}
                    onClick={() => { setSelectedVideoId(v.id); }}
                    className="p-4 rounded-xl border border-white/10 hover:border-white/20 hover:bg-white/5 transition-all text-left"
                  >
                    <h3 className="font-semibold text-sm text-white truncate">{v.title}</h3>
                    <div className="flex items-center gap-3 mt-2 text-[11px] text-zinc-500">
                      <span>{v.clip_count} clips</span>
                      {v.duration && <span>{Math.round(v.duration / 60)}min</span>}
                      <span>{new Date(v.created_at).toLocaleDateString()}</span>
                    </div>
                    {v.youtube_id && (
                      <span className="inline-block mt-2 text-[10px] px-2 py-0.5 rounded-full bg-red-500/10 text-red-400 border border-red-500/20">
                        YouTube
                      </span>
                    )}
                  </button>
                ))}
                {videos.length === 0 && (
                  <p className="text-zinc-600 text-sm col-span-full text-center py-8">No videos in library yet. Process a video to get started.</p>
                )}
              </div>
            )}

            {/* Video Detail Panel (scenes + analysis) */}
            {selectedVideoId && videoDetail && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
                {/* Scenes */}
                {videoDetail.scenes.length > 0 && (
                  <div className="bg-surface border border-white/10 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Layers size={14} className="text-blue-400" />
                      <h3 className="text-sm font-semibold text-white">Detected Scenes ({videoDetail.scenes.length})</h3>
                    </div>
                    <div className="max-h-48 overflow-y-auto space-y-1">
                      {videoDetail.scenes.map(s => (
                        <div key={s.id} className="flex items-center justify-between text-[11px] px-2 py-1 rounded hover:bg-white/5">
                          <span className="text-zinc-400">Scene {s.scene_index}</span>
                          <span className="text-zinc-300">{s.start_time.toFixed(1)}s - {s.end_time.toFixed(1)}s</span>
                          <span className="text-zinc-500">{s.duration?.toFixed(1)}s</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Analyses */}
                {videoDetail.analyses.length > 0 && (
                  <div className="bg-surface border border-white/10 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <BarChart3 size={14} className="text-purple-400" />
                      <h3 className="text-sm font-semibold text-white">Gemini Analyses ({videoDetail.analyses.length})</h3>
                    </div>
                    <div className="space-y-2">
                      {videoDetail.analyses.map(a => (
                        <div key={a.id} className="text-[11px] px-2 py-2 rounded bg-white/5">
                          <div className="flex items-center gap-3 text-zinc-300">
                            <span className="font-medium">{a.gemini_model || 'unknown'}</span>
                            <span>{a.mode}</span>
                            <span>{a.clip_count} clips from {a.scene_count} scenes</span>
                            {a.total_cost > 0 && <span className="text-emerald-400">${a.total_cost.toFixed(4)}</span>}
                          </div>
                          <div className="flex items-center gap-3 mt-1 text-zinc-500">
                            {a.input_tokens && <span>{a.input_tokens.toLocaleString()} in</span>}
                            {a.output_tokens && <span>{a.output_tokens.toLocaleString()} out</span>}
                            <span>{new Date(a.created_at).toLocaleString()}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Transcript */}
                {videoDetail.transcript && (
                  <div className="bg-surface border border-white/10 rounded-xl p-4 lg:col-span-2">
                    <div className="flex items-center gap-2 mb-3">
                      <FileText size={14} className="text-green-400" />
                      <h3 className="text-sm font-semibold text-white">Transcript</h3>
                      <span className="text-[10px] text-zinc-500">{videoDetail.transcript.language}</span>
                    </div>
                    <p className="text-[11px] text-zinc-400 max-h-32 overflow-y-auto leading-relaxed whitespace-pre-wrap">
                      {typeof videoDetail.transcript.full_text === 'string'
                        ? videoDetail.transcript.full_text.slice(0, 2000)
                        : JSON.stringify(videoDetail.transcript.full_text).slice(0, 2000)}
                      {(videoDetail.transcript.full_text?.length || 0) > 2000 && '...'}
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Clip Grid */}
            {(activeView === 'clips' || selectedVideoId) && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {clips.map(clip => (
                  <div key={clip.id} className="bg-surface border border-white/10 rounded-xl overflow-hidden group hover:border-white/20 transition-all">
                    {/* Video Preview */}
                    <div className="relative aspect-[9/16] bg-black">
                      <video
                        src={getApiUrl(clip.serving_url) + '#t=0.1'}
                        className="w-full h-full object-contain"
                        controls={playingClip === clip.id}
                        preload="metadata"
                        onClick={() => setPlayingClip(playingClip === clip.id ? null : clip.id)}
                      />
                      {playingClip !== clip.id && (
                        <div
                          className="absolute inset-0 flex items-center justify-center cursor-pointer bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity"
                          onClick={() => setPlayingClip(clip.id)}
                        >
                          <Play size={40} className="text-white/80" fill="currentColor" />
                        </div>
                      )}
                      {clip.rank && (
                        <span className="absolute top-2 left-2 text-[10px] font-bold px-2 py-0.5 rounded-full bg-red-500/90 text-white">
                          #{clip.rank}
                        </span>
                      )}
                    </div>

                    {/* Info */}
                    <div className="p-3">
                      <h3 className="text-sm font-semibold text-white truncate" title={clip.title}>
                        {clip.title || 'Untitled'}
                      </h3>
                      <p className="text-[11px] text-zinc-500 mt-1 truncate">{clip.video_title}</p>
                      <div className="flex items-center gap-2 mt-1 text-[10px] text-zinc-600">
                        {clip.start_time != null && clip.end_time != null && (
                          <span>{clip.start_time.toFixed(1)}s - {clip.end_time.toFixed(1)}s</span>
                        )}
                        <span>{clip.mode}</span>
                        {clip.gemini_model && <span>{clip.gemini_model.replace('gemini-', '')}</span>}
                      </div>

                      {/* Actions */}
                      <div className="flex items-center justify-between mt-3 pt-2 border-t border-white/5">
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => toggleFavorite(clip.id)}
                            className={`p-1.5 rounded-lg transition-colors ${
                              clip.favorite ? 'text-yellow-400 bg-yellow-500/10' : 'text-zinc-600 hover:text-yellow-400'
                            }`}
                          >
                            <Heart size={14} fill={clip.favorite ? 'currentColor' : 'none'} />
                          </button>
                          <button
                            onClick={() => deleteClip(clip.id)}
                            className="p-1.5 rounded-lg text-zinc-600 hover:text-red-400 transition-colors"
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                        <a
                          href={getApiUrl(clip.serving_url)}
                          download
                          className="text-[10px] text-zinc-500 hover:text-blue-400 flex items-center gap-1"
                        >
                          Download <ExternalLink size={10} />
                        </a>
                      </div>
                    </div>
                  </div>
                ))}
                {clips.length === 0 && (
                  <p className="text-zinc-600 text-sm col-span-full text-center py-12">
                    {search || favoriteOnly ? 'No clips match your filters.' : 'No clips in library yet. Process a video to get started.'}
                  </p>
                )}
              </div>
            )}
          </>
        )}

        {/* Jobs View */}
        {activeView === 'jobs' && (
          <div className="space-y-2">
            {jobs.map(job => (
              <div key={job.id} className="flex items-center justify-between p-4 bg-surface border border-white/10 rounded-xl">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-medium text-white truncate">{job.video_title || job.id.slice(0, 8)}</h3>
                    <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                      job.status === 'completed' ? 'bg-green-500/10 text-green-400 border border-green-500/20'
                        : job.status === 'failed' ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                        : 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/20'
                    }`}>
                      {job.status}
                    </span>
                  </div>
                  <div className="flex items-center gap-4 mt-1 text-[11px] text-zinc-500">
                    <span>{job.mode}</span>
                    {job.gemini_model && <span>{job.gemini_model}</span>}
                    {job.clip_count > 0 && <span>{job.clip_count} clips</span>}
                    {job.total_cost > 0 && <span className="text-emerald-500">${job.total_cost.toFixed(4)}</span>}
                    {job.duration_secs > 0 && <span>{Math.round(job.duration_secs)}s</span>}
                    <span>{new Date(job.created_at).toLocaleString()}</span>
                  </div>
                  {job.error && <p className="text-[11px] text-red-400 mt-1 truncate">{job.error}</p>}
                </div>
              </div>
            ))}
            {jobs.length === 0 && (
              <p className="text-zinc-600 text-sm text-center py-12">No job history yet.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
