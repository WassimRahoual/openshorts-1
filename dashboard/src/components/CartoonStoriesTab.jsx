import React, { useState, useEffect } from 'react';
import { Sparkles, Play, Download, Loader2, RefreshCw, ChevronRight, ChevronLeft, Volume2, Eye, Film, Palette, BookOpen, Globe } from 'lucide-react';
import { getApiUrl } from '../config';

const STYLE_OPTIONS = [
  { id: 'pixar_3d', label: 'Pixar 3D', desc: 'Vibrant, cinematic Pixar style' },
  { id: 'anime', label: 'Anime', desc: 'Studio Ghibli inspired' },
  { id: 'watercolor', label: 'Watercolor', desc: 'Soft dreamy storybook' },
  { id: 'comic', label: 'Comic Book', desc: 'Bold graphic novel style' },
  { id: 'chibi', label: 'Chibi / Kawaii', desc: 'Cute big-eyed characters' },
  { id: 'dark_fantasy', label: 'Dark Fantasy', desc: 'Moody gothic fairytale' },
  { id: 'flat_modern', label: 'Flat Modern', desc: 'Clean minimalist digital art' },
];

const GENRE_OPTIONS = [
  { id: 'fairy_tale', label: 'Conte / Fairy Tale' },
  { id: 'horror', label: 'Horreur / Horror' },
  { id: 'comedy', label: 'Comedie / Comedy' },
  { id: 'moral', label: 'Morale / Life Lesson' },
  { id: 'mystery', label: 'Mystere / Mystery' },
  { id: 'romance', label: 'Romance' },
  { id: 'adventure', label: 'Aventure / Adventure' },
  { id: 'scary_kids', label: 'Scary (Kids)' },
];

const LANGUAGE_OPTIONS = [
  { id: 'fr', label: 'Francais' },
  { id: 'en', label: 'English' },
  { id: 'es', label: 'Espanol' },
  { id: 'ar', label: 'Arabe' },
  { id: 'de', label: 'Deutsch' },
  { id: 'pt', label: 'Portugues' },
  { id: 'it', label: 'Italiano' },
  { id: 'ja', label: 'Japanese' },
];

export default function CartoonStoriesTab({ geminiApiKey, elevenLabsKey, falKey }) {
  const [step, setStep] = useState(0);

  // Step 0: Config
  const [topic, setTopic] = useState('');
  const [genre, setGenre] = useState('fairy_tale');
  const [style, setStyle] = useState('pixar_3d');
  const [language, setLanguage] = useState('fr');
  const [numScenes, setNumScenes] = useState(6);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Step 1: Story review
  const [story, setStory] = useState(null);
  const [editedNarration, setEditedNarration] = useState('');

  // Step 2: Voice selection
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState('');
  const [videoMode, setVideoMode] = useState('lowcost');

  // Step 3: Generation
  const [jobId, setJobId] = useState(null);
  const [logs, setLogs] = useState([]);
  const [genStatus, setGenStatus] = useState('');

  // Step 4: Result
  const [result, setResult] = useState(null);

  // Fetch voices on mount
  useEffect(() => {
    if (!elevenLabsKey) return;
    fetch(`${getApiUrl()}/api/saasshorts/voices`, {
      headers: { 'X-ElevenLabs-Key': elevenLabsKey },
    })
      .then(r => r.json())
      .then(data => {
        setVoices(data.voices || []);
        if (data.voices?.length > 0) setSelectedVoice(data.voices[0].voice_id);
      })
      .catch(() => {});
  }, [elevenLabsKey]);

  // Poll job status
  useEffect(() => {
    if (!jobId || genStatus === 'completed' || genStatus === 'failed') return;
    const interval = setInterval(async () => {
      try {
        const resp = await fetch(`${getApiUrl()}/api/cartoon/status/${jobId}`);
        const data = await resp.json();
        setLogs(data.logs || []);
        setGenStatus(data.status);
        if (data.status === 'completed' && data.result) {
          setResult(data.result);
          setStep(4);
          clearInterval(interval);
        } else if (data.status === 'failed') {
          setError(data.logs?.[data.logs.length - 1] || 'Generation failed');
          clearInterval(interval);
        }
      } catch {}
    }, 2000);
    return () => clearInterval(interval);
  }, [jobId, genStatus]);

  const generateStory = async () => {
    if (!topic.trim()) { setError('Enter a topic or story idea'); return; }
    if (!geminiApiKey) { setError('Set your Gemini API key in Settings'); return; }
    setLoading(true);
    setError('');
    try {
      const resp = await fetch(`${getApiUrl()}/api/cartoon/story`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Gemini-Key': geminiApiKey },
        body: JSON.stringify({ topic, genre, style, language, num_scenes: numScenes }),
      });
      if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed');
      const data = await resp.json();
      setStory(data.story);
      setEditedNarration(data.story.full_narration || '');
      setStep(1);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const startGeneration = async () => {
    if (!falKey) { setError('Set your fal.ai API key in Settings'); return; }
    if (!elevenLabsKey) { setError('Set your ElevenLabs API key in Settings'); return; }
    setError('');
    setLogs([]);
    setGenStatus('processing');
    setStep(3);

    // Update narration if edited
    const storyToSend = { ...story, full_narration: editedNarration };

    try {
      const resp = await fetch(`${getApiUrl()}/api/cartoon/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Fal-Key': falKey,
          'X-ElevenLabs-Key': elevenLabsKey,
        },
        body: JSON.stringify({
          story: storyToSend,
          voice_id: selectedVoice,
          video_mode: videoMode,
        }),
      });
      if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed');
      const data = await resp.json();
      setJobId(data.job_id);
    } catch (e) {
      setError(e.message);
      setGenStatus('failed');
    }
  };

  return (
    <div className="h-full overflow-y-auto px-4 py-6 md:px-8">
      <div className="max-w-3xl mx-auto space-y-6">

        {/* Header */}
        <div className="text-center space-y-2">
          <h1 className="text-3xl font-black bg-gradient-to-r from-pink-400 via-purple-400 to-blue-400 bg-clip-text text-transparent">
            Cartoon Stories
          </h1>
          <p className="text-zinc-400">Generate animated AI cartoon stories for TikTok & Reels</p>
        </div>

        {/* Step indicators */}
        <div className="flex items-center justify-center gap-2 text-sm">
          {['Story Setup', 'Review Script', 'Configure', 'Generating', 'Result'].map((label, i) => (
            <React.Fragment key={i}>
              {i > 0 && <ChevronRight size={14} className="text-zinc-600" />}
              <span className={`px-3 py-1 rounded-full ${step === i ? 'bg-purple-500/20 text-purple-400 font-semibold' : step > i ? 'text-zinc-400' : 'text-zinc-600'}`}>
                {label}
              </span>
            </React.Fragment>
          ))}
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 rounded-xl text-sm">
            {error}
          </div>
        )}

        {/* ── Step 0: Story Setup ── */}
        {step === 0 && (
          <div className="space-y-5 animate-[fadeIn_0.3s_ease-out]">

            {/* Topic */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-zinc-300">Story idea / topic</label>
              <textarea
                value={topic}
                onChange={e => setTopic(e.target.value)}
                placeholder="Ex: Un petit dragon qui a peur du feu et qui doit sauver son village..."
                className="w-full bg-zinc-800/50 border border-zinc-700/50 rounded-xl px-4 py-3 text-white placeholder-zinc-500 focus:outline-none focus:border-purple-500/50 resize-none h-24"
              />
            </div>

            {/* Genre */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                <BookOpen size={16} /> Genre
              </label>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {GENRE_OPTIONS.map(g => (
                  <button
                    key={g.id}
                    onClick={() => setGenre(g.id)}
                    className={`px-3 py-2 rounded-lg text-sm transition-colors ${genre === g.id ? 'bg-purple-500/20 text-purple-400 border border-purple-500/50' : 'bg-zinc-800/50 text-zinc-400 border border-zinc-700/30 hover:border-zinc-600'}`}
                  >
                    {g.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Style */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                <Palette size={16} /> Visual Style
              </label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                {STYLE_OPTIONS.map(s => (
                  <button
                    key={s.id}
                    onClick={() => setStyle(s.id)}
                    className={`px-3 py-2.5 rounded-lg text-left transition-colors ${style === s.id ? 'bg-purple-500/20 border border-purple-500/50' : 'bg-zinc-800/50 border border-zinc-700/30 hover:border-zinc-600'}`}
                  >
                    <div className={`text-sm font-medium ${style === s.id ? 'text-purple-400' : 'text-zinc-300'}`}>{s.label}</div>
                    <div className="text-xs text-zinc-500">{s.desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Language + Scenes */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                  <Globe size={16} /> Language
                </label>
                <select
                  value={language}
                  onChange={e => setLanguage(e.target.value)}
                  className="w-full bg-zinc-800/50 border border-zinc-700/50 rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-purple-500/50"
                >
                  {LANGUAGE_OPTIONS.map(l => (
                    <option key={l.id} value={l.id}>{l.label}</option>
                  ))}
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                  <Film size={16} /> Scenes
                </label>
                <select
                  value={numScenes}
                  onChange={e => setNumScenes(Number(e.target.value))}
                  className="w-full bg-zinc-800/50 border border-zinc-700/50 rounded-xl px-4 py-2.5 text-white focus:outline-none focus:border-purple-500/50"
                >
                  {[4, 5, 6, 7, 8].map(n => (
                    <option key={n} value={n}>{n} scenes (~{n * 5}-{n * 8}s)</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Generate button */}
            <button
              onClick={generateStory}
              disabled={loading || !topic.trim()}
              className="w-full py-3.5 rounded-xl font-semibold text-white bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all"
            >
              {loading ? (
                <><Loader2 size={18} className="animate-spin" /> Generating story...</>
              ) : (
                <><Sparkles size={18} /> Generate Story</>
              )}
            </button>
          </div>
        )}

        {/* ── Step 1: Review Story ── */}
        {step === 1 && story && (
          <div className="space-y-5 animate-[fadeIn_0.3s_ease-out]">

            <div className="bg-zinc-800/50 border border-zinc-700/30 rounded-xl p-5 space-y-3">
              <h3 className="text-lg font-bold text-white">{story.title}</h3>
              <p className="text-xs text-zinc-500">{story.scenes?.length} scenes &middot; {story.language}</p>
            </div>

            {/* Scenes preview */}
            <div className="space-y-3">
              <h4 className="text-sm font-medium text-zinc-300">Scenes</h4>
              {story.scenes?.map((scene, i) => (
                <div key={i} className="bg-zinc-800/30 border border-zinc-700/20 rounded-lg p-4 space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="bg-purple-500/20 text-purple-400 text-xs font-bold px-2 py-0.5 rounded-full">
                      Scene {scene.scene_number || i + 1}
                    </span>
                    <span className="text-xs text-zinc-500 capitalize">{scene.emotion}</span>
                  </div>
                  <p className="text-sm text-zinc-300">{scene.narration}</p>
                  <p className="text-xs text-zinc-500 italic">{scene.image_prompt?.slice(0, 120)}...</p>
                </div>
              ))}
            </div>

            {/* Editable narration */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-zinc-300">Full narration (editable)</label>
              <textarea
                value={editedNarration}
                onChange={e => setEditedNarration(e.target.value)}
                className="w-full bg-zinc-800/50 border border-zinc-700/50 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-purple-500/50 resize-none h-32 text-sm"
              />
            </div>

            <div className="flex gap-3">
              <button
                onClick={() => setStep(0)}
                className="px-5 py-2.5 rounded-xl text-zinc-400 border border-zinc-700/50 hover:border-zinc-600 transition-colors"
              >
                <ChevronLeft size={16} className="inline mr-1" /> Back
              </button>
              <button
                onClick={() => setStep(2)}
                className="flex-1 py-2.5 rounded-xl font-semibold text-white bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 flex items-center justify-center gap-2"
              >
                Configure voice & quality <ChevronRight size={16} />
              </button>
            </div>
          </div>
        )}

        {/* ── Step 2: Configure ── */}
        {step === 2 && (
          <div className="space-y-5 animate-[fadeIn_0.3s_ease-out]">

            {/* Voice selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                <Volume2 size={16} /> Narrator voice
              </label>
              <div className="grid grid-cols-2 gap-2 max-h-48 overflow-y-auto">
                {voices.map(v => (
                  <button
                    key={v.voice_id}
                    onClick={() => setSelectedVoice(v.voice_id)}
                    className={`px-3 py-2.5 rounded-lg text-left text-sm transition-colors ${selectedVoice === v.voice_id ? 'bg-purple-500/20 border border-purple-500/50 text-purple-400' : 'bg-zinc-800/50 border border-zinc-700/30 text-zinc-300 hover:border-zinc-600'}`}
                  >
                    <div className="font-medium">{v.name}</div>
                    <div className="text-xs text-zinc-500">{v.labels?.gender || v.category}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Video mode */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-zinc-300">Animation quality</label>
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={() => setVideoMode('lowcost')}
                  className={`p-4 rounded-xl text-left transition-colors ${videoMode === 'lowcost' ? 'bg-green-500/10 border-2 border-green-500/50' : 'bg-zinc-800/50 border border-zinc-700/30 hover:border-zinc-600'}`}
                >
                  <div className={`font-semibold ${videoMode === 'lowcost' ? 'text-green-400' : 'text-zinc-300'}`}>Low Cost</div>
                  <div className="text-xs text-zinc-500 mt-1">Hailuo 2.3 &middot; ~$1.50/video</div>
                </button>
                <button
                  onClick={() => setVideoMode('premium')}
                  className={`p-4 rounded-xl text-left transition-colors ${videoMode === 'premium' ? 'bg-amber-500/10 border-2 border-amber-500/50' : 'bg-zinc-800/50 border border-zinc-700/30 hover:border-zinc-600'}`}
                >
                  <div className={`font-semibold ${videoMode === 'premium' ? 'text-amber-400' : 'text-zinc-300'}`}>Premium</div>
                  <div className="text-xs text-zinc-500 mt-1">Kling v2 &middot; ~$2.50/video</div>
                </button>
              </div>
            </div>

            <div className="flex gap-3">
              <button
                onClick={() => setStep(1)}
                className="px-5 py-2.5 rounded-xl text-zinc-400 border border-zinc-700/50 hover:border-zinc-600 transition-colors"
              >
                <ChevronLeft size={16} className="inline mr-1" /> Back
              </button>
              <button
                onClick={startGeneration}
                className="flex-1 py-3 rounded-xl font-semibold text-white bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 flex items-center justify-center gap-2"
              >
                <Play size={18} /> Generate Video
              </button>
            </div>
          </div>
        )}

        {/* ── Step 3: Generating ── */}
        {step === 3 && (
          <div className="space-y-5 animate-[fadeIn_0.3s_ease-out]">
            <div className="bg-zinc-800/50 border border-zinc-700/30 rounded-xl p-5 space-y-3">
              <div className="flex items-center gap-3">
                {genStatus === 'processing' && <Loader2 size={20} className="animate-spin text-purple-400" />}
                <span className="font-semibold text-white">
                  {genStatus === 'processing' ? 'Generating your cartoon...' : genStatus === 'failed' ? 'Generation failed' : 'Done!'}
                </span>
              </div>

              <div className="bg-black/30 rounded-lg p-3 max-h-64 overflow-y-auto font-mono text-xs text-zinc-400 space-y-1">
                {logs.map((log, i) => (
                  <div key={i} className={log.includes('Error') || log.includes('error') ? 'text-red-400' : log.includes('ready') || log.includes('complete') || log.includes('cached') ? 'text-green-400' : ''}>
                    {log}
                  </div>
                ))}
                {genStatus === 'processing' && (
                  <div className="text-purple-400 animate-pulse">...</div>
                )}
              </div>
            </div>

            {genStatus === 'failed' && (
              <button
                onClick={() => { setGenStatus(''); startGeneration(); }}
                className="w-full py-3 rounded-xl font-semibold text-white bg-zinc-700 hover:bg-zinc-600 flex items-center justify-center gap-2"
              >
                <RefreshCw size={16} /> Retry
              </button>
            )}
          </div>
        )}

        {/* ── Step 4: Result ── */}
        {step === 4 && result && (
          <div className="space-y-5 animate-[fadeIn_0.3s_ease-out]">

            <div className="bg-zinc-800/50 border border-zinc-700/30 rounded-xl overflow-hidden">
              <video
                src={`${getApiUrl()}${result.video_url}`}
                controls
                autoPlay
                className="w-full max-h-[70vh] bg-black"
              />
            </div>

            <div className="flex gap-3">
              <a
                href={`${getApiUrl()}${result.video_url}`}
                download
                className="flex-1 py-3 rounded-xl font-semibold text-white bg-gradient-to-r from-purple-600 to-pink-600 flex items-center justify-center gap-2"
              >
                <Download size={18} /> Download
              </a>
              <button
                onClick={() => { setStep(0); setStory(null); setResult(null); setJobId(null); setLogs([]); setGenStatus(''); }}
                className="px-5 py-3 rounded-xl text-zinc-400 border border-zinc-700/50 hover:border-zinc-600 transition-colors"
              >
                New Story
              </button>
            </div>

            {result.cost_estimate && (
              <div className="bg-zinc-800/30 border border-zinc-700/20 rounded-lg p-3 text-xs text-zinc-500">
                <span className="font-medium text-zinc-400">Cost: </span>
                ${result.cost_estimate.total?.toFixed(2)}
                {Object.entries(result.cost_estimate).filter(([k]) => k !== 'total').map(([k, v]) => (
                  <span key={k} className="ml-3">{k.replace(/_/g, ' ')}: ${v.toFixed?.(3) || v}</span>
                ))}
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}
