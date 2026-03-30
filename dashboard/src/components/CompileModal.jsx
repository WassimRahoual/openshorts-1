import React, { useState } from 'react';
import { X, Loader2, Film, ArrowUp, ArrowDown, Music, Hash } from 'lucide-react';
import { getApiUrl } from '../config';

const TRANSITIONS = [
    { value: 'fade', label: 'Fade' },
    { value: 'wipeleft', label: 'Wipe Left' },
    { value: 'wiperight', label: 'Wipe Right' },
    { value: 'slidedown', label: 'Slide Down' },
    { value: 'slideup', label: 'Slide Up' },
    { value: 'none', label: 'None (Cut)' },
];

export default function CompileModal({ isOpen, onClose, clips, jobId, onCompileComplete }) {
    const [selectedIndices, setSelectedIndices] = useState(() => clips.map((_, i) => i));
    const [rankingDirection, setRankingDirection] = useState('desc');
    const [rankingStyle, setRankingStyle] = useState('circle');
    const [rankingPosition, setRankingPosition] = useState('top-left');
    const [transitionType, setTransitionType] = useState('fade');
    const [transitionDuration, setTransitionDuration] = useState(0.5);
    const [bgMusic, setBgMusic] = useState(null);
    const [bgMusicVolume, setBgMusicVolume] = useState(0.15);
    const [isCompiling, setIsCompiling] = useState(false);

    if (!isOpen) return null;

    const toggleClip = (index) => {
        setSelectedIndices(prev =>
            prev.includes(index)
                ? prev.filter(i => i !== index)
                : [...prev, index]
        );
    };

    const moveClip = (fromIdx, direction) => {
        const arr = [...selectedIndices];
        const pos = arr.indexOf(fromIdx);
        if (pos === -1) return;
        const newPos = pos + direction;
        if (newPos < 0 || newPos >= arr.length) return;
        [arr[pos], arr[newPos]] = [arr[newPos], arr[pos]];
        setSelectedIndices(arr);
    };

    const handleCompile = async () => {
        if (selectedIndices.length < 1) return;
        setIsCompiling(true);

        try {
            const formData = new FormData();
            formData.append('job_id', jobId);
            formData.append('clip_indices', JSON.stringify(selectedIndices));
            formData.append('ranking_direction', rankingDirection);
            formData.append('ranking_style', rankingStyle);
            formData.append('ranking_position', rankingPosition);
            formData.append('transition_type', transitionType);
            formData.append('transition_duration', transitionDuration);
            formData.append('bg_music_volume', bgMusicVolume);
            if (bgMusic) {
                formData.append('bg_music', bgMusic);
            }

            const res = await fetch(getApiUrl('/api/compile'), {
                method: 'POST',
                body: formData,
            });

            const data = await res.json();
            if (data.success) {
                onCompileComplete(data.compilation_url);
                onClose();
            } else {
                alert(data.detail || 'Compilation failed');
            }
        } catch (err) {
            console.error('Compile error:', err);
            alert('Compilation failed: ' + err.message);
        } finally {
            setIsCompiling(false);
        }
    };

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-[fadeIn_0.2s_ease-out]">
            <div className="bg-[#121214] border border-white/10 p-6 rounded-2xl w-full max-w-4xl shadow-2xl relative flex flex-col md:flex-row gap-6 max-h-[90vh]">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 text-zinc-500 hover:text-white z-10"
                >
                    <X size={20} />
                </button>

                {/* Left: Clip Order Preview */}
                <div className="flex-1 flex flex-col bg-black/40 rounded-lg border border-white/5 overflow-hidden">
                    <div className="px-4 py-3 border-b border-white/5 bg-white/5">
                        <h4 className="text-sm font-bold text-zinc-300 flex items-center gap-2">
                            <Film size={14} /> Clip Order
                        </h4>
                    </div>
                    <div className="flex-1 overflow-y-auto custom-scrollbar p-3 space-y-2">
                        {selectedIndices.map((clipIdx, orderPos) => {
                            const clip = clips[clipIdx];
                            const rankNum = rankingDirection === 'desc'
                                ? selectedIndices.length - orderPos
                                : rankingDirection === 'asc'
                                    ? orderPos + 1
                                    : null;

                            return (
                                <div
                                    key={clipIdx}
                                    className="flex items-center gap-3 bg-white/5 rounded-lg p-2 border border-white/5"
                                >
                                    {rankNum && rankingStyle !== 'none' && (
                                        <div className="w-8 h-8 rounded-full bg-red-600 flex items-center justify-center text-white text-xs font-bold shrink-0">
                                            #{rankNum}
                                        </div>
                                    )}
                                    <div className="flex-1 min-w-0">
                                        <p className="text-xs text-white truncate">
                                            Clip {clipIdx + 1}
                                        </p>
                                        <p className="text-[10px] text-zinc-500 truncate">
                                            {clip?.video_title_for_youtube_short || clip?.video_description_for_tiktok || ''}
                                        </p>
                                    </div>
                                    <div className="flex gap-1 shrink-0">
                                        <button
                                            onClick={() => moveClip(clipIdx, -1)}
                                            className="p-1 text-zinc-500 hover:text-white"
                                            disabled={orderPos === 0}
                                        >
                                            <ArrowUp size={14} />
                                        </button>
                                        <button
                                            onClick={() => moveClip(clipIdx, 1)}
                                            className="p-1 text-zinc-500 hover:text-white"
                                            disabled={orderPos === selectedIndices.length - 1}
                                        >
                                            <ArrowDown size={14} />
                                        </button>
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* Clip selection checkboxes */}
                    <div className="px-3 py-2 border-t border-white/5 bg-white/5">
                        <p className="text-[10px] text-zinc-500 mb-1">Select clips:</p>
                        <div className="flex flex-wrap gap-2">
                            {clips.map((_, i) => (
                                <button
                                    key={i}
                                    onClick={() => toggleClip(i)}
                                    className={`px-2 py-1 rounded text-xs font-bold transition-all border ${selectedIndices.includes(i)
                                        ? 'bg-red-600 text-white border-red-500'
                                        : 'bg-white/5 text-zinc-500 border-white/5 hover:bg-white/10'
                                        }`}
                                >
                                    Clip {i + 1}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Right: Controls */}
                <div className="w-full md:w-80 flex flex-col">
                    <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                        <Film className="text-red-400" /> Compile Ranking
                    </h3>

                    <div className="space-y-5 flex-1 overflow-y-auto custom-scrollbar pr-2">
                        {/* Ranking Direction */}
                        <div>
                            <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                                <Hash size={12} /> Ranking
                            </label>
                            <div className="grid grid-cols-3 gap-2">
                                {[
                                    { value: 'desc', label: 'Countdown' },
                                    { value: 'asc', label: 'Count Up' },
                                    { value: 'none', label: 'None' },
                                ].map(opt => (
                                    <button
                                        key={opt.value}
                                        onClick={() => setRankingDirection(opt.value)}
                                        className={`py-2 px-1 rounded-lg text-xs font-bold transition-all border ${rankingDirection === opt.value
                                            ? 'bg-white text-black border-white'
                                            : 'bg-white/5 text-zinc-400 border-white/5 hover:bg-white/10'
                                            }`}
                                    >
                                        {opt.label}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Ranking Style */}
                        {rankingDirection !== 'none' && (
                            <div>
                                <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2 block">Number Style</label>
                                <div className="grid grid-cols-2 gap-2">
                                    {[
                                        { value: 'circle', label: 'Circle' },
                                        { value: 'bold', label: 'Bold Text' },
                                    ].map(opt => (
                                        <button
                                            key={opt.value}
                                            onClick={() => setRankingStyle(opt.value)}
                                            className={`py-2 px-1 rounded-lg text-xs font-bold transition-all border ${rankingStyle === opt.value
                                                ? 'bg-white text-black border-white'
                                                : 'bg-white/5 text-zinc-400 border-white/5 hover:bg-white/10'
                                                }`}
                                        >
                                            {opt.label}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Ranking Position */}
                        {rankingDirection !== 'none' && (
                            <div>
                                <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2 block">Number Position</label>
                                <div className="grid grid-cols-2 gap-2">
                                    {[
                                        { value: 'top-left', label: 'Top Left' },
                                        { value: 'top-right', label: 'Top Right' },
                                    ].map(opt => (
                                        <button
                                            key={opt.value}
                                            onClick={() => setRankingPosition(opt.value)}
                                            className={`py-2 px-1 rounded-lg text-xs font-bold transition-all border ${rankingPosition === opt.value
                                                ? 'bg-white text-black border-white'
                                                : 'bg-white/5 text-zinc-400 border-white/5 hover:bg-white/10'
                                                }`}
                                        >
                                            {opt.label}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Transition Type */}
                        <div>
                            <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2 block">Transition</label>
                            <select
                                value={transitionType}
                                onChange={e => setTransitionType(e.target.value)}
                                className="w-full bg-black/40 border border-white/10 rounded-xl p-2.5 text-white text-sm focus:outline-none focus:border-red-500/50"
                            >
                                {TRANSITIONS.map(t => (
                                    <option key={t.value} value={t.value}>{t.label}</option>
                                ))}
                            </select>
                        </div>

                        {/* Transition Duration */}
                        {transitionType !== 'none' && (
                            <div>
                                <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2 block">
                                    Transition Duration: {transitionDuration}s
                                </label>
                                <input
                                    type="range"
                                    min="0.3"
                                    max="1.5"
                                    step="0.1"
                                    value={transitionDuration}
                                    onChange={e => setTransitionDuration(parseFloat(e.target.value))}
                                    className="w-full accent-red-500"
                                />
                            </div>
                        )}

                        {/* Background Music */}
                        <div>
                            <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                                <Music size={12} /> Background Music
                            </label>
                            <input
                                type="file"
                                accept="audio/*"
                                onChange={e => setBgMusic(e.target.files[0] || null)}
                                className="w-full text-xs text-zinc-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-bold file:bg-white/10 file:text-zinc-300 hover:file:bg-white/20"
                            />
                            {bgMusic && (
                                <div className="mt-2">
                                    <label className="text-[10px] text-zinc-500">
                                        Volume: {Math.round(bgMusicVolume * 100)}%
                                    </label>
                                    <input
                                        type="range"
                                        min="0.05"
                                        max="0.5"
                                        step="0.05"
                                        value={bgMusicVolume}
                                        onChange={e => setBgMusicVolume(parseFloat(e.target.value))}
                                        className="w-full accent-red-500"
                                    />
                                </div>
                            )}
                        </div>

                        <div className="p-3 bg-white/5 rounded-lg border border-white/5 text-[11px] text-zinc-400">
                            <strong>Tip:</strong> Select at least 3-5 clips for a good ranking compilation. Countdown (#5 to #1) works best for engagement.
                        </div>
                    </div>

                    <button
                        onClick={handleCompile}
                        disabled={isCompiling || selectedIndices.length < 1}
                        className="w-full py-4 mt-4 bg-gradient-to-r from-red-500 to-orange-600 hover:from-red-400 hover:to-orange-500 text-white font-bold rounded-xl shadow-lg shadow-red-500/20 transition-all active:scale-[0.98] flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
                    >
                        {isCompiling ? <Loader2 size={20} className="animate-spin" /> : <Film size={20} />}
                        {isCompiling ? 'Compiling...' : `Compile ${selectedIndices.length} Clips`}
                    </button>
                </div>
            </div>
        </div>
    );
}
