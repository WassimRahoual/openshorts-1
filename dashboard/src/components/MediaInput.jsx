import React, { useState } from 'react';
import { Youtube, Upload, FileVideo, X, Plus, Trash2 } from 'lucide-react';

export default function MediaInput({ onProcess, isProcessing }) {
    const [mode, setMode] = useState('url'); // 'url' | 'file'
    const [urls, setUrls] = useState(['']);
    const [file, setFile] = useState(null);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (mode === 'url') {
            const validUrls = urls.map(u => u.trim()).filter(Boolean);
            if (validUrls.length === 0) return;
            if (validUrls.length === 1) {
                onProcess({ type: 'url', payload: validUrls[0] });
            } else {
                onProcess({ type: 'urls', payload: validUrls });
            }
        } else if (mode === 'file' && file) {
            onProcess({ type: 'file', payload: file });
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
            setMode('file');
        }
    };

    const addUrl = () => setUrls(prev => [...prev, '']);
    const removeUrl = (i) => setUrls(prev => prev.filter((_, idx) => idx !== i));
    const updateUrl = (i, val) => setUrls(prev => prev.map((u, idx) => idx === i ? val : u));

    const hasValidInput = mode === 'url'
        ? urls.some(u => u.trim())
        : !!file;

    return (
        <div className="bg-surface border border-white/5 rounded-2xl p-6 animate-[fadeIn_0.6s_ease-out]">
            <div className="flex gap-4 mb-6 border-b border-white/5 pb-4">
                <button
                    onClick={() => setMode('url')}
                    className={`flex items-center gap-2 pb-2 px-2 transition-all ${mode === 'url'
                        ? 'text-primary border-b-2 border-primary -mb-[17px]'
                        : 'text-zinc-400 hover:text-white'}`}
                >
                    <Youtube size={18} />
                    YouTube URL
                </button>
                <button
                    onClick={() => setMode('file')}
                    className={`flex items-center gap-2 pb-2 px-2 transition-all ${mode === 'file'
                        ? 'text-primary border-b-2 border-primary -mb-[17px]'
                        : 'text-zinc-400 hover:text-white'}`}
                >
                    <Upload size={18} />
                    Upload File
                </button>
            </div>

            <form onSubmit={handleSubmit}>
                {mode === 'url' ? (
                    <div className="space-y-2">
                        {urls.map((url, i) => (
                            <div key={i} className="flex gap-2 items-center">
                                <input
                                    type="url"
                                    value={url}
                                    onChange={(e) => updateUrl(i, e.target.value)}
                                    placeholder="https://www.youtube.com/watch?v=..."
                                    className="input-field flex-1"
                                />
                                {urls.length > 1 && (
                                    <button
                                        type="button"
                                        onClick={() => removeUrl(i)}
                                        className="p-2 text-zinc-500 hover:text-red-400 transition-colors"
                                    >
                                        <Trash2 size={16} />
                                    </button>
                                )}
                            </div>
                        ))}
                        <button
                            type="button"
                            onClick={addUrl}
                            className="flex items-center gap-1.5 text-xs text-zinc-500 hover:text-primary transition-colors mt-1 px-1"
                        >
                            <Plus size={14} /> Ajouter une URL
                        </button>
                    </div>
                ) : (
                    <div
                        className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${file ? 'border-primary/50 bg-primary/5' : 'border-zinc-700 hover:border-zinc-500 bg-white/5'}`}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                    >
                        {file ? (
                            <div className="flex items-center justify-center gap-3 text-white">
                                <FileVideo className="text-primary" />
                                <span className="font-medium">{file.name}</span>
                                <button type="button" onClick={() => setFile(null)} className="p-1 hover:bg-white/10 rounded-full">
                                    <X size={16} />
                                </button>
                            </div>
                        ) : (
                            <label className="cursor-pointer block">
                                <input
                                    type="file"
                                    accept="video/*"
                                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                                    className="hidden"
                                />
                                <Upload className="mx-auto mb-3 text-zinc-500" size={24} />
                                <p className="text-zinc-400">Click to upload or drag and drop</p>
                                <p className="text-xs text-zinc-600 mt-1">MP4, MOV up to 500MB</p>
                            </label>
                        )}
                    </div>
                )}

                <button
                    type="submit"
                    disabled={isProcessing || !hasValidInput}
                    className="w-full btn-primary mt-6 flex items-center justify-center gap-2"
                >
                    {isProcessing ? (
                        <>
                            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            Processing...
                        </>
                    ) : (
                        <>
                            {mode === 'url' && urls.filter(u => u.trim()).length > 1
                                ? `Générer ${urls.filter(u => u.trim()).length} Rankings`
                                : 'Generate Clips'}
                        </>
                    )}
                </button>
            </form>
        </div>
    );
}
