import { useState, useRef, useEffect, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";

const CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"];

const CLASS_CONFIG = {
  cardboard: { emoji: "📦", bg: "bg-amber-900/30",   border: "border-amber-700/50",   bar: "bg-amber-500",   text: "text-amber-400",   tip: "Flatten boxes before recycling" },
  glass:     { emoji: "🫙", bg: "bg-cyan-900/30",    border: "border-cyan-700/50",    bar: "bg-cyan-400",    text: "text-cyan-400",    tip: "Rinse containers before recycling" },
  metal:     { emoji: "🥫", bg: "bg-slate-700/30",   border: "border-slate-500/50",   bar: "bg-slate-400",   text: "text-slate-300",   tip: "Empty and rinse cans before recycling" },
  paper:     { emoji: "📄", bg: "bg-yellow-900/30",  border: "border-yellow-700/50",  bar: "bg-yellow-400",  text: "text-yellow-400",  tip: "Keep paper dry for recycling" },
  plastic:   { emoji: "♻️", bg: "bg-emerald-900/30", border: "border-emerald-700/50", bar: "bg-emerald-400", text: "text-emerald-400", tip: "Check the recycling number on the bottom" },
  trash:     { emoji: "🗑️", bg: "bg-red-900/30",     border: "border-red-800/50",     bar: "bg-red-500",     text: "text-red-400",     tip: "Dispose in general waste bin" },
};

export default function App() {
  const [model, setModel]           = useState(null);
  const [modelLoading, setLoading]  = useState(true);
  const [image, setImage]           = useState(null);
  const [results, setResults]       = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [dragOver, setDragOver]     = useState(false);
  const [revealed, setRevealed]     = useState(false);
  const fileInputRef = useRef(null);
  const imgRef       = useRef(null);

  useEffect(() => {
    (async () => {
      try {
        
        const m = await tf.loadGraphModel(
  "https://huggingface.co/jedidiah117/waste-classifier/resolve/main/model.json?download=true"
);
setModel(m);
      } catch (e) {
        console.error("Model load error:", e);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith("image/")) return;
    setResults(null);
    setRevealed(false);
    setImage(URL.createObjectURL(file));
  }, []);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  };

 const predict = async () => {
  if (!model || !imgRef.current) return;
  setPredicting(true);
  try {
    const tensor = tf.browser
      .fromPixels(imgRef.current)
      .resizeBilinear([224, 224])
      .toFloat().div(127.5).sub(1)
      .expandDims(0);

    const output = model.predict(tensor);
    tensor.dispose();

    // graph models return an object, extract the actual tensor
    const preds = await (output instanceof tf.Tensor
      ? output
      : Object.values(output)[0]
    ).data();

    const sorted = Array.from(preds)
      .map((prob, i) => ({ label: CLASS_NAMES[i], prob }))
      .sort((a, b) => b.prob - a.prob);

    setResults(sorted);
    setTimeout(() => setRevealed(true), 100);
  } catch (e) {
    console.error(e);
  } finally {
    setPredicting(false);
  }
};

  const reset = () => {
    setImage(null);
    setResults(null);
    setRevealed(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const top    = results?.[0];
  const topCfg = top ? CLASS_CONFIG[top.label] : null;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col" style={{ fontFamily: "'DM Sans', sans-serif" }}>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
        @keyframes scan { 0% { top: 0% } 100% { top: 100% } }
        @keyframes fadein { from { opacity:0; transform:translateY(6px) } to { opacity:1; transform:translateY(0) } }
        .animate-scan { animation: scan 1.2s ease-in-out infinite; }
        .animate-fadein { animation: fadein 0.4s ease forwards; }
      `}</style>

      {/* ── Header ── */}
      <header className="sticky top-0 z-20 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-md">
        <div className="max-w-5xl mx-auto px-6 h-16 flex items-center justify-between">

          <div className="flex items-center gap-3">
            <div className="w-11 h-11 rounded-xl bg-emerald-500/10 border border-emerald-500/40 grid place-items-center text-lg select-none">
              ♻
            </div>
            <div>
              <h1 className="text-md font-bold leading-none" style={{ fontFamily: "'Syne', sans-serif" }}>WasteAI</h1>
              <p className="text-[13px] text-zinc-500 mt-0.5">Intelligent Waste Classifier</p>
            </div>
          </div>

          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border transition-all duration-500 ${
            modelLoading
              ? "border-zinc-700 text-zinc-400 bg-zinc-900"
              : "border-emerald-600/50 text-emerald-400 bg-emerald-500/10"
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full animate-pulse ${modelLoading ? "bg-zinc-500" : "bg-emerald-400"}`} />
            {modelLoading ? "Loading model…" : "Model ready"}
          </div>

        </div>
      </header>

      {/* ── Main ── */}
      <main className="flex-1 px-6 py-10">
        <div className="max-w-5xl mx-auto">

          {/* Title */}
          <div className="mb-10 text-center">
            <h2 className="text-3xl font-extrabold tracking-tight mb-2" style={{ fontFamily: "'Syne', sans-serif" }}>
              What kind of waste is this?
            </h2>
            <p className="text-zinc-500 text-sm max-w-md mx-auto">
              Upload a photo and let MobileNetV2 classify it into one of 6 categories
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

            {/* ── Left: Upload Panel ── */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-6 flex flex-col gap-4">

              <div>
                <h3 className="font-bold text-xs tracking-widest text-zinc-400 uppercase" style={{ fontFamily: "'Syne', sans-serif" }}>
                  Upload Image
                </h3>
                <p className="text-xs text-zinc-600 mt-0.5">Drag & drop or click to browse</p>
              </div>

              {/* Dropzone or Preview */}
              {!image ? (
                <div
                  onClick={() => !modelLoading && fileInputRef.current?.click()}
                  onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={handleDrop}
                  className={`flex-1 min-h-56 rounded-xl border-2 border-dashed flex flex-col items-center justify-center gap-3 transition-all duration-200 select-none ${
                    dragOver
                      ? "border-emerald-500 bg-emerald-500/5 cursor-copy"
                      : modelLoading
                      ? "border-zinc-800 opacity-40 cursor-not-allowed"
                      : "border-zinc-700 hover:border-zinc-500 hover:bg-zinc-800/40 cursor-pointer"
                  }`}
                >
                  <div className="w-12 h-12 rounded-xl bg-zinc-800 border border-zinc-700 grid place-items-center text-xl">
                    ↑
                  </div>
                  <div className="text-center">
                    <p className="text-sm font-medium text-zinc-300">Drop image here</p>
                    <p className="text-xs text-zinc-600 mt-1">JPG · PNG · WEBP</p>
                  </div>
                </div>
              ) : (
                <div className="relative rounded-xl overflow-hidden bg-zinc-800 flex-1 min-h-56 flex items-center justify-center">
                  <img
                    ref={imgRef}
                    src={image}
                    alt="uploaded waste"
                    crossOrigin="anonymous"
                    className="w-full max-h-64 object-contain"
                  />
                  <button
                    onClick={reset}
                    className="absolute top-2 right-2 bg-black/60 hover:bg-red-600/80 text-white text-xs px-3 py-1.5 rounded-full backdrop-blur-sm transition-colors"
                  >
                    ✕ Remove
                  </button>
                </div>
              )}

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => handleFile(e.target.files[0])}
              />

              {/* CTA */}
              {image && !results && (
                <button
                  onClick={predict}
                  disabled={predicting || modelLoading}
                  className="w-full py-3.5 rounded-xl bg-emerald-500 hover:bg-emerald-600 active:scale-[0.98] disabled:opacity-40 disabled:cursor-not-allowed text-zinc-950 font-bold text-sm tracking-wide transition-all hover:scale-102 flex items-center justify-center gap-2"
                  style={{ fontFamily: "'Syne', sans-serif" }}
                >
                  {predicting ? (
                    <>
                      <svg className="animate-spin w-4 h-4 shrink-0" viewBox="0 0 24 24" fill="none">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                      </svg>
                      Analyzing…
                    </>
                  ) : (
                    "Classify Waste →"
                  )}
                </button>
              )}

              {results && (
                <button
                  onClick={reset}
                  className="w-full py-3.5 rounded-xl bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 hover:border-zinc-500 text-zinc-300 font-bold text-sm tracking-wide transition-all active:scale-[0.98]"
                  style={{ fontFamily: "'Syne', sans-serif" }}
                >
                  Try Another Image
                </button>
              )}
            </div>

            {/* ── Right: Results Panel ── */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-6 flex flex-col gap-4">

              <div>
                <h3 className="font-bold text-xs tracking-widest text-zinc-400 uppercase" style={{ fontFamily: "'Syne', sans-serif" }}>
                  Results
                </h3>
                <p className="text-xs text-zinc-600 mt-0.5">Classification confidence breakdown</p>
              </div>

              {/* Empty */}
              {!results && !predicting && (
                <div className="flex-1 flex flex-col items-center justify-center gap-3 text-zinc-600 min-h-56">
                  <span className="text-5xl">🔍</span>
                  <p className="text-sm">Upload an image to see results</p>
                </div>
              )}

              {/* Scanning */}
              {predicting && (
                <div className="flex-1 flex flex-col items-center justify-center gap-4 min-h-56">
                  <div className="w-16 h-16 rounded-xl border border-emerald-500/40 bg-emerald-500/5 relative overflow-hidden">
                    <div className="absolute inset-x-0 h-0.5 bg-emerald-400 animate-scan" style={{ boxShadow: "0 0 8px #34d399" }} />
                  </div>
                  <p className="text-sm text-zinc-500 animate-pulse">Scanning image…</p>
                </div>
              )}

              {/* Results */}
              {results && top && topCfg && (
                <div className="flex flex-col gap-5 animate-fadein">

                  {/* Top Result Card */}
                  <div className={`${topCfg.bg} ${topCfg.border} border rounded-xl p-4 flex items-center gap-4`}>
                    <span className="text-4xl shrink-0">{topCfg.emoji}</span>
                    <div className="flex-1 min-w-0">
                      <p className={`text-xl font-extrabold uppercase tracking-widest leading-none ${topCfg.text}`} style={{ fontFamily: "'Syne', sans-serif" }}>
                        {top.label}
                      </p>
                      <p className="text-xs text-zinc-500 mt-1">{(top.prob * 100).toFixed(1)}% confidence</p>
                      <p className="text-xs text-zinc-500 mt-2">💡 {topCfg.tip}</p>
                    </div>
                    <div className={`text-2xl font-black shrink-0 ${topCfg.text}`} style={{ fontFamily: "'Syne', sans-serif" }}>
                      {(top.prob * 100).toFixed(0)}%
                    </div>
                  </div>

                  {/* All Bars */}
                  <div className="flex flex-col gap-3">
                    {results.map(({ label, prob }, i) => {
                      const cfg = CLASS_CONFIG[label];
                      return (
                        <div key={label} className="grid items-center gap-2" style={{ gridTemplateColumns: "20px 72px 1fr 40px" }}>
                          <span className="text-sm text-center">{cfg.emoji}</span>
                          <span className="text-xs text-zinc-400 capitalize truncate">{label}</span>
                          <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-700 ${cfg.bar}`}
                              style={{
                                width: revealed ? `${(prob * 100).toFixed(1)}%` : "0%",
                                transitionDelay: `${i * 70}ms`,
                                minWidth: prob > 0.001 ? "3px" : "0px",
                              }}
                            />
                          </div>
                          <span className="text-xs text-zinc-500 text-right tabular-nums">{(prob * 100).toFixed(1)}%</span>
                        </div>
                      );
                    })}
                  </div>

                </div>
              )}
            </div>
          </div>

          {/* Footer note */}
          <p className="text-center text-xs text-zinc-700 mt-8">
            Powered by MobileNetV2 · Trained on TrashNet Dataset
          </p>
        </div>
      </main>
    </div>
  );
}