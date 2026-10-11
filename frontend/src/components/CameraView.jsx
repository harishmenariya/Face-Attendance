import React from 'react';

export default function CameraView() {
  return (
    <section className="lg:col-span-8 bg-[#0F172A]/50 border border-slate-800 rounded-3xl p-6 flex flex-col justify-between relative overflow-hidden backdrop-blur-sm">
      
      {/* Viewport Header */}
      <div className="flex items-center justify-between mb-4 z-10">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-cyan-400"></span>
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">Biometric Stream (640x480)</span>
        </div>
        <span className="text-xs font-mono text-cyan-400/80">FPS: 29.8 | YuNet ONNX</span>
      </div>

      {/* Video Canvas Box / HUD */}
      <div className="relative w-full aspect-video bg-slate-950/80 rounded-2xl border border-slate-800/80 flex items-center justify-center overflow-hidden group">
        
        {/* Scanline Effect Overlay */}
        <div className="absolute inset-0 bg-[linear-gradient(to_bottom,transparent_50%,rgba(6,182,212,0.05)_51%)] bg-[length:100%_4px] pointer-events-none" />

        {/* Dynamic Target HUD Box */}
        <div className="relative w-64 h-64 border-2 border-cyan-400/40 rounded-2xl flex flex-col justify-between p-3 animate-pulse">
          {/* Corner Markers */}
          <div className="absolute -top-1 -left-1 w-4 h-4 border-t-2 border-l-2 border-cyan-400"></div>
          <div className="absolute -top-1 -right-1 w-4 h-4 border-t-2 border-r-2 border-cyan-400"></div>
          <div className="absolute -bottom-1 -left-1 w-4 h-4 border-b-2 border-l-2 border-cyan-400"></div>
          <div className="absolute -bottom-1 -right-1 w-4 h-4 border-b-2 border-r-2 border-cyan-400"></div>

          <div className="flex justify-between text-[10px] font-mono text-cyan-300 bg-cyan-950/80 px-2 py-0.5 rounded border border-cyan-500/30">
            <span>CONF: 99.1%</span>
            <span>L2: 0.18</span>
          </div>

          <div className="text-center">
            <span className="text-xs font-medium px-3 py-1 bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 rounded-full backdrop-blur-md">
              Harish Menariya (EMP101)
            </span>
          </div>
        </div>

      </div>

      {/* Bottom Controls */}
      <div className="mt-5 flex items-center justify-between z-10">
        <span className="text-xs text-slate-400">Auto-attendance activates immediately upon genuine verification</span>
        <button className="px-6 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-bold transition-all shadow-lg shadow-cyan-500/20 active:scale-95">
          Simulate Punch
        </button>
      </div>
    </section>
  );
}
