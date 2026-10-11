import React from 'react';
import { Camera, Users } from 'lucide-react';

export default function Header({ activeTab, setActiveTab }) {
  return (
    <header className="h-16 border-b border-slate-800/80 px-8 flex items-center justify-between bg-[#0B111E]/60 backdrop-blur-md sticky top-0 z-50">
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-cyan-600 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
          <Camera className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="text-base font-semibold tracking-wide flex items-center gap-2">
            AURA<span className="text-cyan-400 font-extrabold tracking-widest text-xs px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-500/30">BIOMETRIC</span>
          </h1>
        </div>
      </div>

      {/* Center Pill Switcher */}
      <div className="flex bg-[#111927] p-1 rounded-xl border border-slate-800">
        <button 
          onClick={() => setActiveTab('kiosk')}
          className={`px-5 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 flex items-center gap-2 ${
            activeTab === 'kiosk' ? 'bg-cyan-500 text-slate-950 font-semibold shadow-md shadow-cyan-500/25' : 'text-slate-400 hover:text-white'
          }`}>
          <Camera className="w-3.5 h-3.5" /> Kiosk Scanner
        </button>
        <button 
          onClick={() => setActiveTab('admin')}
          className={`px-5 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 flex items-center gap-2 ${
            activeTab === 'admin' ? 'bg-cyan-500 text-slate-950 font-semibold shadow-md shadow-cyan-500/25' : 'text-slate-400 hover:text-white'
          }`}>
          <Users className="w-3.5 h-3.5" /> Directory & Logs
        </button>
      </div>

      {/* Status indicator */}
      <div className="flex items-center gap-2 text-xs font-medium px-3 py-1.5 rounded-full bg-emerald-950/40 border border-emerald-500/30 text-emerald-400">
        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
        Neural Engine Live
      </div>
    </header>
  );
}
