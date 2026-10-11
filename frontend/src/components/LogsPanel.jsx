import React from 'react';
import { Clock } from 'lucide-react';

export default function LogsPanel({ recentLogs, setActiveTab }) {
  return (
    <section className="lg:col-span-4 bg-[#0F172A]/50 border border-slate-800 rounded-3xl p-6 flex flex-col backdrop-blur-sm">
      <div className="flex items-center justify-between pb-4 border-b border-slate-800">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <Clock className="w-4 h-4 text-cyan-400" /> Recent Verifications
        </h3>
        <span className="text-[11px] font-mono text-slate-400">Realtime</span>
      </div>

      <div className="mt-4 flex-1 flex flex-col gap-3 overflow-y-auto">
        {recentLogs.map((log) => (
          <div key={log.id} className="p-3.5 rounded-xl bg-[#111927] border border-slate-800/80 flex items-center justify-between hover:border-slate-700 transition">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center font-bold text-xs text-cyan-400">
                {log.name.charAt(0)}
              </div>
              <div>
                <h5 className="text-xs font-semibold text-white">{log.name}</h5>
                <span className="text-[10px] font-mono text-slate-400">ID: {log.code} • UID: #{log.uid}</span>
              </div>
            </div>
            <div className="text-right">
              <span className="text-xs font-mono text-emerald-400 font-semibold block">{log.time}</span>
              <span className="text-[10px] text-slate-500 font-mono">EAR {log.ear}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Quick Registration Footer Card */}
      <div className="mt-4 p-4 rounded-2xl bg-gradient-to-b from-cyan-950/20 to-slate-900 border border-cyan-500/20 text-center">
        <p className="text-xs text-slate-300 font-medium">New User Registration</p>
        <p className="text-[11px] text-slate-500 mt-0.5 mb-3">Enrolls 5-vector temporal samples into FAISS</p>
        <button 
          onClick={() => setActiveTab('admin')}
          className="w-full py-2 bg-slate-800 hover:bg-slate-700 text-cyan-400 border border-cyan-500/30 text-xs font-semibold rounded-xl transition">
          Open Directory Portal
        </button>
      </div>
    </section>
  );
}
