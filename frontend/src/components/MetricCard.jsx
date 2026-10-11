import React from 'react';

export default function MetricCard({ title, value, icon: Icon, colorClass, bgClass, borderClass }) {
  return (
    <div className="bg-[#0F172A]/70 border border-slate-800/80 rounded-2xl p-4 flex items-center justify-between">
      <div>
        <p className="text-xs text-slate-400 font-medium">{title}</p>
        <h3 className={`text-2xl font-bold mt-1 tracking-tight ${colorClass === 'text-emerald-400' ? 'text-emerald-400' : 'text-white'}`}>
          {value}
        </h3>
      </div>
      <div className={`p-3 rounded-xl ${bgClass} ${borderClass} ${colorClass}`}>
        <Icon className="w-5 h-5" />
      </div>
    </div>
  );
}
