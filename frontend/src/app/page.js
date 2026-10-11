"use client";

import React, { useState } from 'react';
import Header from '../components/Header';
import MetricCard from '../components/MetricCard';
import CameraView from '../components/CameraView';
import LogsPanel from '../components/LogsPanel';
import { Users, UserCheck, Eye, Activity } from 'lucide-react';

export default function FaceAttendanceUI() {
  const [activeTab, setActiveTab] = useState('kiosk');

  const recentLogs = [
    { id: 1, uid: '1', code: 'EMP101', name: 'Harish Menariya', time: '09:02:14 AM', status: 'Success', ear: '0.28' },
    { id: 2, uid: '4', code: 'emp1', name: 'Dev Sharma', time: '09:05:40 AM', status: 'Success', ear: '0.26' },
    { id: 3, uid: '5', code: 'emp2', name: 'Rahul Verma', time: '09:12:02 AM', status: 'Success', ear: '0.31' },
  ];

  return (
    <div className="min-h-screen bg-[#070B14] text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30">
      
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="flex-1 max-w-7xl w-full mx-auto p-6 md:p-8 flex flex-col gap-6">
        
        {/* KPI Header Bar */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <MetricCard 
            title="Total Registered" 
            value="5 Users" 
            icon={Users} 
            colorClass="text-cyan-400" 
            bgClass="bg-cyan-500/10" 
            borderClass="border-cyan-500/20" 
          />
          <MetricCard 
            title="Today's Check-ins" 
            value="14 Verified" 
            icon={UserCheck} 
            colorClass="text-emerald-400" 
            bgClass="bg-emerald-500/10" 
            borderClass="border-emerald-500/20" 
          />
          <MetricCard 
            title="Anti-Spoof Status" 
            value="EAR Active" 
            icon={Eye} 
            colorClass="text-amber-400" 
            bgClass="bg-amber-500/10" 
            borderClass="border-amber-500/20" 
          />
          <MetricCard 
            title="Index Match Speed" 
            value="0.42 ms" 
            icon={Activity} 
            colorClass="text-indigo-400" 
            bgClass="bg-indigo-500/10" 
            borderClass="border-indigo-500/20" 
          />
        </div>

        {/* Core Layout Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1">
          <CameraView />
          <LogsPanel recentLogs={recentLogs} setActiveTab={setActiveTab} />
        </div>

      </main>
    </div>
  );
}
