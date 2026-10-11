"use client";

import React, { useState, useEffect, useRef } from 'react';
import Webcam from 'react-webcam';
import {
  Camera,
  Users,
  Activity,
  Eye,
  UserCheck,
  ShieldCheck,
  Clock,
  Search,
  UserPlus,
  Trash2,
  Edit3,
  Download,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Sparkles
} from 'lucide-react';

export default function Home() {
  const [activeTab, setActiveTab] = useState<'kiosk' | 'directory'>('kiosk');
  const [isScanning, setIsScanning] = useState<boolean>(true);
  const [verifiedToast, setVerifiedToast] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [isCameraOn, setIsCameraOn] = useState<boolean>(true);
  const [scanResult, setScanResult] = useState<any>(null);
  const [hudStatus, setHudStatus] = useState<string>("Ready to Scan");
  const [hudType, setHudType] = useState<'neutral' | 'success' | 'error' | 'warning'>('neutral');

  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [isCheckingAuth, setIsCheckingAuth] = useState<boolean>(true);
  const [loginIdentifier, setLoginIdentifier] = useState<string>('');
  const [loginPassword, setLoginPassword] = useState<string>('');
  const [loginLoading, setLoginLoading] = useState<boolean>(false);
  const [loginError, setLoginError] = useState<string>('');

  const [enrollStatus, setEnrollStatus] = useState<string>('');
  const [isEnrolling, setIsEnrolling] = useState<boolean>(false);

  useEffect(() => {
    const token = localStorage.getItem("auraface_token");
    if (token) {
      setIsAuthenticated(true);
    }
    setIsCheckingAuth(false);
  }, []);

  const [reportTab, setReportTab] = useState<'directory' | 'daily' | 'monthly'>('directory');
  const [dailyDate, setDailyDate] = useState<string>(new Date().toISOString().split('T')[0]);
  const [monthlyMonth, setMonthlyMonth] = useState<string>(new Date().toISOString().slice(0, 7));
  const [dailyReport, setDailyReport] = useState<any>(null);
  const [monthlyReport, setMonthlyReport] = useState<any>(null);

  const [editingUser, setEditingUser] = useState<any>(null);
  const [editCode, setEditCode] = useState('');
  const [editName, setEditName] = useState('');

  const [individualReportUser, setIndividualReportUser] = useState<any>(null);
  const [individualReport, setIndividualReport] = useState<any>(null);
  const [individualReportMonth, setIndividualReportMonth] = useState<string>(new Date().toISOString().slice(0, 7));

  // Registered database from backend
  const [users, setUsers] = useState<any[]>([]);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/users')
      .then(res => res.json())
      .then(data => setUsers(data))
      .catch(err => setUsers([]));
  }, []);

  useEffect(() => {
    if (activeTab === 'directory' && reportTab === 'daily') {
      fetch(`http://127.0.0.1:8000/api/reports/daily?date_str=${dailyDate}`)
        .then(res => res.json())
        .then(data => setDailyReport(data))
        .catch(err => console.error(err));
    }
  }, [activeTab, reportTab, dailyDate]);

  useEffect(() => {
    if (activeTab === 'directory' && reportTab === 'monthly') {
      fetch(`http://127.0.0.1:8000/api/reports/monthly?month_str=${monthlyMonth}`)
        .then(res => res.json())
        .then(data => setMonthlyReport(data))
        .catch(err => console.error(err));
    }
  }, [activeTab, reportTab, monthlyMonth]);

  useEffect(() => {
    if (individualReportUser) {
      fetch(`http://127.0.0.1:8000/api/reports/user-monthly?user_id=${individualReportUser.uid}&month_str=${individualReportMonth}`)
        .then(res => res.json())
        .then(data => setIndividualReport(data))
        .catch(err => console.error(err));
    }
  }, [individualReportUser, individualReportMonth]);

  // Attendance punch feed
  const [logs, setLogs] = useState<any[]>([]);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/attendance-logs')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) {
          setLogs(data);
        }
      })
      .catch(err => console.error("Failed to load today's check-ins:", err));
  }, []);

  const presentToday = new Set(logs.map(log => log.code)).size;
  const absentToday = Math.max(0, users.length - presentToday);
  const attendanceRate = users.length > 0 ? Math.round((presentToday / users.length) * 100) : 0;

  const webcamRef = useRef<Webcam>(null);
  const enrollWebcamRef = useRef<Webcam>(null);
  const [enrollCode, setEnrollCode] = useState('');
  const [enrollName, setEnrollName] = useState('');

  const [flashType, setFlashType] = useState<'none' | 'success' | 'error'>('none');

  const playSound = (type: 'scan' | 'success' | 'error') => {
    const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioContext) return;
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.connect(gain);
    gain.connect(ctx.destination);

    if (type === 'scan') {
      osc.type = 'sine';
      osc.frequency.setValueAtTime(800, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(1200, ctx.currentTime + 0.08);
      gain.gain.setValueAtTime(0.1, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.08);
      osc.start();
      osc.stop(ctx.currentTime + 0.08);
    } else if (type === 'success') {
      const osc2 = ctx.createOscillator();
      osc.type = 'sine';
      osc2.type = 'sine';
      osc.frequency.value = 520;
      osc2.frequency.value = 880;
      osc.connect(gain);
      osc2.connect(gain);
      gain.gain.setValueAtTime(0.1, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
      osc.start();
      osc2.start();
      osc.stop(ctx.currentTime + 0.3);
      osc2.stop(ctx.currentTime + 0.3);
    } else if (type === 'error') {
      osc.type = 'sawtooth';
      osc.frequency.value = 150;
      gain.gain.setValueAtTime(0.2, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.2);
      osc.start();
      osc.stop(ctx.currentTime + 0.2);
    }
  };

  const toggleCamera = () => {
    if (isCameraOn && webcamRef.current && webcamRef.current.video) {
      const stream = webcamRef.current.video.srcObject as MediaStream;
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    }
    setIsCameraOn(!isCameraOn);
  };

  // Live scan trigger
  const handleLiveScan = async () => {
    playSound('scan');
    if (!webcamRef.current) {
      setHudStatus("Camera not ready");
      setHudType('error');
      setTimeout(() => { setHudStatus("Ready to Scan"); setHudType('neutral'); }, 2000);
      return;
    }

    setHudStatus("Analyzing Biometrics...");
    setHudType('neutral');

    const imageSrc = webcamRef.current?.getScreenshot();
    if (!imageSrc) {
      console.warn("Webcam not ready");
      setHudStatus("Camera not ready");
      setHudType('error');
      setTimeout(() => { setHudStatus("Ready to Scan"); setHudType('neutral'); }, 2000);
      return;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/scan-attendance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: imageSrc }),
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Scan Response:", data);

      if (data.status === 'success') {
        setScanResult({
          name: data.name,
          code: data.user_code,
          uid: data.uid,
          distance: data.distance
        });
        playSound('success');
        setFlashType('success');
        setTimeout(() => setFlashType('none'), 300);
        setVerifiedToast(true);
        setHudType('success');
        setHudStatus(`Verified: ${data.name}`);
        const newLog = {
          id: Date.now(),
          uid: data.uid,
          code: data.user_code,
          name: data.name,
          time: new Date().toLocaleTimeString(),
          status: 'Verified'
        };
        setLogs(prev => [newLog, ...prev]);
        setTimeout(() => {
          setVerifiedToast(false);
          setScanResult(null);
          setHudType('neutral');
          setHudStatus("Ready to Scan");
        }, 2000);
      } else if (data.status === 'already_marked') {
        setScanResult(null);
        setHudStatus(`Already Marked Today — ${data.name}`);
        setHudType('warning');
        setTimeout(() => { setHudStatus("Ready to Scan"); setHudType('neutral'); }, 2000);
      } else if (data.status === 'unknown') {
        setScanResult(null);
        playSound('error');
        setFlashType('error');
        setTimeout(() => setFlashType('none'), 300);
        setHudStatus("Access Denied — Unknown Face");
        setHudType('error');
        setTimeout(() => { setHudStatus("Ready to Scan"); setHudType('neutral'); }, 2000);
      } else if (data.status === 'no_face') {
        setScanResult(null);
        setHudStatus("Align Face in Frame");
        setHudType('warning');
        setTimeout(() => { setHudStatus("Ready to Scan"); setHudType('neutral'); }, 2000);
      } else {
        setScanResult(null);
        setHudStatus(data.message || 'Face not recognized');
        setHudType('error');
        setTimeout(() => { setHudStatus("Ready to Scan"); setHudType('neutral'); }, 2000);
      }
    } catch (error) {
      console.error("Fetch Scan Error:", error);
      setScanResult(null);
      setHudStatus("Scan Error / Retry");
      setHudType('error');
      setTimeout(() => { setHudStatus("Ready to Scan"); setHudType('neutral'); }, 2000);
    }
  };

  const handleEnroll = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!enrollWebcamRef.current) return;
    if (!enrollCode || !enrollName) {
      alert("Please provide Code, Name, and ensure camera is active.");
      return;
    }

    setIsEnrolling(true);
    setEnrollStatus('Capturing 5 face samples... Please look at the camera.');

    const images: string[] = [];
    for (let i = 0; i < 5; i++) {
      const src = enrollWebcamRef.current.getScreenshot();
      if (src) images.push(src);
      setEnrollStatus(`Capturing sample ${i + 1}/5...`);
      await new Promise(r => setTimeout(r, 300));
    }

    setEnrollStatus('Processing biometrics...');

    try {
      const response = await fetch('http://127.0.0.1:8000/api/enroll', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_code: enrollCode,
          name: enrollName,
          images_base64: images
        }),
      });
      const data = await response.json();

      if (response.ok && data.status === 'success') {
        alert(`Successfully enrolled ${data.name}!`);
        setEnrollCode('');
        setEnrollName('');
        setEnrollStatus('');
        // Refresh users
        fetch('http://127.0.0.1:8000/api/users')
          .then(res => res.json())
          .then(resData => setUsers(resData));
      } else {
        alert(`Enrollment failed: ${data.detail || data.message || 'Unknown error'}`);
        setEnrollStatus('Enrollment failed.');
      }
    } catch (error) {
      console.error(error);
      alert('Error connecting to backend');
      setEnrollStatus('');
    }
    setIsEnrolling(false);
  };

  const handleAdminLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoginLoading(true);
    setLoginError('');
    try {
      const res = await fetch('http://127.0.0.1:8000/api/admin/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: loginIdentifier, password: loginPassword })
      });
      if (res.ok) {
        localStorage.setItem("auraface_token", "active");
        setIsAuthenticated(true);
        setActiveTab('kiosk');
      } else {
        setLoginError("Invalid Admin Credentials");
      }
    } catch {
      setLoginError("Error connecting to backend");
    } finally {
      setLoginLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("auraface_token");
    setIsAuthenticated(false);
  };

  const filteredUsers = users.filter(u =>
    u.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    u.code.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleEditUser = (user: any) => {
    setEditingUser(user);
    setEditCode(user.code);
    setEditName(user.name);
  };

  const handleSaveEdit = async () => {
    if (!editingUser) return;
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/users/${editingUser.uid}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_code: editCode, name: editName })
      });
      if (res.ok) {
        setEditingUser(null);
        fetch('http://127.0.0.1:8000/api/users')
          .then(res => res.json())
          .then(data => setUsers(data));
      } else {
        const err = await res.json();
        alert(err.detail || "Error updating user");
      }
    } catch (e) {
      alert("Error updating user");
    }
  };

  const handleDeleteUser = async (user: any) => {
    if (!confirm(`Are you sure you want to delete ${user.name} and ALL their attendance records? This cannot be undone.`)) return;
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/users/${user.uid}`, {
        method: 'DELETE'
      });
      if (res.ok) {
        fetch('http://127.0.0.1:8000/api/users')
          .then(res => res.json())
          .then(data => setUsers(data));
      } else {
        const err = await res.json();
        alert(err.detail || "Error deleting user");
      }
    } catch (e) {
      alert("Error deleting user");
    }
  };

  if (isCheckingAuth) {
    return <div className="min-h-screen bg-[#070b13] flex items-center justify-center text-cyan-400 font-mono">Verifying Biometric Security Clearance...</div>;
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-[#070b13] flex items-center justify-center p-4">
        <div className="w-full max-w-md bg-slate-900/90 border border-cyan-500/40 rounded-2xl p-8 shadow-[0_0_30px_rgba(6,182,212,0.15)]">
          <div className="text-center mb-6">
            <h1 className="text-2xl font-bold tracking-wider text-cyan-400">AURAFACE HUD</h1>
            <p className="text-xs text-slate-500 mt-1 uppercase tracking-widest font-mono">Restricted Access // Admin Authorization</p>
          </div>

          <form onSubmit={handleAdminLogin} className="space-y-4">
            {loginError && (
              <div className="p-2.5 text-xs text-rose-400 bg-rose-950/50 border border-rose-500/40 rounded-lg text-center font-mono">
                ⚠️ {loginError}
              </div>
            )}

            <div>
              <label className="block text-xs uppercase font-mono text-slate-400 mb-1">User ID / Email</label>
              <input
                type="text"
                required
                value={loginIdentifier}
                onChange={(e) => setLoginIdentifier(e.target.value)}
                placeholder="admin or admin@auraface.ai"
                className="w-full px-4 py-2.5 bg-slate-950/80 border border-slate-700 rounded-lg text-sm text-slate-100 focus:outline-none focus:border-cyan-500 font-mono"
              />
            </div>

            <div>
              <label className="block text-xs uppercase font-mono text-slate-400 mb-1">Password</label>
              <input
                type="password"
                required
                value={loginPassword}
                onChange={(e) => setLoginPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full px-4 py-2.5 bg-slate-950/80 border border-slate-700 rounded-lg text-sm text-slate-100 focus:outline-none focus:border-cyan-500 font-mono"
              />
            </div>

            <button
              type="submit"
              disabled={loginLoading}
              className="w-full py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white text-xs font-bold uppercase tracking-wider rounded-lg transition disabled:opacity-50 mt-2"
            >
              {loginLoading ? "Authenticating Clearance..." : "Authorize & Enter System"}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#060913] text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30">
      <style>{`
        @keyframes scanline {
          0% { top: 0%; }
          100% { top: 100%; }
        }
        .animate-scanline {
          animation: scanline 2.5s linear infinite;
        }
        @keyframes pulse-bracket {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        .animate-pulse-bracket {
          animation: pulse-bracket 2s ease-in-out infinite;
        }
      `}</style>
      {/* Glow Backdrops */}
      <div className="fixed top-0 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="fixed bottom-0 right-1/4 w-96 h-96 bg-blue-600/10 rounded-full blur-3xl pointer-events-none" />

      {/* Top Futuristic Navbar */}
      <header className="h-16 border-b border-slate-800/80 px-8 flex items-center justify-between bg-[#0A0F1D]/80 backdrop-blur-xl sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/25 border border-cyan-400/30">
            <Camera className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-base font-bold tracking-wider text-white">AURA<span className="text-cyan-400">FACE</span></span>
              <span className="text-[10px] uppercase font-mono tracking-widest px-2 py-0.5 rounded bg-cyan-950/80 border border-cyan-500/30 text-cyan-300 font-semibold">
                NEURAL HUD
              </span>
            </div>
            <p className="text-[11px] text-slate-400">Enterprise Biometric Attendance Suite</p>
          </div>
        </div>

        {/* View Switcher Pill */}
        <div className="flex bg-[#0F172A] p-1 rounded-xl border border-slate-800 shadow-inner">
          <button
            onClick={() => setActiveTab('kiosk')}
            className={`px-5 py-2 rounded-lg text-xs font-semibold transition-all duration-200 flex items-center gap-2 ${activeTab === 'kiosk'
              ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-slate-950 shadow-md shadow-cyan-500/30'
              : 'text-slate-400 hover:text-white'
              }`}>
            <Camera className="w-4 h-4" /> Kiosk Scanner
          </button>
          <button
            onClick={() => setActiveTab('directory')}
            className={`px-5 py-2 rounded-lg text-xs font-semibold transition-all duration-200 flex items-center gap-2 ${activeTab === 'directory'
              ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-slate-950 shadow-md shadow-cyan-500/30'
              : 'text-slate-400 hover:text-white'
              }`}>
            <Users className="w-4 h-4" /> Directory & Logs
          </button>
        </div>

        {/* Engine Status Badge */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs font-medium px-3.5 py-1.5 rounded-full bg-emerald-950/50 border border-emerald-500/30 text-emerald-400 shadow-sm shadow-emerald-500/10">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
            Core Models Online
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-cyan-950/40 border border-cyan-500/30 text-cyan-400 text-xs font-medium">
            <ShieldCheck className="w-4 h-4" /> Logged in: admin
          </div>
          <button onClick={handleLogout} className="flex items-center gap-1 text-xs px-3 py-1.5 bg-rose-500/10 border border-rose-500/30 text-rose-400 rounded-lg hover:bg-rose-500/20 transition font-bold">
            🔒 Logout
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 md:p-8 flex flex-col gap-6 relative z-10">

        {/* KPI Metrics Strip */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="backdrop-blur-xl bg-slate-900/60 border border-cyan-500/20 shadow-[0_0_30px_rgba(6,182,212,0.08)] rounded-2xl p-4 flex items-center justify-between hover:border-cyan-500/40 transition">
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wider font-medium">Registered Total</p>
              <h3 className="text-2xl font-bold mt-1 text-white">{users.length} Users</h3>
              <p className="text-[11px] text-cyan-400/80 mt-0.5">Total Enrolled</p>
            </div>
            <div className="p-3 bg-cyan-500/10 border border-cyan-500/20 rounded-xl text-cyan-400">
              <Users className="w-5 h-5" />
            </div>
          </div>

          <div className="backdrop-blur-xl bg-slate-900/60 border border-cyan-500/20 shadow-[0_0_30px_rgba(6,182,212,0.08)] rounded-2xl p-4 flex items-center justify-between hover:border-emerald-500/40 transition">
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wider font-medium">Present Today</p>
              <h3 className="text-2xl font-bold mt-1 text-emerald-400">{presentToday} Punches</h3>
              <p className="text-[11px] text-emerald-400/80 mt-0.5">Successful Scans</p>
            </div>
            <div className="p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-xl text-emerald-400">
              <UserCheck className="w-5 h-5" />
            </div>
          </div>

          <div className="backdrop-blur-xl bg-slate-900/60 border border-cyan-500/20 shadow-[0_0_30px_rgba(6,182,212,0.08)] rounded-2xl p-4 flex items-center justify-between hover:border-rose-500/40 transition">
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wider font-medium">Absent Today</p>
              <h3 className="text-2xl font-bold mt-1 text-rose-400">{absentToday} Missing</h3>
              <p className="text-[11px] text-rose-400/80 mt-0.5">Yet to arrive</p>
            </div>
            <div className="p-3 bg-rose-500/10 border border-rose-500/20 rounded-xl text-rose-400">
              <AlertTriangle className="w-5 h-5" />
            </div>
          </div>

          <div className="backdrop-blur-xl bg-slate-900/60 border border-cyan-500/20 shadow-[0_0_30px_rgba(6,182,212,0.08)] rounded-2xl p-4 flex items-center justify-between hover:border-indigo-500/40 transition">
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wider font-medium">Attendance Rate %</p>
              <h3 className="text-2xl font-bold mt-1 text-indigo-300">{attendanceRate}%</h3>
              <p className="text-[11px] text-indigo-400/80 mt-0.5">Daily Turnout</p>
            </div>
            <div className="p-3 bg-indigo-500/10 border border-indigo-500/20 rounded-xl text-indigo-400">
              <Activity className="w-5 h-5" />
            </div>
          </div>
        </div>

        {activeTab === 'kiosk' ? (
          /* ========================================================================= */
          /* KIOSK SCANNER VIEW                                                        */
          /* ========================================================================= */
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1">

            {/* Viewport Scanner Card */}
            <div className="lg:col-span-8 backdrop-blur-xl bg-slate-900/60 border border-cyan-500/20 shadow-[0_0_30px_rgba(6,182,212,0.08)] rounded-3xl p-6 flex flex-col justify-between relative overflow-hidden">

              {/* Header inside viewport */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full bg-cyan-400 animate-ping"></span>
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-200">
                    Live Optical Feed • YuNet ONNX
                  </span>
                </div>
                <div className="flex items-center gap-3 font-mono text-xs text-cyan-400/90">
                  <span>RES: 640x480</span>
                  <span>•</span>
                  <span>FPS: 30.0</span>
                </div>
              </div>

              {/* Camera Frame Simulation */}
              <div className={`relative w-full aspect-video bg-slate-950/90 rounded-2xl border border-slate-800 flex items-center justify-center overflow-hidden group shadow-inner transition-colors duration-300 ${flashType === 'success' ? 'bg-emerald-500/10 border-emerald-400 shadow-[0_0_35px_rgba(16,185,129,0.3)]' : flashType === 'error' ? 'bg-rose-500/10 border-rose-500 shadow-[0_0_35px_rgba(244,63,94,0.3)]' : ''}`}>
                {isCameraOn ? (
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    screenshotFormat="image/jpeg"
                    videoConstraints={{ width: 640, height: 480, facingMode: "user" }}
                    className="absolute inset-0 w-full h-full object-cover"
                  />
                ) : (
                  <div className="absolute inset-0 w-full h-full bg-slate-900 flex items-center justify-center text-slate-500 font-medium z-10">
                    Camera Offline
                  </div>
                )}

                {/* Top-Left Badge */}
                <div className="absolute top-4 left-4 z-20 font-mono text-[10px] tracking-wider text-cyan-400/80 bg-slate-950/80 px-2.5 py-1 rounded border border-cyan-500/20 backdrop-blur-sm pointer-events-none">
                  ● OPTICAL 640x480 • 30 FPS • YUNET ONNX
                </div>

                {/* Top-Right Badge */}
                <div className="absolute top-4 right-4 z-20 font-mono text-[10px] tracking-wider text-cyan-400/80 bg-slate-950/80 px-2.5 py-1 rounded border border-cyan-500/20 backdrop-blur-sm pointer-events-none">
                  FAISS INDEX FLAT L2 • L2 THRESHOLD: 0.55
                </div>

                {/* Bottom-Left Badge */}
                <div className="absolute bottom-4 left-4 z-20 font-mono text-[10px] tracking-wider text-cyan-400/80 bg-slate-950/80 px-2.5 py-1 rounded border border-cyan-500/20 backdrop-blur-sm pointer-events-none">
                  MOBILENETV2 • 128-DIM EMBEDDING
                </div>

                {/* Bottom-Right Badge */}
                <div className="absolute bottom-4 right-4 z-20 font-mono text-[10px] tracking-wider text-cyan-400/80 bg-slate-950/80 px-2.5 py-1 rounded border border-cyan-500/20 backdrop-blur-sm pointer-events-none">
                  LATENCY: ~16ms • SQLITE SYNCED
                </div>

                {/* Cyber Scanline Overlay */}
                <div className="absolute inset-0 bg-[linear-gradient(to_bottom,transparent_50%,rgba(6,182,212,0.06)_51%)] bg-[length:100%_4px] pointer-events-none z-10" />

                {/* Vertical Scanning Beam Animation */}
                <div className="absolute w-full h-[2px] bg-cyan-400 shadow-[0_0_12px_3px_rgba(34,211,238,0.6)] animate-scanline pointer-events-none z-20" />

                {/* Biometric Target Bracket HUD */}
                <div className={`relative w-72 h-72 rounded-3xl flex flex-col justify-between p-4 transition-colors duration-300 z-20`}>
                  {/* Glowing Reticle Corners with Pulse */}
                  <div className={`absolute top-0 left-0 w-8 h-8 border-t-[3px] border-l-[3px] animate-pulse-bracket transition-colors duration-300 ${hudType === 'success' ? 'border-emerald-400' : hudType === 'warning' ? 'border-amber-500' : hudType === 'error' ? 'border-rose-500' : 'border-cyan-400'
                    }`} />
                  <div className={`absolute top-0 right-0 w-8 h-8 border-t-[3px] border-r-[3px] animate-pulse-bracket transition-colors duration-300 ${hudType === 'success' ? 'border-emerald-400' : hudType === 'warning' ? 'border-amber-500' : hudType === 'error' ? 'border-rose-500' : 'border-cyan-400'
                    }`} />
                  <div className={`absolute bottom-0 left-0 w-8 h-8 border-b-[3px] border-l-[3px] animate-pulse-bracket transition-colors duration-300 ${hudType === 'success' ? 'border-emerald-400' : hudType === 'warning' ? 'border-amber-500' : hudType === 'error' ? 'border-rose-500' : 'border-cyan-400'
                    }`} />
                  <div className={`absolute bottom-0 right-0 w-8 h-8 border-b-[3px] border-r-[3px] animate-pulse-bracket transition-colors duration-300 ${hudType === 'success' ? 'border-emerald-400' : hudType === 'warning' ? 'border-amber-500' : hudType === 'error' ? 'border-rose-500' : 'border-cyan-400'
                    }`} />

                  {/* Top HUD Tag */}
                  <div className="flex justify-between items-center text-[10px] font-mono text-cyan-300 bg-cyan-950/80 px-2.5 py-1 rounded-md border border-cyan-500/30 backdrop-blur-md">
                    <span>{scanResult ? `MATCH: ${(100 - scanResult.distance * 100).toFixed(1)}%` : "AWAITING FACE"}</span>
                    <span>{scanResult ? `DIST: ${scanResult.distance}` : "PRESS SCAN"}</span>
                  </div>

                  {/* Identified Subject Badge */}
                  <div className="text-center w-full flex justify-center">
                    {scanResult ? (
                      <span className="text-xs font-semibold px-4 py-1.5 bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 rounded-full backdrop-blur-md shadow-lg shadow-emerald-500/10 whitespace-nowrap">
                        {hudStatus}
                      </span>
                    ) : (
                      <span className={`text-xs font-semibold px-4 py-1.5 rounded-full backdrop-blur-md shadow-lg ${hudType === 'error'
                        ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                        : hudType === 'warning'
                          ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                          : 'bg-slate-800/80 text-slate-400 border border-slate-700'
                        }`}>
                        {hudStatus}
                      </span>
                    )}
                  </div>
                </div>

                {/* Bottom Liveness Prompt Bar (Removed) */}

                {/* Toast Notification on Successful Verification */}
                {verifiedToast && scanResult && (
                  <div className="absolute top-6 inset-x-12 bg-emerald-950/90 border border-emerald-500/50 p-4 rounded-xl flex items-center justify-between text-emerald-200 backdrop-blur-2xl shadow-2xl animate-in fade-in slide-in-from-top duration-300 z-30">
                    <div className="flex items-center gap-3">
                      <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                      <div>
                        <h4 className="text-sm font-bold">Attendance Verified!</h4>
                        <p className="text-xs text-emerald-300/80">Welcome, {scanResult.name}. Punch saved to SQLite & logs.</p>
                      </div>
                    </div>
                    <span className="text-xs font-mono bg-emerald-900/60 px-2 py-1 rounded border border-emerald-500/30">
                      UID #{scanResult.uid} • {scanResult.distance} L2
                    </span>
                  </div>
                )}
              </div>

              {/* Viewport Bottom Controls */}
              <div className="mt-5 flex items-center justify-between">
                <button
                  onClick={toggleCamera}
                  className={`px-4 py-2.5 rounded-xl text-xs font-bold transition-all shadow-md ${isCameraOn ? 'bg-slate-800 text-slate-300 hover:bg-slate-700' : 'bg-rose-500/20 text-rose-400 hover:bg-rose-500/30'} border border-slate-700`}>
                  {isCameraOn ? "Camera OFF" : "Camera ON"}
                </button>
                <button
                  onClick={handleLiveScan}
                  className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 text-xs font-bold transition-all shadow-lg shadow-cyan-500/25 active:scale-95">
                  Scan & Mark Attendance
                </button>
              </div>
            </div>

            {/* Right Feed Panel */}
            <div className="lg:col-span-4 bg-[#0D1424]/60 border border-slate-800 rounded-3xl p-6 flex flex-col backdrop-blur-xl shadow-2xl">
              <div className="flex items-center justify-between pb-4 border-b border-slate-800">
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <Clock className="w-4 h-4 text-cyan-400" /> Realtime Check-ins
                </h3>
                <span className="text-[11px] font-mono text-cyan-400 bg-cyan-950/60 border border-cyan-500/30 px-2 py-0.5 rounded">
                  Live Stream
                </span>
              </div>

              {/* Feed List */}
              <div className="mt-4 flex-1 flex flex-col gap-3 overflow-y-auto max-h-[440px] pr-1">
                {logs.map((log) => (
                  <div
                    key={log.id}
                    className="p-3.5 rounded-xl bg-[#0A0F1D]/80 border border-slate-800/80 flex items-center justify-between hover:border-cyan-500/30 transition">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-slate-800 to-slate-700 border border-slate-700 flex items-center justify-center font-bold text-sm text-cyan-400">
                        {log.name.charAt(0)}
                      </div>
                      <div>
                        <h5 className="text-xs font-bold text-white">{log.name}</h5>
                        <p className="text-[10px] font-mono text-slate-400">ID: {log.code} • UID: #{log.uid}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="text-xs font-mono text-emerald-400 font-semibold block">{log.time}</span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Bottom Quick Action Box */}
              <div className="mt-4 p-4 rounded-2xl bg-gradient-to-b from-cyan-950/30 to-slate-900/60 border border-cyan-500/20 text-center">
                <p className="text-xs font-semibold text-slate-200">Face Not Enrolled?</p>
                <p className="text-[11px] text-slate-400 mt-0.5 mb-3">Register new face samples in the management directory.</p>
                <button
                  onClick={() => setActiveTab('directory')}
                  className="w-full py-2.5 bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-cyan-500/30 text-xs font-semibold rounded-xl transition flex items-center justify-center gap-2">
                  <UserPlus className="w-3.5 h-3.5" /> Go to User Directory
                </button>
              </div>
            </div>

          </div>
        ) : (
          /* ========================================================================= */
          /* USER DIRECTORY & MANAGEMENT VIEW                                          */
          /* ========================================================================= */
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1">

            {/* Left: Enrollment Form */}
            <div className="lg:col-span-4 bg-[#0D1424]/60 border border-slate-800 rounded-3xl p-6 backdrop-blur-xl flex flex-col justify-between shadow-2xl">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <UserPlus className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-sm font-bold text-white">Biometric Enrollment</h3>
                </div>
                <p className="text-[11px] text-slate-400 mb-6">Extracts 5-vector temporal samples into SQLite & FAISS</p>

                <form className="space-y-4" onSubmit={handleEnroll}>
                  <div>
                    <label className="text-xs font-medium text-slate-300 block mb-1.5">User / Employee Code</label>
                    <input
                      type="text"
                      value={enrollCode}
                      onChange={(e) => setEnrollCode(e.target.value)}
                      placeholder="e.g. EMP106"
                      className="w-full bg-[#0A0F1D] border border-slate-700/80 rounded-xl px-4 py-2.5 text-xs text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none transition"
                    />
                  </div>

                  <div>
                    <label className="text-xs font-medium text-slate-300 block mb-1.5">Full Name</label>
                    <input
                      type="text"
                      value={enrollName}
                      onChange={(e) => setEnrollName(e.target.value)}
                      placeholder="e.g. Rohini Rathore"
                      className="w-full bg-[#0A0F1D] border border-slate-700/80 rounded-xl px-4 py-2.5 text-xs text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none transition"
                    />
                  </div>

                  <div>
                    <label className="text-xs font-medium text-slate-300 block mb-1.5">Biometric Sensor Viewport</label>
                    <div className="w-full h-40 rounded-xl bg-slate-950 border border-slate-800 flex flex-col items-center justify-center text-slate-500 gap-2 relative overflow-hidden">
                      <Webcam
                        ref={enrollWebcamRef}
                        audio={false}
                        screenshotFormat="image/jpeg"
                        videoConstraints={{ width: 320, height: 240, facingMode: "user" }}
                        className="absolute inset-0 w-full h-full object-cover opacity-80"
                      />
                      <div className="absolute inset-x-0 bottom-0 h-1 bg-cyan-500/30" />
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={isEnrolling}
                    className="w-full py-3 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-slate-950 font-bold text-xs shadow-lg shadow-emerald-500/20 transition active:scale-95 disabled:opacity-50">
                    {isEnrolling ? 'Enrolling...' : 'Enroll Biometrics (Strict 0.15 Gated)'}
                  </button>
                  {enrollStatus && <p className="text-xs text-emerald-400 text-center mt-2">{enrollStatus}</p>}
                </form>
              </div>

              <div className="mt-6 pt-4 border-t border-slate-800/80 text-[11px] text-slate-400 flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
                Duplicate prevention guarantees zero duplicate registrations.
              </div>
            </div>

            {/* Right: Registered Database Table */}
            <div className="lg:col-span-8 bg-[#0D1424]/60 border border-slate-800 rounded-3xl p-6 backdrop-blur-xl flex flex-col shadow-2xl">

              {/* Table Toolbar */}
              <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pb-5 border-b border-slate-800">
                <div>
                  <div className="flex items-center gap-4 mb-2">
                    <button onClick={() => setReportTab('directory')} className={`text-sm font-bold pb-1 transition-colors ${reportTab === 'directory' ? 'text-white border-b-2 border-cyan-500' : 'text-slate-400 hover:text-white'}`}>Directory</button>
                    <button onClick={() => setReportTab('daily')} className={`text-sm font-bold pb-1 transition-colors ${reportTab === 'daily' ? 'text-white border-b-2 border-cyan-500' : 'text-slate-400 hover:text-white'}`}>Daily Report</button>
                    <button onClick={() => setReportTab('monthly')} className={`text-sm font-bold pb-1 transition-colors ${reportTab === 'monthly' ? 'text-white border-b-2 border-cyan-500' : 'text-slate-400 hover:text-white'}`}>Monthly Report</button>
                  </div>
                  <p className="text-[11px] text-slate-400">
                    {reportTab === 'directory' ? `Total ${users.length} active identities indexed in FAISS` :
                      reportTab === 'daily' ? `Daily attendance insights` : `Monthly summary insights`}
                  </p>
                </div>

                <div className="flex items-center gap-3 w-full sm:w-auto">
                  {reportTab === 'directory' && (
                    <>
                      <div className="relative flex-1 sm:w-60">
                        <Search className="w-3.5 h-3.5 absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" />
                        <input
                          type="text"
                          placeholder="Search code or name..."
                          value={searchQuery}
                          onChange={(e) => setSearchQuery(e.target.value)}
                          className="w-full bg-[#0A0F1D] border border-slate-700/80 rounded-xl pl-9 pr-3 py-2 text-xs text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none transition"
                        />
                      </div>
                      <button
                        onClick={() => window.open('http://127.0.0.1:8000/api/export/today', '_blank')}
                        className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition">
                        <Download className="w-3.5 h-3.5" /> Export Today
                      </button>
                      <button
                        onClick={() => window.open('http://127.0.0.1:8000/api/export/summary', '_blank')}
                        className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition">
                        <Download className="w-3.5 h-3.5" /> Summary
                      </button>
                    </>
                  )}
                  {reportTab === 'daily' && (
                    <>
                      <input
                        type="date"
                        value={dailyDate}
                        onChange={(e) => setDailyDate(e.target.value)}
                        className="bg-[#0A0F1D] border border-slate-700/80 rounded-xl px-3 py-2 text-xs text-white focus:border-cyan-500 focus:outline-none transition"
                      />
                      <button
                        onClick={() => {
                          const csvRows = [["User Code", "Name", "Date", "Punch Time", "Status"]];
                          dailyReport?.records?.forEach((r: any) => csvRows.push([r.user_code, r.name, r.date, r.punch_time, r.status]));
                          const csvString = csvRows.map((r: any) => r.join(",")).join("\n");
                          const blob = new Blob([csvString], { type: 'text/csv' });
                          const url = window.URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = `Daily_Report_${dailyDate}.csv`;
                          a.click();
                        }}
                        className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition">
                        <Download className="w-3.5 h-3.5" /> Export CSV
                      </button>
                    </>
                  )}
                  {reportTab === 'monthly' && (
                    <>
                      <input
                        type="month"
                        value={monthlyMonth}
                        onChange={(e) => setMonthlyMonth(e.target.value)}
                        className="bg-[#0A0F1D] border border-slate-700/80 rounded-xl px-3 py-2 text-xs text-white focus:border-cyan-500 focus:outline-none transition"
                      />
                      <button
                        onClick={() => {
                          const csvRows = [["User Code", "Name", "Month", "Total Days", "Days Present", "Days Absent", "Attendance %"]];
                          monthlyReport?.records?.forEach((r: any) => csvRows.push([r.user_code, r.name, r.month, r.total_days, r.days_present, r.days_absent, r.attendance_percentage]));
                          const csvString = csvRows.map((r: any) => r.join(",")).join("\n");
                          const blob = new Blob([csvString], { type: 'text/csv' });
                          const url = window.URL.createObjectURL(blob);
                          const a = document.createElement('a');
                          a.href = url;
                          a.download = `Monthly_Report_${monthlyMonth}.csv`;
                          a.click();
                        }}
                        className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition">
                        <Download className="w-3.5 h-3.5" /> Export CSV
                      </button>
                    </>
                  )}
                </div>
              </div>

              {/* Data Table */}
              <div className="mt-4 flex-1 overflow-x-auto">
                {reportTab === 'directory' && (
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="border-b border-slate-800 text-[11px] font-mono uppercase text-slate-400">
                        <th className="py-3 px-4">S.No</th>
                        <th className="py-3 px-4">UID</th>
                        <th className="py-3 px-4">User Code</th>
                        <th className="py-3 px-4">Full Name</th>
                        <th className="py-3 px-4">Reg Date</th>
                        <th className="py-3 px-4 text-right">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 text-xs">
                      {filteredUsers.map((user, idx) => (
                        <tr key={user.uid} className="hover:bg-slate-800/30 transition group">
                          <td className="py-3.5 px-4 font-mono text-slate-400">{idx + 1}</td>
                          <td className="py-3.5 px-4 font-mono font-bold text-cyan-400">#{user.uid}</td>
                          <td className="py-3.5 px-4 font-mono text-slate-300">{user.code}</td>
                          <td className="py-3.5 px-4 font-medium text-white">{user.name}</td>
                          <td className="py-3.5 px-4 text-slate-400 font-mono text-[11px]">{user.date}</td>
                          <td className="py-3.5 px-4 text-right">
                            <div className="flex items-center justify-end gap-2">
                              <button onClick={() => setIndividualReportUser(user)} className="p-1.5 hover:bg-slate-700 text-slate-400 hover:text-cyan-400 rounded-lg transition" title="View Report">
                                <Activity className="w-3.5 h-3.5" />
                              </button>
                              <button onClick={() => handleEditUser(user)} className="p-1.5 hover:bg-slate-700 text-slate-400 hover:text-cyan-400 rounded-lg transition" title="Edit Record">
                                <Edit3 className="w-3.5 h-3.5" />
                              </button>
                              <button onClick={() => handleDeleteUser(user)} className="p-1.5 hover:bg-rose-500/20 text-slate-400 hover:text-rose-400 rounded-lg transition" title="Delete User">
                                <Trash2 className="w-3.5 h-3.5" />
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}

                {reportTab === 'daily' && dailyReport && (
                  <div>
                    <div className="grid grid-cols-4 gap-4 mb-4">
                      <div className="bg-[#0A0F1D] p-3 rounded-xl border border-slate-800/80 text-center">
                        <div className="text-[10px] text-slate-400 uppercase">Total Users</div>
                        <div className="text-lg font-bold text-white">{dailyReport.total_users}</div>
                      </div>
                      <div className="bg-emerald-900/20 p-3 rounded-xl border border-emerald-500/20 text-center">
                        <div className="text-[10px] text-emerald-400 uppercase">Present</div>
                        <div className="text-lg font-bold text-emerald-400">{dailyReport.present_count}</div>
                      </div>
                      <div className="bg-rose-900/20 p-3 rounded-xl border border-rose-500/20 text-center">
                        <div className="text-[10px] text-rose-400 uppercase">Absent</div>
                        <div className="text-lg font-bold text-rose-400">{dailyReport.absent_count}</div>
                      </div>
                      <div className="bg-cyan-900/20 p-3 rounded-xl border border-cyan-500/20 text-center">
                        <div className="text-[10px] text-cyan-400 uppercase">Attendance Rate</div>
                        <div className="text-lg font-bold text-cyan-400">{dailyReport.attendance_rate}%</div>
                      </div>
                    </div>
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="border-b border-slate-800 text-[11px] font-mono uppercase text-slate-400">
                          <th className="py-3 px-4">User Code</th>
                          <th className="py-3 px-4">Name</th>
                          <th className="py-3 px-4">Date</th>
                          <th className="py-3 px-4">Check-in Time</th>
                          <th className="py-3 px-4">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 text-xs">
                        {dailyReport.records.map((r: any) => (
                          <tr key={r.uid} className="hover:bg-slate-800/30 transition">
                            <td className="py-3 px-4 font-mono text-slate-300">{r.user_code}</td>
                            <td className="py-3 px-4 font-medium text-white">{r.name}</td>
                            <td className="py-3 px-4 text-slate-400">{r.date}</td>
                            <td className="py-3 px-4 font-mono text-cyan-400">{r.punch_time}</td>
                            <td className="py-3 px-4">
                              {r.status === 'Present' ? (
                                <span className="px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded-md text-[10px] font-bold border border-emerald-500/30">PRESENT</span>
                              ) : (
                                <span className="px-2 py-1 bg-rose-500/20 text-rose-400 rounded-md text-[10px] font-bold border border-rose-500/30">ABSENT</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {reportTab === 'monthly' && monthlyReport && (
                  <div>
                    <div className="mb-4 bg-[#0A0F1D] p-3 rounded-xl border border-slate-800/80 text-center inline-block">
                      <div className="text-[10px] text-slate-400 uppercase px-4">Working Days</div>
                      <div className="text-lg font-bold text-white px-4">{monthlyReport.working_days}</div>
                    </div>
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="border-b border-slate-800 text-[11px] font-mono uppercase text-slate-400">
                          <th className="py-3 px-4">User Code</th>
                          <th className="py-3 px-4">Name</th>
                          <th className="py-3 px-4 text-center">Total Days</th>
                          <th className="py-3 px-4 text-center">Days Present</th>
                          <th className="py-3 px-4 text-center">Days Absent</th>
                          <th className="py-3 px-4 text-right">Attendance Rate</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 text-xs">
                        {monthlyReport.records.map((r: any) => (
                          <tr key={r.uid} className="hover:bg-slate-800/30 transition">
                            <td className="py-3 px-4 font-mono text-slate-300">{r.user_code}</td>
                            <td className="py-3 px-4 font-medium text-white">{r.name}</td>
                            <td className="py-3 px-4 text-center text-slate-400 font-mono">{r.total_days}</td>
                            <td className="py-3 px-4 text-center text-emerald-400 font-mono">{r.days_present}</td>
                            <td className="py-3 px-4 text-center text-rose-400 font-mono">{r.days_absent}</td>
                            <td className="py-3 px-4 text-right text-cyan-400 font-bold">{r.attendance_percentage}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              {/* Sequential Integrity Notice */}
              <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between text-[11px] text-slate-400">
                <span>Vector Dimension: 128-d Unit Normal L2</span>
                <span className="font-mono text-cyan-400">FAISS Status: Synchronized</span>
              </div>
            </div>

          </div>
        )}

      </main>

      {/* Admin Login Modal */}
      {!isAuthenticated && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[100]">
          <div className="bg-[#0D1424] border border-slate-700 p-8 rounded-2xl w-full max-w-md shadow-2xl relative">
            <h3 className="text-xl font-bold text-white flex items-center gap-2 mb-6">
              <ShieldCheck className="w-6 h-6 text-cyan-400" /> Admin Portal Login
            </h3>
            <form onSubmit={handleAdminLogin} className="space-y-4">
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Username</label>
                <input type="text" value={loginIdentifier} onChange={e => setLoginIdentifier(e.target.value)} className="w-full bg-[#0A0F1D] border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-white" />
              </div>
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Password</label>
                <input type="password" value={loginPassword} onChange={e => setLoginPassword(e.target.value)} className="w-full bg-[#0A0F1D] border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-white" />
              </div>
              <button type="submit" className="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl mt-4">Login</button>
            </form>
          </div>
        </div>
      )}

      {/* Edit User Modal */}
      {editingUser && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[100]">
          <div className="bg-[#0D1424] border border-slate-700 p-8 rounded-2xl w-full max-w-md shadow-2xl relative">
            <button onClick={() => setEditingUser(null)} className="absolute top-4 right-4 text-slate-400 hover:text-white">✕</button>
            <h3 className="text-xl font-bold text-white flex items-center gap-2 mb-6">
              <Edit3 className="w-6 h-6 text-cyan-400" /> Edit User Record
            </h3>
            <div className="space-y-4">
              <div>
                <label className="text-xs text-slate-400 mb-1 block">User Code</label>
                <input type="text" value={editCode} onChange={e => setEditCode(e.target.value)} className="w-full bg-[#0A0F1D] border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-white" />
              </div>
              <div>
                <label className="text-xs text-slate-400 mb-1 block">Name</label>
                <input type="text" value={editName} onChange={e => setEditName(e.target.value)} className="w-full bg-[#0A0F1D] border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-white" />
              </div>
              <button onClick={handleSaveEdit} className="w-full py-3 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl mt-4">Save Changes</button>
            </div>
          </div>
        </div>
      )}

      {/* Individual User Report Modal */}
      {individualReportUser && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[100]">
          <div className="bg-[#0D1424] border border-slate-700 p-6 rounded-2xl w-full max-w-3xl shadow-2xl relative flex flex-col max-h-[90vh]">
            <button onClick={() => { setIndividualReportUser(null); setIndividualReport(null); }} className="absolute top-4 right-4 text-slate-400 hover:text-white">✕</button>
            <h3 className="text-xl font-bold text-white flex items-center gap-2 mb-2">
              <Activity className="w-6 h-6 text-cyan-400" /> Individual Attendance Report
            </h3>
            <p className="text-xs text-slate-400 mb-6 font-mono">UID: #{individualReportUser.uid} • {individualReportUser.name} ({individualReportUser.code})</p>

            <div className="flex items-center gap-4 mb-6">
              <label className="text-sm font-semibold text-slate-300">Select Month:</label>
              <input
                type="month"
                value={individualReportMonth}
                onChange={(e) => setIndividualReportMonth(e.target.value)}
                className="bg-[#0A0F1D] border border-slate-700/80 rounded-xl px-3 py-2 text-sm text-white focus:border-cyan-500 focus:outline-none transition"
              />
            </div>

            {individualReport ? (
              individualReport.status === 'success' ? (
                <>
                  <div className="grid grid-cols-4 gap-4 mb-6">
                    <div className="bg-[#0A0F1D] p-3 rounded-xl border border-slate-800/80 text-center">
                      <div className="text-[10px] text-slate-400 uppercase">Working Days</div>
                      <div className="text-lg font-bold text-white">{individualReport.total_working_days}</div>
                    </div>
                    <div className="bg-emerald-900/20 p-3 rounded-xl border border-emerald-500/20 text-center">
                      <div className="text-[10px] text-emerald-400 uppercase">Days Present</div>
                      <div className="text-lg font-bold text-emerald-400">{individualReport.days_present}</div>
                    </div>
                    <div className="bg-rose-900/20 p-3 rounded-xl border border-rose-500/20 text-center">
                      <div className="text-[10px] text-rose-400 uppercase">Days Absent</div>
                      <div className="text-lg font-bold text-rose-400">{individualReport.days_absent}</div>
                    </div>
                    <div className="bg-cyan-900/20 p-3 rounded-xl border border-cyan-500/20 text-center">
                      <div className="text-[10px] text-cyan-400 uppercase">Attendance Rate</div>
                      <div className="text-lg font-bold text-cyan-400">{individualReport.attendance_rate}%</div>
                    </div>
                  </div>

                  <h4 className="text-sm font-bold text-white mb-3">Daily Breakdown</h4>
                  <div className="flex-1 overflow-y-auto">
                    <table className="w-full text-left border-collapse">
                      <thead>
                        <tr className="border-b border-slate-800 text-[11px] font-mono uppercase text-slate-400 sticky top-0 bg-[#0D1424]">
                          <th className="py-2 px-4">Date</th>
                          <th className="py-2 px-4">Punch Time</th>
                          <th className="py-2 px-4">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/60 text-xs">
                        {individualReport.daily_breakdown.length > 0 ? individualReport.daily_breakdown.map((r: any, i: number) => (
                          <tr key={i} className="hover:bg-slate-800/30 transition">
                            <td className="py-2 px-4 text-slate-300 font-mono">{r.date}</td>
                            <td className="py-2 px-4 font-mono text-cyan-400">{r.punch_time}</td>
                            <td className="py-2 px-4">
                              <span className="px-2 py-1 bg-emerald-500/20 text-emerald-400 rounded-md text-[10px] font-bold border border-emerald-500/30">PRESENT</span>
                            </td>
                          </tr>
                        )) : (
                          <tr><td colSpan={3} className="py-4 text-center text-slate-500">No attendance records for this month.</td></tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <div className="text-rose-400 p-4 bg-rose-900/20 rounded-xl border border-rose-500/30 text-sm">
                  {individualReport.message || "Error loading report"}
                </div>
              )
            ) : (
              <div className="flex-1 flex items-center justify-center text-slate-400">Loading...</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
