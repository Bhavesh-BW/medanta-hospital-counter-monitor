import React, { useState, useEffect, useRef } from 'react';

const API_BASE = window.location.origin.includes(':8196') ? 'http://127.0.0.1:8195' : '';

export default function App() {
  const [state, setState] = useState(null);
  const [counters, setCounters] = useState([]);
  const [selectedCounterId, setSelectedCounterId] = useState('recep1');
  const [timeRange, setTimeRange] = useState('today');
  const [trends, setTrends] = useState([]);
  const [events, setEvents] = useState([]);
  const [eventFilter, setEventFilter] = useState('all');
  const [modalSnapshot, setModalSnapshot] = useState(null);
  const [bigScreenCounter, setBigScreenCounter] = useState(null);
  const [activeZoneType, setActiveZoneType] = useState('staff_zone');
  const [isAddCounterOpen, setIsAddCounterOpen] = useState(false);
  const [viewMode, setViewMode] = useState('single'); // 'single' (focused selected counter) | 'grid' (all side-by-side)
  const [isStreamPlaying, setIsStreamPlaying] = useState(true);
  const [trendViewMode, setTrendViewMode] = useState('rolling'); // 'rolling' | 'hourly'
  const [newCounterForm, setNewCounterForm] = useState({
    id: '',
    name: '',
    stream: '',
    role: 'counter',
    capacity_threshold: 5,
    monitoring_start: '09:00',
    monitoring_end: '17:00',
    is_24_7: false,
  });
  const [saveStatus, setSaveStatus] = useState('');
  const [hoveredTrendPoint, setHoveredTrendPoint] = useState(null);
  const [feedSource, setFeedSource] = useState('live');

  // Fetch state periodically
  const fetchState = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/state?counter_id=${selectedCounterId}&time_range=${timeRange}`);
      if (res.ok) {
        const data = await res.json();
        setState(data);
        if (data.counters && data.counters.length > 0) {
          setCounters(data.counters);
        }
      }
      fetch(`${API_BASE}/api/feed-source`).then(r => r.ok ? r.json() : null).then(fs => {
        if (fs?.mode) setFeedSource(fs.mode);
      }).catch(() => null);
    } catch (e) {
      console.error("Failed to fetch state:", e);
    }
  };

  // Fetch live real trends time-series from backend
  const fetchTrends = async (cid, tr) => {
    try {
      const targetCid = cid || selectedCounterId || 'all';
      const targetTr = tr || timeRange || 'today';
      const res = await fetch(`${API_BASE}/api/trends?counter_id=${targetCid}&limit=35&time_range=${targetTr}`);
      if (res.ok) {
        const data = await res.json();
        setTrends(data);
      }
    } catch (e) {
      console.error("Failed to fetch trends:", e);
    }
  };

  // Fetch events periodically
  const fetchEvents = async (cid, tr) => {
    try {
      const targetCid = cid || selectedCounterId || 'all';
      const targetTr = tr || timeRange || 'today';
      const res = await fetch(`${API_BASE}/api/events?counter_id=${targetCid}&limit=50&time_range=${targetTr}`);
      if (res.ok) {
        const data = await res.json();
        setEvents(data);
      }
    } catch (e) {
      console.error("Failed to fetch events:", e);
    }
  };

  useEffect(() => {
    fetchState();
    fetchTrends(selectedCounterId, timeRange);
    fetchEvents(selectedCounterId, timeRange);

    const interval = setInterval(() => {
      fetchState();
      fetchTrends(selectedCounterId, timeRange);
      fetchEvents(selectedCounterId, timeRange);
    }, 2500);
    return () => clearInterval(interval);
  }, [selectedCounterId, timeRange]);

  const selectedCounter = (selectedCounterId === 'all' ? counters[0] : counters.find(c => c.id === selectedCounterId)) || counters[0] || {
    id: 'recep1',
    name: 'Main Reception 1',
    stream: 'recep1',
    capacity_threshold: 5,
    unattended_seconds: 60,
    service_time_mins: 2.5,
    staff_zone: [[0.42, 0.15], [0.88, 0.15], [0.88, 0.70], [0.42, 0.70]],
    queue_zone: [[0.05, 0.28], [0.40, 0.28], [0.40, 0.95], [0.05, 0.95]],
  };

  const openBigScreen = (counter) => {
    setBigScreenCounter({
      id: counter.id,
      name: counter.name,
      stream: counter.stream,
      staff_zone: JSON.parse(JSON.stringify(counter.staff_zone)),
      queue_zone: JSON.parse(JSON.stringify(counter.queue_zone)),
      capacity_threshold: counter.capacity_threshold || 5,
      unattended_seconds: counter.unattended_seconds || 60,
      service_time_mins: counter.service_time_mins || 2.5,
      monitoring_start: counter.monitoring_start || '09:00',
      monitoring_end: counter.monitoring_end || '17:00',
      is_24_7: Boolean(counter.is_24_7),
    });
    setActiveZoneType('staff_zone');
    setSaveStatus('');
  };

  const handleSaveZones = async () => {
    if (!bigScreenCounter) return;
    setSaveStatus('Saving zones & shift rules to engine...');
    try {
      const res = await fetch(`${API_BASE}/api/counters/${bigScreenCounter.id}/zones`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          staff_zone: bigScreenCounter.staff_zone,
          queue_zone: bigScreenCounter.queue_zone,
          capacity_threshold: bigScreenCounter.capacity_threshold,
          unattended_seconds: bigScreenCounter.unattended_seconds,
          service_time_mins: bigScreenCounter.service_time_mins,
          monitoring_start: bigScreenCounter.monitoring_start,
          monitoring_end: bigScreenCounter.monitoring_end,
          is_24_7: bigScreenCounter.is_24_7,
        }),
      });
      if (res.ok) {
        setSaveStatus('✅ Zones & shift hours updated live in AI engine!');
        setCounters(prev => prev.map(c => c.id === bigScreenCounter.id ? { ...c, ...bigScreenCounter } : c));
        fetchState();
        setTimeout(() => setSaveStatus(''), 2500);
      } else {
        setSaveStatus('❌ Failed to save zones');
      }
    } catch (err) {
      setSaveStatus('❌ Network error: ' + err.message);
    }
  };

  const handleResetZones = () => {
    if (!bigScreenCounter) return;
    setBigScreenCounter({
      ...bigScreenCounter,
      staff_zone: [[0.42, 0.15], [0.88, 0.15], [0.88, 0.70], [0.42, 0.70]],
      queue_zone: [[0.05, 0.28], [0.40, 0.28], [0.40, 0.95], [0.05, 0.95]],
      capacity_threshold: 5,
      unattended_seconds: 60,
      monitoring_start: '09:00',
      monitoring_end: '17:00',
      is_24_7: false,
    });
  };

  const handleToggleFeedSource = async () => {
    const nextMode = feedSource === 'demo' ? 'live' : 'demo';
    try {
      const res = await fetch(`${API_BASE}/api/feed-source`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: nextMode })
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && data.mode) {
        setFeedSource(data.mode);
        fetchState();
        fetchEvents();
      }
    } catch (e) {
      console.error("Toggle feed source error:", e);
    }
  };

  const handleClearEvents = async () => {
    if (!window.confirm("Clear all recorded alert events from log?")) return;
    try {
      const res = await fetch(`${API_BASE}/api/events`, { method: 'DELETE' });
      if (res.ok) {
        setEvents([]);
      }
    } catch (e) {
      console.error("Failed to clear events:", e);
    }
  };

  const handleAddCounter = async (e) => {
    e.preventDefault();
    if (!newCounterForm.id || !newCounterForm.name || !newCounterForm.stream) return;
    try {
      const res = await fetch(`${API_BASE}/api/counters`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newCounterForm),
      });
      if (res.ok) {
        setIsAddCounterOpen(false);
        setNewCounterForm({
          id: '',
          name: '',
          stream: '',
          role: 'counter',
          capacity_threshold: 5,
          monitoring_start: '09:00',
          monitoring_end: '17:00',
          is_24_7: false,
        });
        fetchState();
      } else {
        alert("Failed to add counter.");
      }
    } catch (err) {
      alert("Error adding counter: " + err.message);
    }
  };

  const activeKpis = (state?.counter_kpis && state.counter_kpis[selectedCounterId])
    ? state.counter_kpis[selectedCounterId]
    : (state?.kpis || {
        avg_wait_mins: 0,
        current_queue_total: 0,
        max_queue_single: 0,
        staffed_count: 0,
        total_counters: counters.length,
        staffed_percentage: 100,
        unattended_count: 0,
        total_staff_active: 0,
      });

  // Compute live SVG trend curve coordinates
  const chartW = 380;
  const chartH = 150;
  const paddingL = 35;
  const paddingR = 15;
  const paddingT = 20;
  const paddingB = 30;
  const innerW = chartW - paddingL - paddingR;
  const innerH = chartH - paddingT - paddingB;

  const maxValY = Math.max(15, ...trends.map(t => t.est_wait_mins || 0)) * 1.15;
  const minValY = 0;

  const points = trends.map((t, idx) => {
    const x = paddingL + (idx / Math.max(1, trends.length - 1)) * innerW;
    const yVal = t.est_wait_mins ?? (t.queue * 2.5);
    const y = paddingT + innerH - (yVal / maxValY) * innerH;
    return { x, y, ...t };
  });

  const polylineStr = points.map(p => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
  const areaPolygonStr = points.length > 0
    ? `${points[0].x.toFixed(1)},${paddingT + innerH} ${polylineStr} ${points[points.length - 1].x.toFixed(1)},${paddingT + innerH}`
    : '';

  const hourlyHours = state?.hourly_rush?.hours || [];
  const maxHourlyWait = Math.max(10, ...(hourlyHours.map(h => h.avg_wait_mins || 0)));
  const peakHour = state?.hourly_rush?.peak_hour;

  const selectedWorkerState = state?.workers?.[selectedCounter.id] || {};
  const isSelectedUnattended = selectedWorkerState.status === 'unattended';

  return (
    <div className="app-container">
      {/* ── TOP HEADER (Matching Screenshot) ── */}
      <header className="app-header">
        <div className="header-title-group">
          <h1>Cash Counter Monitoring Dashboard</h1>
          <div className="header-subtitle">
            Queue Management &nbsp;|&nbsp; Staff Presence &nbsp;|&nbsp; Service Efficiency &nbsp;|&nbsp; Crowd Monitoring
          </div>
        </div>
        <div className="header-actions">
          {/* Feed Source Mode Toggle Pill */}
          <button
            type="button"
            onClick={handleToggleFeedSource}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '7px',
              padding: '6px 14px',
              borderRadius: '8px',
              fontSize: '12px',
              fontWeight: '700',
              cursor: 'pointer',
              border: feedSource === 'demo' ? '1.5px solid #8b5cf6' : '1.5px solid #10b981',
              backgroundColor: feedSource === 'demo' ? '#f5f3ff' : '#ecfdf5',
              color: feedSource === 'demo' ? '#6d28d9' : '#047857',
              boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
              transition: 'all 0.2s ease'
            }}
            title={feedSource === 'demo' ? "Currently playing Demo Recordings. Click to switch to Live RTSP Feeds." : "Currently streaming Live RTSP Feeds. Click to switch to Demo Recordings."}
          >
            <span style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: feedSource === 'demo' ? '#7c3aed' : '#10b981',
              boxShadow: feedSource === 'demo' ? '0 0 6px #7c3aed' : '0 0 6px #10b981'
            }}></span>
            <span>{feedSource === 'demo' ? '🎬 Demo Mode (Recordings)' : '🟢 Live Feeds Active'}</span>
            <span style={{ fontSize: '10px', opacity: 0.7, textDecoration: 'underline', marginLeft: '2px' }}>
              (Switch)
            </span>
          </button>

          <span style={{ fontSize: 13, color: '#475569', fontWeight: 600 }}>
            {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 12, color: '#64748b', fontWeight: 600 }}>Period:</span>
            <select
              className="campus-select"
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              title="Filter by shift or historical evidence"
            >
              <option value="today">Today (Live Shift)</option>
              <option value="yesterday">Yesterday</option>
              <option value="3days">Last 3 Days</option>
              <option value="7days">Last 7 Days</option>
              <option value="all">All Time (Evidence Archive)</option>
            </select>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 12, color: '#64748b', fontWeight: 600 }}>Focus:</span>
            <select
              className="campus-select"
              value={selectedCounterId}
              onChange={(e) => setSelectedCounterId(e.target.value)}
              title="Switch focused counter camera & trend"
            >
              <option value="all">All Counters (Campus Overview)</option>
              {counters.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
          </div>
          <div className="live-badge">
            <span className="live-dot"></span> Live
          </div>
          <button className="btn btn-primary" onClick={() => setIsAddCounterOpen(true)}>
            + Add New Counter
          </button>
        </div>
      </header>

      {/* ── TOP STATS RIBBON (5 Live Vision KPI Cards) ── */}
      <section className="kpi-grid-5">
        <div className="kpi-card">
          <div className="kpi-icon-wrap kpi-icon-purple">⏱️</div>
          <div>
            <div className="kpi-val">{activeKpis.avg_wait_mins} <span style={{ fontSize: 15 }}>mins</span></div>
            <div className="kpi-label">Avg. Queue Wait Time</div>
            <div className="kpi-sub green">Target &lt; 5m</div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-icon-wrap kpi-icon-blue">👥</div>
          <div>
            <div className="kpi-val">{activeKpis.current_queue_total} <span style={{ fontSize: 15 }}>people</span></div>
            <div className="kpi-label">{timeRange === 'today' ? 'Current Queue Length' : 'Avg. Queue Length'}</div>
            <div className="kpi-sub blue">{selectedCounterId === 'all' ? 'Across all counters' : `${selectedCounter.name}`}</div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-icon-wrap kpi-icon-amber">📈</div>
          <div>
            <div className="kpi-val">{activeKpis.max_queue_single || 0} <span style={{ fontSize: 15 }}>people</span></div>
            <div className="kpi-label">Peak Queue ({timeRange === 'today' ? 'Today' : 'Period'})</div>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>({activeKpis.max_queue_time_today || 'Shift'})</div>
          </div>
        </div>

        <div className="kpi-card" style={activeKpis.unattended_count > 0 ? { borderColor: '#ef4444', background: '#fef2f2' } : {}}>
          <div className="kpi-icon-wrap kpi-icon-red">🚨</div>
          <div>
            <div className="kpi-val" style={activeKpis.unattended_count > 0 ? { color: '#dc2626' } : {}}>
              {activeKpis.unattended_count}
            </div>
            <div className="kpi-label">{timeRange === 'today' ? 'Unattended Counters' : 'Unattended Alerts'}</div>
            <div className="kpi-sub red">
              {activeKpis.unattended_count > 0 ? `↑ ${activeKpis.unattended_count} alert active` : 'All staffed'}
            </div>
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-icon-wrap kpi-icon-teal">👨‍💼</div>
          <div>
            <div className="kpi-val">{activeKpis.total_staff_active ?? activeKpis.staffed_count} <span style={{ fontSize: 15 }}>active</span></div>
            <div className="kpi-label">Active Staff on Duty</div>
            <div className="kpi-sub green">
              {selectedCounterId === 'all'
                ? `${activeKpis.staffed_count} / ${activeKpis.total_counters} Counters (${activeKpis.staffed_percentage}%)`
                : (activeKpis.total_staff_active > 0 ? `${activeKpis.total_staff_active} Staff Present at Desk` : 'Staff Absent')}
            </div>
          </div>
        </div>
      </section>

      {/* ── MIDDLE 3-COLUMN SECTION (Exact match to screenshot!) ── */}
      <div className="middle-3col-grid">
        {/* Column 1: Live Real Queue Wait Time Trend / Hourly Rush Timeline */}
        <div className="panel-card">
          <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 6 }}>
            <div>
              <h3 className="panel-title" style={{ margin: 0 }}>Queue Wait Time Trend</h3>
              <div style={{ fontSize: 11.5, color: '#0284c7', fontWeight: 600, marginTop: 2 }}>
                ● {selectedCounter.name} ({trendViewMode === 'rolling' ? 'Live Rolling Analysis' : 'Hourly Rush & Peak Timeline'})
              </div>
            </div>
            <div style={{ display: 'flex', gap: 4, background: '#f1f5f9', padding: 2, borderRadius: 6 }}>
              <button
                type="button"
                onClick={() => setTrendViewMode('rolling')}
                style={{
                  border: 'none',
                  background: trendViewMode === 'rolling' ? '#ffffff' : 'transparent',
                  color: trendViewMode === 'rolling' ? '#0284c7' : '#64748b',
                  fontWeight: 700,
                  fontSize: 11,
                  padding: '3px 8px',
                  borderRadius: 4,
                  cursor: 'pointer',
                  boxShadow: trendViewMode === 'rolling' ? '0 1px 2px rgba(0,0,0,0.1)' : 'none',
                }}
              >
                ⏱ Live Curve
              </button>
              <button
                type="button"
                onClick={() => setTrendViewMode('hourly')}
                style={{
                  border: 'none',
                  background: trendViewMode === 'hourly' ? '#ffffff' : 'transparent',
                  color: trendViewMode === 'hourly' ? '#0284c7' : '#64748b',
                  fontWeight: 700,
                  fontSize: 11,
                  padding: '3px 8px',
                  borderRadius: 4,
                  cursor: 'pointer',
                  boxShadow: trendViewMode === 'hourly' ? '0 1px 2px rgba(0,0,0,0.1)' : 'none',
                }}
              >
                📊 Hourly Rush
              </button>
            </div>
          </div>

          {trendViewMode === 'rolling' ? (
            <div style={{ position: 'relative', width: '100%', flex: 1, minHeight: 180, display: 'flex', flexDirection: 'column' }}>
              <svg viewBox={`0 0 ${chartW} ${chartH}`} style={{ width: '100%', height: '100%', overflow: 'visible' }}>
                <defs>
                  <linearGradient id="trendGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#0284c7" stopOpacity="0.35" />
                    <stop offset="100%" stopColor="#0284c7" stopOpacity="0.02" />
                  </linearGradient>
                </defs>

                {/* Horizontal Grid lines */}
                <line x1={paddingL} y1={paddingT} x2={chartW - paddingR} y2={paddingT} stroke="#f1f5f9" strokeDasharray="3 3" />
                <line x1={paddingL} y1={paddingT + innerH * 0.33} x2={chartW - paddingR} y2={paddingT + innerH * 0.33} stroke="#f1f5f9" strokeDasharray="3 3" />
                <line x1={paddingL} y1={paddingT + innerH * 0.66} x2={chartW - paddingR} y2={paddingT + innerH * 0.66} stroke="#f1f5f9" strokeDasharray="3 3" />
                <line x1={paddingL} y1={paddingT + innerH} x2={chartW - paddingR} y2={paddingT + innerH} stroke="#e2e8f0" />

                {/* Y Axis Labels */}
                <text x={paddingL - 8} y={paddingT + 4} fontSize="9" fill="#94a3b8" textAnchor="end" fontFamily="monospace">
                  {Math.round(maxValY)}m
                </text>
                <text x={paddingL - 8} y={paddingT + innerH * 0.5 + 4} fontSize="9" fill="#94a3b8" textAnchor="end" fontFamily="monospace">
                  {Math.round(maxValY * 0.5)}m
                </text>
                <text x={paddingL - 8} y={paddingT + innerH} fontSize="9" fill="#94a3b8" textAnchor="end" fontFamily="monospace">
                  0m
                </text>

                {/* Fill area */}
                {areaPolygonStr && (
                  <polygon points={areaPolygonStr} fill="url(#trendGrad)" />
                )}

                {/* Curve Line */}
                {polylineStr && (
                  <polyline
                    fill="none"
                    stroke="#0284c7"
                    strokeWidth="2.5"
                    strokeLinejoin="round"
                    strokeLinecap="round"
                    points={polylineStr}
                  />
                )}

                {/* Data Point Nodes */}
                {points.map((pt, i) => (
                  <circle
                    key={`pt-${i}`}
                    cx={pt.x}
                    cy={pt.y}
                    r={hoveredTrendPoint?.idx === i ? "5" : "2.5"}
                    fill="#0284c7"
                    stroke="#ffffff"
                    strokeWidth="1.5"
                    style={{ cursor: 'pointer', transition: 'r 0.15s' }}
                    onMouseEnter={() => setHoveredTrendPoint({ ...pt, idx: i })}
                    onMouseLeave={() => setHoveredTrendPoint(null)}
                  />
                ))}

                {/* X Axis Time Marks (in IST) */}
                {points.length > 0 && (
                  <>
                    <text x={points[0].x} y={paddingT + innerH + 16} fontSize="9" fill="#94a3b8">
                      {points[0].short_time || points[0].time || 'Earlier'}
                    </text>
                    {points.length > 2 && (
                      <text x={points[Math.floor(points.length / 2)].x} y={paddingT + innerH + 16} fontSize="9" fill="#94a3b8" textAnchor="middle">
                        {points[Math.floor(points.length / 2)].short_time || points[Math.floor(points.length / 2)].time}
                      </text>
                    )}
                    <text x={points[points.length - 1].x} y={paddingT + innerH + 16} fontSize="9" fill="#94a3b8" textAnchor="end">
                      {points[points.length - 1].short_time || points[points.length - 1].time || 'Now (IST)'}
                    </text>
                  </>
                )}
              </svg>

              {/* Hover Tooltip */}
              {hoveredTrendPoint && (
                <div
                  style={{
                    position: 'absolute',
                    left: Math.min(hoveredTrendPoint.x + 8, chartW - 120),
                    top: Math.max(hoveredTrendPoint.y - 45, 10),
                    background: '#0f172a',
                    color: 'white',
                    padding: '4px 8px',
                    borderRadius: 6,
                    fontSize: 11,
                    boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
                    pointerEvents: 'none',
                    zIndex: 10,
                  }}
                >
                  <div><strong>{hoveredTrendPoint.time || hoveredTrendPoint.short_time} IST</strong></div>
                  <div>Queue: <strong>{hoveredTrendPoint.queue}</strong> | Wait: <strong>{hoveredTrendPoint.est_wait_mins}m</strong></div>
                </div>
              )}

              <div style={{ fontSize: 11, color: '#64748b', textAlign: 'center', marginTop: 4 }}>
                Live dynamic time-series sampled continuously in IST from AI tracker engine
              </div>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 180 }}>
              {peakHour && (
                <div
                  style={{
                    background: '#fffbeb',
                    border: '1px solid #fde68a',
                    borderRadius: 6,
                    padding: '6px 10px',
                    marginBottom: 8,
                    fontSize: 11.5,
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <span style={{ color: '#92400e', fontWeight: 700 }}>
                    🔥 Peak Busy Hour: {peakHour.hour_label} (Max {peakHour.max_queue} in queue, ~{peakHour.avg_wait_mins}m wait)
                  </span>
                  <span style={{ color: '#b45309', fontSize: 10.5, fontWeight: 600 }}>
                    Coverage: {peakHour.staff_percentage}%
                  </span>
                </div>
              )}

              {hourlyHours.length === 0 ? (
                <div style={{ padding: 24, textAlign: 'center', color: '#94a3b8', fontSize: 12 }}>
                  Collecting hourly shift metrics... Check back as traffic accumulates today.
                </div>
              ) : (
                <div
                  style={{
                    display: 'flex',
                    gap: 8,
                    overflowX: 'auto',
                    padding: '4px 2px 10px 2px',
                    scrollbarWidth: 'thin',
                  }}
                >
                  {hourlyHours.map((h, i) => {
                    const isPeak = peakHour?.hour_key === h.hour_key;
                    const barH = Math.max(8, Math.min(52, Math.round((h.avg_wait_mins / (maxHourlyWait || 1)) * 52)));
                    return (
                      <div
                        key={h.hour_key || i}
                        style={{
                          flex: '0 0 74px',
                          background: isPeak ? '#fef3c7' : '#f8fafc',
                          border: isPeak ? '2px solid #f59e0b' : '1px solid #e2e8f0',
                          borderRadius: 8,
                          padding: '7px 5px',
                          textAlign: 'center',
                          display: 'flex',
                          flexDirection: 'column',
                          alignItems: 'center',
                          gap: 3,
                          position: 'relative',
                        }}
                      >
                        {isPeak && (
                          <span style={{ fontSize: 8.5, fontWeight: 800, background: '#f59e0b', color: '#ffffff', padding: '1px 4px', borderRadius: 4 }}>
                            PEAK
                          </span>
                        )}
                        <div style={{ fontSize: 11, fontWeight: 700, color: '#1e293b' }}>
                          {h.hour_label}
                        </div>
                        <div style={{ fontSize: 9.5, color: '#94a3b8' }}>
                          {h.date_label}
                        </div>
                        <div style={{ height: 52, width: '100%', display: 'flex', alignItems: 'flex-end', justifyContent: 'center', margin: '4px 0' }}>
                          <div
                            style={{
                              width: 22,
                              height: `${barH}px`,
                              background: isPeak ? '#f59e0b' : (h.is_busy ? '#fb923c' : '#0284c7'),
                              borderRadius: '4px 4px 0 0',
                            }}
                            title={`Avg wait: ${h.avg_wait_mins}m, Max queue: ${h.max_queue}`}
                          />
                        </div>
                        <div style={{ fontSize: 11, fontWeight: 700, color: isPeak ? '#b45309' : '#0284c7' }}>
                          {h.avg_wait_mins}m wait
                        </div>
                        <div style={{ fontSize: 10, color: '#64748b' }}>
                          Max Q: <strong>{h.max_queue}</strong>
                        </div>
                        <div style={{ fontSize: 9.5, color: h.staff_percentage >= 75 ? '#16a34a' : '#ea580c', fontWeight: 600 }}>
                          {h.staff_percentage}% staff
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
              <div style={{ fontSize: 10.5, color: '#64748b', textAlign: 'center', marginTop: 'auto' }}>
                ↔ Scroll horizontally across operational hours to review peak rush periods (IST)
              </div>
            </div>
          )}
        </div>

        {/* Column 2: Counter Status Table (Click to Select Counter!) */}
        <div className="panel-card">
          <div className="panel-header">
            <h3 className="panel-title">Counter Status</h3>
            <span style={{ fontSize: 12, color: '#64748b' }}>Select to Switch Feed</span>
          </div>

          <table className="status-table">
            <thead>
              <tr>
                <th>Counter</th>
                <th>Staff</th>
                <th>Status</th>
                <th>Queue</th>
                <th>Avg. Wait</th>
              </tr>
            </thead>
            <tbody>
              {counters.map((c) => {
                const w = state?.workers?.[c.id] || {};
                const qLen = w.queue_length ?? 0;
                const estWait = w.est_wait_mins ?? 0;
                const isUnattended = w.status === 'unattended';
                const isOffShift = w.is_in_monitoring === false || w.status === 'closed';
                const isSelected = c.id === selectedCounter.id;
                const staffCount = w.staff_count ?? (w.staff_present ? 1 : 0);

                return (
                  <tr
                    key={c.id}
                    className={isSelected ? 'is-selected' : ''}
                    onClick={() => setSelectedCounterId(c.id)}
                    title="Click to view live stream & trend for this counter"
                  >
                    <td>
                      <div style={{ fontWeight: 700, color: isSelected ? '#0284c7' : '#1e293b' }}>
                        {c.name}
                      </div>
                      <div style={{ fontSize: 10.5, color: '#94a3b8', display: 'flex', gap: 6, alignItems: 'center', marginTop: 1 }}>
                        <span>{isSelected ? '● Active Focus' : c.stream}</span>
                        <span style={{ fontSize: 9.5, background: '#f1f5f9', padding: '1px 5px', borderRadius: 3, color: '#64748b' }}>
                          {c.is_24_7 ? '24/7' : `${c.monitoring_start || '09:00'} - ${c.monitoring_end || '17:00'} IST`}
                        </span>
                      </div>
                    </td>
                    <td>
                      {isOffShift ? (
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontWeight: 600, color: '#64748b', fontSize: 12 }}>
                          <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#94a3b8', display: 'inline-block' }}></span>
                          Off Duty
                        </span>
                      ) : w.staff_present ? (
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontWeight: 700, color: '#16a34a', fontSize: 12 }}>
                          <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#16a34a', display: 'inline-block' }}></span>
                          {staffCount} on duty
                        </span>
                      ) : (
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontWeight: 600, color: '#dc2626', fontSize: 12 }}>
                          <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#dc2626', display: 'inline-block' }}></span>
                          0 (Away)
                        </span>
                      )}
                    </td>
                    <td>
                      {isOffShift ? (
                        <span className="badge" style={{ background: '#f1f5f9', color: '#64748b', border: '1px solid #e2e8f0' }}>Closed (Off-Shift)</span>
                      ) : isUnattended ? (
                        <span className="badge badge-unattended">Unattended</span>
                      ) : w.anger_detected ? (
                        <span className="badge" style={{ background: '#fef2f2', color: '#dc2626', border: '1px solid #fecaca', fontWeight: 700 }}>😠 Angry Customer</span>
                      ) : w.phone_usage_active ? (
                        <span className="badge" style={{ background: '#fdf2f8', color: '#db2777', border: '1px solid #fbcfe8', fontWeight: 700 }}>📱 Phone in Use</span>
                      ) : w.staff_present ? (
                        <span className="badge badge-normal">Active</span>
                      ) : (
                        <span className="badge badge-idle">Idle</span>
                      )}
                    </td>
                    <td style={{ fontWeight: 700, fontSize: 14 }}>
                      {isOffShift ? '--' : qLen}
                    </td>
                    <td>
                      {isOffShift ? '--' : (qLen > 0 ? `${estWait} mins` : '--')}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <div style={{ marginTop: 'auto', paddingTop: 14, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 11.5, color: '#94a3b8' }}>
              Showing {counters.length} active counters
            </span>
            <button className="btn btn-secondary btn-sm" onClick={() => setIsAddCounterOpen(true)}>
              + Add Counter
            </button>
          </div>
        </div>

        {/* Column 3: Live Camera - Cash Counters (Selected Counter Live Feed) */}
        <div className="panel-card">
          <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <h3 className="panel-title" style={{ margin: 0 }}>Live Camera — Cash Counters</h3>
                {selectedWorkerState.is_in_monitoring === false && (
                  <span style={{ fontSize: 10, fontWeight: 700, padding: '1px 6px', borderRadius: 4, background: '#f1f5f9', color: '#64748b' }}>
                    ⚪ Off-Shift
                  </span>
                )}
              </div>
              <div style={{ fontSize: 12, color: '#0284c7', fontWeight: 600, marginTop: 2 }}>
                {selectedCounter.name} ({isStreamPlaying ? (selectedWorkerState.fps ? `${selectedWorkerState.fps} FPS` : 'Connecting...') : 'Paused'})
              </div>
            </div>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <button
                className={`btn ${isStreamPlaying ? 'btn-secondary' : 'btn-primary'} btn-sm`}
                onClick={() => setIsStreamPlaying(!isStreamPlaying)}
                title={isStreamPlaying ? "Pause video stream to save browser bandwidth and CPU" : "Resume video stream"}
                style={{ fontSize: 11.5, padding: '3px 9px' }}
              >
                {isStreamPlaying ? '⏸ Pause Stream' : '▶ Play Stream'}
              </button>
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => openBigScreen(selectedCounter)}
                style={{ fontSize: 11.5, padding: '3px 9px' }}
              >
                📐 Setup Zones
              </button>
            </div>
          </div>

          <div
            className={`feed-box-wrap ${isSelectedUnattended && isStreamPlaying ? 'is-unattended' : ''}`}
            onClick={() => isStreamPlaying && openBigScreen(selectedCounter)}
            title={isStreamPlaying ? "Click to open Little Big Screen Zone Setup" : "Stream paused"}
          >
            {isStreamPlaying ? (
              <>
                <img
                  className="feed-img"
                  src={`${API_BASE}/devices/${selectedCounter.id}/stream.mjpg`}
                  alt={selectedCounter.name}
                />

                <div className="feed-hover-overlay">
                  🔍 Click for Large Zone Setup
                </div>
              </>
            ) : (
              <div
                style={{
                  width: '100%',
                  aspectRatio: '16/9',
                  background: '#0f172a',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 10,
                  borderRadius: 'var(--radius-md)',
                  color: '#94a3b8',
                }}
              >
                <div style={{ fontSize: 32 }}>⏸️</div>
                <div style={{ fontWeight: 700, color: '#f8fafc', fontSize: 13.5 }}>
                  Camera Stream Paused
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8', maxWidth: '82%', textAlign: 'center' }}>
                  Live video decoding stopped to prevent browser slowdown. Background AI engine continues tracking queues.
                </div>
                <button
                  className="btn btn-primary btn-sm"
                  onClick={(e) => { e.stopPropagation(); setIsStreamPlaying(true); }}
                  style={{ marginTop: 4, fontSize: 12, padding: '5px 14px' }}
                >
                  ▶ Turn Stream ON
                </button>
              </div>
            )}

            {/* Time Stamp overlay matching screenshot */}
            <div
              style={{
                position: 'absolute',
                bottom: 8,
                right: 8,
                background: 'rgba(0, 0, 0, 0.75)',
                color: '#f8fafc',
                padding: '2px 8px',
                borderRadius: 4,
                fontSize: 11,
                fontFamily: 'monospace',
                pointerEvents: 'none',
              }}
            >
              {new Date().toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })} {new Date().toLocaleTimeString()}
            </div>
          </div>

          {/* Quick HUD Metrics under the video */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 10, fontSize: 12.5, color: '#475569' }}>
            <span>Queue: <strong>{selectedWorkerState.queue_length ?? 0} persons</strong></span>
            <span>Est. Wait: <strong>{selectedWorkerState.est_wait_mins ?? 0} mins</strong></span>
            <span>
              {selectedWorkerState.is_in_monitoring === false ? (
                <span style={{ color: '#64748b', fontWeight: 600 }}>
                  ⚪ Off-Shift ({selectedCounter.is_24_7 ? '24/7' : `${selectedCounter.monitoring_start || '09:00'} - ${selectedCounter.monitoring_end || '17:00'} IST`})
                </span>
              ) : selectedWorkerState.staff_present ? (
                <span style={{ color: '#16a34a', fontWeight: 700 }}>
                  ● {selectedWorkerState.staff_count ?? 1} Staff on Duty
                </span>
              ) : (
                <span style={{ color: '#dc2626', fontWeight: 700 }}>
                  ● Staff Absent ({selectedWorkerState.unattended_seconds ?? 0}s)
                </span>
              )}
            </span>
          </div>
        </div>
      </div>

      {/* ── RECENT EVENTS TABLE (Matching Screenshot) ── */}
      <section className="events-panel">
        <div className="panel-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
          <div>
            <h3 className="panel-title">Recent Events &amp; Audit Log</h3>
            <span style={{ fontSize: 12, color: '#64748b' }}>
              {timeRange === 'today' ? 'Live Audit Stream' : `Historical Shift Audit (${timeRange})`}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
              <span style={{ color: '#64748b', fontWeight: 600 }}>Filter:</span>
              <select
                value={eventFilter}
                onChange={(e) => setEventFilter(e.target.value)}
                style={{ padding: '4px 8px', borderRadius: 6, border: '1px solid #cbd5e1', fontSize: 12, background: 'white', fontWeight: 600, color: '#1e293b' }}
              >
                <option value="all">All Counters ({events.length})</option>
                {counters.map(c => (
                  <option key={c.id} value={c.id}>{c.name}</option>
                ))}
              </select>
            </div>
            <button
              className="btn btn-secondary btn-sm"
              style={{ fontSize: 11.5, padding: '4px 10px', color: '#dc2626', borderColor: '#fca5a5' }}
              onClick={handleClearEvents}
              title="Clear all recorded alert events from database"
            >
              🗑️ Clear Log
            </button>
          </div>
        </div>
        <table className="events-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Counter</th>
              <th>Event</th>
              <th>Details</th>
              <th>Severity</th>
              <th>Snapshot</th>
            </tr>
          </thead>
          <tbody>
            {events.filter(ev => eventFilter === 'all' || ev.counter_id === eventFilter).length === 0 ? (
              <tr>
                <td colSpan="6" style={{ textAlign: 'center', color: '#94a3b8', padding: 18 }}>
                  No recent alert events logged. All counters operating normally.
                </td>
              </tr>
            ) : (
              events
                .filter(ev => eventFilter === 'all' || ev.counter_id === eventFilter)
                .map((ev) => (
                <tr key={ev.id}>
                  <td style={{ fontFamily: 'monospace', fontSize: 12, whiteSpace: 'nowrap' }}>
                    {timeRange !== 'today' && ev.date ? `${ev.date} ` : ''}{new Date(ev.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </td>
                  <td style={{ fontWeight: 600, color: ev.event_type === 'unattended_alert' ? '#dc2626' : '#1e293b' }}>
                    {ev.counter_name}
                  </td>
                  <td>
                    {ev.event_type === 'unattended_alert' && (
                      <span className="badge badge-unattended">🚨 Unattended counter</span>
                    )}
                    {ev.event_type === 'high_queue' && (
                      <span className="badge badge-high-queue">Queue length high</span>
                    )}
                    {ev.event_type === 'presence_restored' && (
                      <span className="badge badge-normal">Normal (Staff Restored)</span>
                    )}
                    {ev.event_type === 'customer_anger' && (
                      <span
                        className="badge"
                        style={{
                          background: '#fef2f2',
                          color: '#dc2626',
                          border: '1px solid #fecaca',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: 4,
                          fontWeight: 700,
                        }}
                      >
                        😠 Angry Customer
                      </span>
                    )}
                    {(ev.event_type === 'phone_usage' || ev.event_type === 'phone_usage_staff' || ev.event_type === 'phone_usage_queue') && (
                      <span
                        className="badge"
                        style={{
                          background: '#fdf2f8',
                          color: '#db2777',
                          border: '1px solid #fbcfe8',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: 4,
                          fontWeight: 700,
                        }}
                      >
                        📱 {ev.event_type === 'phone_usage_staff' ? 'Staff Phone Usage' : 'Mobile Phone Detected'}
                      </span>
                    )}
                  </td>
                  <td>{ev.details}</td>
                  <td>
                    <span
                      style={{
                        padding: '3px 8px',
                        borderRadius: 4,
                        fontSize: 11,
                        fontWeight: 700,
                        textTransform: 'capitalize',
                        background: ev.severity === 'high' ? '#fee2e2' : ev.severity === 'medium' ? '#fef3c7' : '#e0f2fe',
                        color: ev.severity === 'high' ? '#dc2626' : ev.severity === 'medium' ? '#d97706' : '#0284c7',
                      }}
                    >
                      {ev.severity}
                    </span>
                  </td>
                  <td>
                    {ev.photo_url ? (
                      <img
                        src={`${API_BASE}${ev.crop_url || ev.photo_url}`}
                        alt="Evidence snapshot"
                        className="event-thumb"
                        onClick={() => setModalSnapshot({
                          url: `${API_BASE}${ev.photo_url}`,
                          counter_name: ev.counter_name,
                          event_type: ev.event_type,
                          time: (timeRange !== 'today' && ev.date ? `${ev.date} ` : '') + (ev.full_time || ev.time || new Date(ev.timestamp).toLocaleTimeString()),
                          details: ev.details
                        })}
                        title="Click to view full camera evidence snapshot"
                      />
                    ) : (
                      <span style={{ color: '#94a3b8', fontSize: 11 }}>—</span>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </section>

      {/* ── "LITTLE BIG SCREEN" ZONE SETUP MODAL ── */}
      {bigScreenCounter && (
        <LittleBigScreenModal
          counter={bigScreenCounter}
          workerState={state?.workers?.[bigScreenCounter.id] || {}}
          activeZoneType={activeZoneType}
          setActiveZoneType={setActiveZoneType}
          onZoneChange={(changes) => setBigScreenCounter({ ...bigScreenCounter, ...changes })}
          onSave={handleSaveZones}
          onReset={handleResetZones}
          onClose={() => setBigScreenCounter(null)}
          saveStatus={saveStatus}
        />
      )}

      {/* ── "+ ADD NEW COUNTER" MODAL ── */}
      {isAddCounterOpen && (
        <div className="modal-backdrop">
          <div className="modal-dialog-large" style={{ maxWidth: 520 }}>
            <div className="modal-header-large">
              <h2 style={{ fontSize: 18, fontWeight: 800, color: '#0f172a' }}>Add New Counter Camera</h2>
              <button className="btn btn-secondary btn-sm" onClick={() => setIsAddCounterOpen(false)}>✕</button>
            </div>
            <form onSubmit={handleAddCounter}>
              <div style={{ padding: 22, display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div>
                  <label className="form-label">Counter Name</label>
                  <input
                    className="form-input"
                    placeholder="e.g. Pharmacy Counter 1 or OPD-3"
                    value={newCounterForm.name}
                    onChange={(e) => {
                      const name = e.target.value;
                      setNewCounterForm({
                        ...newCounterForm,
                        name,
                        id: name.toLowerCase().replace(/[^a-z0-9]/g, '_')
                      });
                    }}
                    required
                  />
                </div>
                <div>
                  <label className="form-label">Counter Slug / ID</label>
                  <input
                    className="form-input"
                    placeholder="e.g. pharmacy_counter_1"
                    value={newCounterForm.id}
                    onChange={(e) => setNewCounterForm({ ...newCounterForm, id: e.target.value })}
                    required
                  />
                </div>
                <div>
                  <label className="form-label">RTSP Stream Path or URL</label>
                  <input
                    className="form-input"
                    placeholder="e.g. recep1, emer_ns_counter, cathlab_ns_counter, or rtsp://..."
                    value={newCounterForm.stream}
                    onChange={(e) => setNewCounterForm({ ...newCounterForm, stream: e.target.value })}
                    required
                  />
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>
                    Quick presets: <a href="#" onClick={(e) => { e.preventDefault(); setNewCounterForm({ ...newCounterForm, stream: 'cathlab_ns_counter', name: 'Cath Lab Counter' }); }}>cathlab_ns_counter</a>, <a href="#" onClick={(e) => { e.preventDefault(); setNewCounterForm({ ...newCounterForm, stream: 'emer_ns_counter', name: 'Emergency Counter' }); }}>emer_ns_counter</a>
                  </div>
                </div>
                <div>
                  <label className="form-label">Max Queue Capacity</label>
                  <input
                    type="number"
                    min="2"
                    max="30"
                    className="form-input"
                    value={newCounterForm.capacity_threshold}
                    onChange={(e) => setNewCounterForm({ ...newCounterForm, capacity_threshold: parseInt(e.target.value) })}
                  />
                </div>

                {/* Shift & Monitoring Hours (IST) Configuration */}
                <div style={{ borderTop: '1px solid #e2e8f0', paddingTop: 12 }}>
                  <label className="form-label" style={{ marginBottom: 6 }}>Monitoring Shift Hours (IST)</label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12.5, fontWeight: 600, color: '#1e293b', cursor: 'pointer', marginBottom: 8 }}>
                    <input
                      type="checkbox"
                      checked={Boolean(newCounterForm.is_24_7)}
                      onChange={(e) => setNewCounterForm({ ...newCounterForm, is_24_7: e.target.checked })}
                      style={{ width: 16, height: 16, accentColor: '#0284c7' }}
                    />
                    <span>24/7 Continuous Monitoring (e.g. Emergency Desk)</span>
                  </label>

                  {!newCounterForm.is_24_7 && (
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 8 }}>
                      <div>
                        <label className="form-label" style={{ fontSize: 11, marginBottom: 3 }}>Shift Start (IST)</label>
                        <input
                          type="time"
                          className="form-input"
                          style={{ fontSize: 12 }}
                          value={newCounterForm.monitoring_start}
                          onChange={(e) => setNewCounterForm({ ...newCounterForm, monitoring_start: e.target.value })}
                        />
                      </div>
                      <div>
                        <label className="form-label" style={{ fontSize: 11, marginBottom: 3 }}>Shift End (IST)</label>
                        <input
                          type="time"
                          className="form-input"
                          style={{ fontSize: 12 }}
                          value={newCounterForm.monitoring_end}
                          onChange={(e) => setNewCounterForm({ ...newCounterForm, monitoring_end: e.target.value })}
                        />
                      </div>
                    </div>
                  )}

                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', fontSize: 11 }}>
                    <span style={{ color: '#64748b', alignSelf: 'center' }}>Presets:</span>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      style={{ fontSize: 10.5, padding: '2px 7px' }}
                      onClick={() => setNewCounterForm({ ...newCounterForm, monitoring_start: '09:00', monitoring_end: '17:00', is_24_7: false })}
                    >
                      ☀️ 09:00 AM - 05:00 PM (Day)
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      style={{ fontSize: 10.5, padding: '2px 7px' }}
                      onClick={() => setNewCounterForm({ ...newCounterForm, monitoring_start: '20:00', monitoring_end: '08:00', is_24_7: false })}
                    >
                      🌙 08:00 PM - 08:00 AM (Overnight)
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      style={{ fontSize: 10.5, padding: '2px 7px' }}
                      onClick={() => setNewCounterForm({ ...newCounterForm, is_24_7: true })}
                    >
                      ⏱ 24/7 Round-the-Clock
                    </button>
                  </div>
                  <div style={{ fontSize: 10.5, color: '#64748b', marginTop: 5 }}>
                    Outside configured shift hours, counter is marked Closed / Off-Shift and absence alerts are silenced.
                  </div>
                </div>
              </div>
              <div className="modal-footer-large">
                <button type="button" className="btn btn-secondary" onClick={() => setIsAddCounterOpen(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">
                  🚀 Launch Counter Stream
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ── INCIDENT SNAPSHOT EVIDENCE MODAL ── */}
      {modalSnapshot && (
        <div className="evidence-modal-backdrop" onClick={() => setModalSnapshot(null)}>
          <div className="evidence-modal-card" onClick={e => e.stopPropagation()}>
            <div className="evidence-modal-head">
              <div>
                <h3 style={{ margin: 0, fontSize: 16, fontWeight: 800, color: '#0f172a' }}>
                  📸 Incident Evidence Snapshot · {modalSnapshot.counter_name}
                </h3>
                <small style={{ color: '#64748b', fontSize: 12, marginTop: 2, display: 'block' }}>
                  {modalSnapshot.time} — {modalSnapshot.details}
                </small>
              </div>
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => setModalSnapshot(null)}
                style={{ fontWeight: 800 }}
              >
                ✕ Close
              </button>
            </div>
            <div className="evidence-modal-body">
              <img src={modalSnapshot.url} alt="Full Frame Incident Evidence" />
            </div>
            <div className="evidence-modal-foot">
              <span style={{ fontSize: 12, color: '#475569', fontWeight: 600 }}>
                Visual evidence with YOLO person bounding boxes, zones, dwell timer, and phone usage overlays.
              </span>
              <a
                href={modalSnapshot.url}
                target="_blank"
                rel="noreferrer"
                className="btn btn-secondary btn-sm"
                style={{ textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: 4 }}
              >
                Open Original Image ↗
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Little Big Screen Zone Setup Modal Component ─────────────────────────────
function LittleBigScreenModal({ counter, workerState, activeZoneType, setActiveZoneType, onZoneChange, onSave, onReset, onClose, saveStatus }) {
  const [draggingVertex, setDraggingVertex] = useState(null);
  const [snapshotNonce, setSnapshotNonce] = useState(Date.now());
  const svgRef = useRef(null);

  const staffPts = counter.staff_zone || [];
  const queuePts = counter.queue_zone || [];

  const W = 1600;
  const H = 900;

  const toSvgPoly = (pts) => pts.map(([x, y]) => `${x * W},${y * H}`).join(' ');

  // Global window pointer listener ensures dragging to the very top edge (or outside canvas) never gets stuck!
  useEffect(() => {
    if (!draggingVertex) return;

    const handleGlobalPointerMove = (e) => {
      if (!svgRef.current) return;
      const rect = svgRef.current.getBoundingClientRect();
      if (!rect.width || !rect.height) return;

      // Smoothly clamp from 0.000 to 1.000 (0.000 is the exact top/left edge)
      const normX = Math.max(0.0, Math.min(1.0, (e.clientX - rect.left) / rect.width));
      const normY = Math.max(0.0, Math.min(1.0, (e.clientY - rect.top) / rect.height));

      const { zone, index } = draggingVertex;
      const updated = JSON.parse(JSON.stringify(counter[zone]));
      updated[index] = [Math.round(normX * 1000) / 1000, Math.round(normY * 1000) / 1000];
      onZoneChange({ [zone]: updated });
    };

    const handleGlobalPointerUp = () => {
      setDraggingVertex(null);
    };

    window.addEventListener('pointermove', handleGlobalPointerMove);
    window.addEventListener('pointerup', handleGlobalPointerUp);

    return () => {
      window.removeEventListener('pointermove', handleGlobalPointerMove);
      window.removeEventListener('pointerup', handleGlobalPointerUp);
    };
  }, [draggingVertex, counter, onZoneChange]);

  const handlePointerDown = (zone, index, e) => {
    e.preventDefault();
    e.stopPropagation();
    try {
      e.target.setPointerCapture(e.pointerId);
    } catch (err) {}
    setActiveZoneType(zone);
    setDraggingVertex({ zone, index });
  };

  const handleSnapToTop = (index) => {
    const updated = JSON.parse(JSON.stringify(counter[activeZoneType]));
    updated[index][1] = 0.0;
    onZoneChange({ [activeZoneType]: updated });
  };

  const handleCoordChange = (index, axis, value) => {
    const updated = JSON.parse(JSON.stringify(counter[activeZoneType]));
    updated[index][axis === 'x' ? 0 : 1] = Math.max(0.0, Math.min(1.0, Math.round(value * 1000) / 1000));
    onZoneChange({ [activeZoneType]: updated });
  };

  const handleNudge = (index, dx, dy) => {
    const updated = JSON.parse(JSON.stringify(counter[activeZoneType]));
    const [x, y] = updated[index];
    updated[index] = [
      Math.max(0.0, Math.min(1.0, Math.round((x + dx) * 1000) / 1000)),
      Math.max(0.0, Math.min(1.0, Math.round((y + dy) * 1000) / 1000)),
    ];
    onZoneChange({ [activeZoneType]: updated });
  };

  const calcAutoCapacity = (poly) => {
    if (!poly || poly.length < 3) return 5;
    let area = 0.0;
    for (let i = 0; i < poly.length; i++) {
      const j = (i + 1) % poly.length;
      area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
    }
    area = Math.abs(area) / 2.0;
    const est = Math.round(area / 0.0125);
    return Math.max(2, Math.min(25, est));
  };

  const activePoints = activeZoneType === 'staff_zone' ? staffPts : queuePts;
  const activeColor = activeZoneType === 'staff_zone' ? '#06b6d4' : '#f59e0b';
  const activePrefix = activeZoneType === 'staff_zone' ? 'S' : 'Q';

  return (
    <div className="modal-backdrop">
      <div className="modal-dialog-large">
        <div className="modal-header-large">
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <h2 style={{ fontSize: 18, fontWeight: 800, color: '#0f172a' }}>
                {counter.name} — Zone & Operational Parameters Setup
              </h2>
              <span className="live-badge" style={{ fontSize: 11 }}>
                Live Stream: {counter.stream}
              </span>
            </div>
            <div style={{ fontSize: 13, color: '#64748b', marginTop: 2 }}>
              Click and drag the circular corner pins directly on the camera frame to reshape zones. Both zones share the 1:1 camera aspect ratio.
            </div>
          </div>
          <button className="btn btn-secondary btn-sm" onClick={onClose} style={{ fontSize: 15, padding: '4px 12px' }}>
            ✕
          </button>
        </div>

        <div className="modal-body-large">
          {/* Left: Large Interactive Video & Canvas Screen */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <span style={{ fontSize: 12, fontWeight: 700, color: '#334155' }}>
                Camera Snapshot (16:9 Calibrated View)
              </span>
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => setSnapshotNonce(Date.now())}
                style={{ fontSize: 11.5, padding: '3px 10px' }}
                title="Fetch current frame from camera"
              >
                🔄 Refresh Frame
              </button>
            </div>

            <div className="editor-canvas-container">
              <img
                src={`${API_BASE}/devices/${counter.id}/snapshot.jpg?t=${snapshotNonce}`}
                alt={counter.name}
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  display: 'block',
                  borderRadius: 'var(--radius-md)',
                  opacity: 0.94,
                }}
              />

              <svg
                ref={svgRef}
                viewBox={`0 0 ${W} ${H}`}
                preserveAspectRatio="none"
                style={{
                  position: 'absolute',
                  inset: 0,
                  width: '100%',
                  height: '100%',
                  overflow: 'visible',
                  cursor: draggingVertex ? 'grabbing' : 'crosshair',
                }}
              >
                {/* Staff Zone Polygon (Cyan) */}
                <polygon
                  points={toSvgPoly(staffPts)}
                  fill="rgba(6, 182, 212, 0.32)"
                  stroke="#06b6d4"
                  strokeWidth={activeZoneType === 'staff_zone' ? '4' : '2'}
                  strokeDasharray={activeZoneType === 'staff_zone' ? 'none' : '6 6'}
                />
                {staffPts.map(([x, y], i) => {
                  const cx = x * W;
                  const cy = y * H;
                  const isNearTop = y < 0.12;
                  const textY = isNearTop ? cy + 24 : cy - 14;

                  return (
                    <g key={`staff-v-${i}`} style={{ cursor: 'grab' }}>
                      {/* Generous touch target */}
                      <circle
                        cx={cx}
                        cy={cy}
                        r="26"
                        fill="transparent"
                        onPointerDown={(e) => handlePointerDown('staff_zone', i, e)}
                      />
                      {/* Visible handle */}
                      <circle
                        cx={cx}
                        cy={cy}
                        r="11"
                        fill="#06b6d4"
                        stroke="#ffffff"
                        strokeWidth="3.5"
                        style={{ filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.6))' }}
                        onPointerDown={(e) => handlePointerDown('staff_zone', i, e)}
                      />
                      {/* Pin label badge */}
                      <rect
                        x={cx - 16}
                        y={textY - 9}
                        width="32"
                        height="18"
                        rx="4"
                        fill="rgba(15, 23, 42, 0.88)"
                        stroke="#06b6d4"
                        strokeWidth="1.5"
                        pointerEvents="none"
                      />
                      <text
                        x={cx}
                        y={textY + 4}
                        fill="#ffffff"
                        fontSize="11"
                        fontFamily="monospace"
                        fontWeight="800"
                        textAnchor="middle"
                        pointerEvents="none"
                      >
                        S{i + 1}
                      </text>
                    </g>
                  );
                })}

                {/* Customer Queue Zone Polygon (Amber) */}
                <polygon
                  points={toSvgPoly(queuePts)}
                  fill="rgba(245, 158, 11, 0.30)"
                  stroke="#f59e0b"
                  strokeWidth={activeZoneType === 'queue_zone' ? '4' : '2'}
                  strokeDasharray={activeZoneType === 'queue_zone' ? 'none' : '6 6'}
                />
                {queuePts.map(([x, y], i) => {
                  const cx = x * W;
                  const cy = y * H;
                  const isNearTop = y < 0.12;
                  const textY = isNearTop ? cy + 24 : cy - 14;

                  return (
                    <g key={`queue-v-${i}`} style={{ cursor: 'grab' }}>
                      {/* Generous touch target */}
                      <circle
                        cx={cx}
                        cy={cy}
                        r="26"
                        fill="transparent"
                        onPointerDown={(e) => handlePointerDown('queue_zone', i, e)}
                      />
                      {/* Visible handle */}
                      <circle
                        cx={cx}
                        cy={cy}
                        r="11"
                        fill="#f59e0b"
                        stroke="#ffffff"
                        strokeWidth="3.5"
                        style={{ filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.6))' }}
                        onPointerDown={(e) => handlePointerDown('queue_zone', i, e)}
                      />
                      {/* Pin label badge */}
                      <rect
                        x={cx - 16}
                        y={textY - 9}
                        width="32"
                        height="18"
                        rx="4"
                        fill="rgba(15, 23, 42, 0.88)"
                        stroke="#f59e0b"
                        strokeWidth="1.5"
                        pointerEvents="none"
                      />
                      <text
                        x={cx}
                        y={textY + 4}
                        fill="#ffffff"
                        fontSize="11"
                        fontFamily="monospace"
                        fontWeight="800"
                        textAnchor="middle"
                        pointerEvents="none"
                      >
                        Q{i + 1}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
            <div style={{ fontSize: 11.5, color: '#64748b', marginTop: 8, textAlign: 'center' }}>
              💡 Blue handles = Staff Desk (S1-S4) &nbsp;|&nbsp; Amber handles = Customer Queue (Q1-Q4). Drag directly to calibrate.
            </div>
          </div>

          {/* Right: Operational Configuration & Live Metrics */}
          <div className="editor-sidebar">
            <div className="sidebar-box">
              <div className="sidebar-box-title">Select Active Zone to Adjust</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                <button
                  className={`btn ${activeZoneType === 'staff_zone' ? 'btn-primary' : 'btn-secondary'}`}
                  style={activeZoneType === 'staff_zone' ? { background: '#0891b2' } : {}}
                  onClick={() => setActiveZoneType('staff_zone')}
                >
                  🔵 Staff Zone ({staffPts.length} pts)
                </button>
                <button
                  className={`btn ${activeZoneType === 'queue_zone' ? 'btn-primary' : 'btn-secondary'}`}
                  style={activeZoneType === 'queue_zone' ? { background: '#d97706' } : {}}
                  onClick={() => setActiveZoneType('queue_zone')}
                >
                  🟡 Queue Zone ({queuePts.length} pts)
                </button>
              </div>

              {/* Corner Coordinates, Sliders & Top-Snap Controls */}
              <div style={{ marginTop: 12, borderTop: '1px solid #e2e8f0', paddingTop: 10 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#334155', marginBottom: 8, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span>{activeZoneType === 'staff_zone' ? 'Staff Desk Pins (S1-S4)' : 'Customer Queue Pins (Q1-Q4)'}</span>
                  <span style={{ fontSize: 10.5, color: '#64748b' }}>Drag pin or use sliders</span>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {activePoints.map(([x, y], idx) => (
                    <div
                      key={`coord-${idx}`}
                      style={{
                        background: '#ffffff',
                        padding: '8px 10px',
                        borderRadius: 6,
                        border: draggingVertex?.index === idx ? `2px solid ${activeColor}` : '1px solid #e2e8f0',
                        fontSize: 11.5,
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                        <span style={{ fontWeight: 800, color: activeColor, fontSize: 12.5 }}>
                          ● {activePrefix}{idx + 1}
                        </span>
                        <div style={{ display: 'flex', gap: 4 }}>
                          <button
                            type="button"
                            className="btn btn-secondary btn-sm"
                            style={{ fontSize: 10, padding: '2px 8px', color: '#0284c7', fontWeight: 700 }}
                            onClick={() => handleSnapToTop(idx)}
                            title="Snap Y to 0% (Top border)"
                          >
                            ⬆ Top (Y: 0%)
                          </button>
                          <button
                            type="button"
                            className="btn btn-secondary btn-sm"
                            style={{ fontSize: 10, padding: '2px 5px' }}
                            onClick={() => handleNudge(idx, 0, -0.02)}
                            title="Nudge Up 2%"
                          >
                            ▲
                          </button>
                          <button
                            type="button"
                            className="btn btn-secondary btn-sm"
                            style={{ fontSize: 10, padding: '2px 5px' }}
                            onClick={() => handleNudge(idx, 0, 0.02)}
                            title="Nudge Down 2%"
                          >
                            ▼
                          </button>
                          <button
                            type="button"
                            className="btn btn-secondary btn-sm"
                            style={{ fontSize: 10, padding: '2px 5px' }}
                            onClick={() => handleNudge(idx, -0.02, 0)}
                            title="Nudge Left 2%"
                          >
                            ◄
                          </button>
                          <button
                            type="button"
                            className="btn btn-secondary btn-sm"
                            style={{ fontSize: 10, padding: '2px 5px' }}
                            onClick={() => handleNudge(idx, 0.02, 0)}
                            title="Nudge Right 2%"
                          >
                            ►
                          </button>
                        </div>
                      </div>

                      {/* X Slider */}
                      <div style={{ display: 'grid', gridTemplateColumns: '16px 1fr 38px', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                        <span style={{ color: '#64748b', fontWeight: 600, fontSize: 11 }}>X:</span>
                        <input
                          type="range"
                          min="0"
                          max="100"
                          value={Math.round(x * 100)}
                          onChange={(e) => handleCoordChange(idx, 'x', parseInt(e.target.value) / 100)}
                          style={{ width: '100%', height: 4 }}
                        />
                        <span style={{ fontFamily: 'monospace', fontSize: 11, textAlign: 'right', fontWeight: 600, color: '#334155' }}>
                          {Math.round(x * 100)}%
                        </span>
                      </div>

                      {/* Y Slider */}
                      <div style={{ display: 'grid', gridTemplateColumns: '16px 1fr 38px', alignItems: 'center', gap: 6 }}>
                        <span style={{ color: '#64748b', fontWeight: 600, fontSize: 11 }}>Y:</span>
                        <input
                          type="range"
                          min="0"
                          max="100"
                          value={Math.round(y * 100)}
                          onChange={(e) => handleCoordChange(idx, 'y', parseInt(e.target.value) / 100)}
                          style={{ width: '100%', height: 4 }}
                        />
                        <span style={{ fontFamily: 'monospace', fontSize: 11, textAlign: 'right', fontWeight: 600, color: '#334155' }}>
                          {Math.round(y * 100)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="sidebar-box">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <div className="sidebar-box-title" style={{ margin: 0 }}>Operational Shift & Monitoring (IST)</div>
                {workerState.is_in_monitoring === false ? (
                  <span style={{ fontSize: 10.5, fontWeight: 700, padding: '2px 7px', borderRadius: 4, background: '#f1f5f9', color: '#64748b' }}>
                    ⚪ Closed / Off-Shift
                  </span>
                ) : (
                  <span style={{ fontSize: 10.5, fontWeight: 700, padding: '2px 7px', borderRadius: 4, background: '#dcfce7', color: '#166534' }}>
                    🟢 Active Shift
                  </span>
                )}
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12.5, fontWeight: 600, color: '#1e293b', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={Boolean(counter.is_24_7)}
                    onChange={(e) => onZoneChange({ is_24_7: e.target.checked })}
                    style={{ width: 16, height: 16, accentColor: '#0284c7' }}
                  />
                  <span>24/7 Continuous Monitoring (e.g. Inpatient / Emergency)</span>
                </label>

                {!counter.is_24_7 && (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                    <div>
                      <label className="form-label" style={{ fontSize: 11, marginBottom: 3 }}>Shift Start (IST)</label>
                      <input
                        type="time"
                        className="form-input"
                        style={{ fontSize: 12, padding: '6px 8px' }}
                        value={counter.monitoring_start || '09:00'}
                        onChange={(e) => onZoneChange({ monitoring_start: e.target.value })}
                      />
                    </div>
                    <div>
                      <label className="form-label" style={{ fontSize: 11, marginBottom: 3 }}>Shift End (IST)</label>
                      <input
                        type="time"
                        className="form-input"
                        style={{ fontSize: 12, padding: '6px 8px' }}
                        value={counter.monitoring_end || '17:00'}
                        onChange={(e) => onZoneChange({ monitoring_end: e.target.value })}
                      />
                    </div>
                  </div>
                )}

                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  <span style={{ fontSize: 11, color: '#64748b', alignSelf: 'center' }}>Presets:</span>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    style={{ fontSize: 10.5, padding: '2px 7px' }}
                    onClick={() => onZoneChange({ monitoring_start: '09:00', monitoring_end: '17:00', is_24_7: false })}
                  >
                    ☀️ Day (09:00 - 17:00)
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    style={{ fontSize: 10.5, padding: '2px 7px' }}
                    onClick={() => onZoneChange({ monitoring_start: '20:00', monitoring_end: '08:00', is_24_7: false })}
                  >
                    🌙 Night (20:00 - 08:00)
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    style={{ fontSize: 10.5, padding: '2px 7px' }}
                    onClick={() => onZoneChange({ is_24_7: true })}
                  >
                    ⏱ 24/7
                  </button>
                </div>
                <div style={{ fontSize: 10.5, color: '#64748b' }}>
                  Outside these shift hours, unattended &amp; absence alerts are silenced and status is marked as Closed.
                </div>
              </div>
            </div>

            <div className="sidebar-box">
              <div className="sidebar-box-title">Detection & Alert Rules</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                    <label className="form-label" style={{ margin: 0 }}>Max Queue Capacity Limit</label>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ fontWeight: 800, color: '#0284c7' }}>{counter.capacity_threshold} people</span>
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        style={{ fontSize: 10.5, padding: '2px 8px', color: '#0284c7', borderColor: '#bae6fd', background: '#f0f9ff', fontWeight: 700 }}
                        onClick={() => onZoneChange({ capacity_threshold: calcAutoCapacity(counter.queue_zone) })}
                        title="Auto-calculate overcrowding limit based on queue zone size"
                      >
                        ⚡ Auto-Calculate ({calcAutoCapacity(counter.queue_zone)})
                      </button>
                    </div>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="15"
                    value={counter.capacity_threshold}
                    onChange={(e) => onZoneChange({ capacity_threshold: parseInt(e.target.value) })}
                    style={{ width: '100%' }}
                  />
                  <div style={{ fontSize: 11, color: '#64748b', marginTop: 4, display: 'flex', justifyContent: 'space-between' }}>
                    <span>Queue Area: <strong>~{Math.round(calcAutoCapacity(counter.queue_zone) * 1.1 * 10) / 10} m²</strong></span>
                    <span>Auto-estimated capacity: <strong>{calcAutoCapacity(counter.queue_zone)} persons</strong></span>
                  </div>
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <label className="form-label" style={{ margin: 0 }}>Unattended Alert Delay</label>
                    <span style={{ fontWeight: 800, color: '#dc2626' }}>{counter.unattended_seconds} seconds</span>
                  </div>
                  <input
                    type="range"
                    min="15"
                    max="180"
                    step="15"
                    value={counter.unattended_seconds}
                    onChange={(e) => onZoneChange({ unattended_seconds: parseInt(e.target.value) })}
                    style={{ width: '100%' }}
                  />
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <label className="form-label" style={{ margin: 0 }}>Avg Service Time per Person</label>
                    <span style={{ fontWeight: 800, color: '#0f172a' }}>{counter.service_time_mins} mins</span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="8.0"
                    step="0.5"
                    value={counter.service_time_mins}
                    onChange={(e) => onZoneChange({ service_time_mins: parseFloat(e.target.value) })}
                    style={{ width: '100%' }}
                  />
                </div>
              </div>
            </div>

            <div className="sidebar-box" style={{ background: '#f1f5f9' }}>
              <div className="sidebar-box-title" style={{ marginBottom: 8 }}>Live Telemetry Preview</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12.5 }}>
                <div>Active Staff: <strong>{workerState.staff_present ? `🟢 ${workerState.staff_count ?? 1} on duty` : '🔴 0 (Away)'}</strong></div>
                <div>Queue Length: <strong>{workerState.queue_length ?? 0} persons</strong></div>
                <div>Est. Wait: <strong>{workerState.est_wait_mins ?? 0} mins</strong></div>
                <div>Status: <strong>{workerState.is_in_monitoring === false ? '⚪ Closed (Off-Shift)' : (workerState.status || 'idle')}</strong></div>
                <div style={{ gridColumn: 'span 2', fontSize: 11.5, color: '#64748b', borderTop: '1px solid #e2e8f0', paddingTop: 6 }}>
                  Operational Shift: <strong style={{ color: '#1e293b' }}>{counter.is_24_7 ? '24/7 Continuous' : `${counter.monitoring_start || '09:00'} - ${counter.monitoring_end || '17:00'} IST`}</strong>
                </div>
              </div>
            </div>

            {saveStatus && (
              <div
                style={{
                  padding: 10,
                  borderRadius: 8,
                  fontSize: 13,
                  fontWeight: 700,
                  background: saveStatus.includes('✅') ? '#ecfdf5' : '#eff6ff',
                  color: saveStatus.includes('✅') ? '#166534' : '#1e40af',
                }}
              >
                {saveStatus}
              </div>
            )}
          </div>
        </div>

        <div className="modal-footer-large">
          <button className="btn btn-secondary" onClick={onReset}>
            ↺ Reset to Default Zones
          </button>
          <div style={{ display: 'flex', gap: 12 }}>
            <button className="btn btn-secondary" onClick={onClose}>
              Cancel
            </button>
            <button className="btn btn-primary" onClick={onSave} style={{ fontSize: 14, padding: '8px 20px' }}>
              💾 Save & Dynamically Apply (Live)
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
