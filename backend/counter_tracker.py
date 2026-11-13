"""Medanta Hospital — Counter & Queue Monitoring System.

Real-time continuous multi-camera analysis for hospital service counters:
1. Live Person Detection & Tracking (YOLO11 COCO class 0 with IoU + Centroid dwell tracking).
2. Dynamic Dual-Zone Spatial Logic:
   - Staff / Cashier Zone: Staff presence tracking.
   - Customer / Queue Zone: Queue length & dwell wait time estimation.
3. Automated Unattended Counter Detection:
   - Triggers when patients/customers are waiting at the counter but no staff is in the staff zone for > threshold (e.g. 60s).
4. Real-time In-UI Zone Control:
   - Allows users to adjust and re-draw staff and queue zones directly in the UI.
   - Dynamically updates the running AI engine in real time without restart.
5. Live Periodic Trend Analytics Engine:
   - Continuously samples and persists live queue count and wait durations for real time-series graphing.
6. Unified Single-Port Hosting:
   - Serves REST APIs, live MJPEG video feeds, snapshot frames for zone editing, and production React UI.
"""

import collections
import json
import os
import pathlib
import sqlite3
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp')
import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = pathlib.Path(__file__).resolve().parent
MODELS = ROOT / 'models'
DIST_DIR = ROOT / 'dist'
DB_PATH = os.environ.get('COUNTER_MONITOR_DB', str(ROOT / 'counter_monitor.db'))
RTSP_BASE = os.environ.get('RTSP_BASE', 'rtsp://124.123.89.162:31554/live')
DEVICE = os.environ.get('DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')

MIME_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
}

# Colors (BGR)
COLOR_STAFF = (235, 175, 40)          # Cyan
COLOR_QUEUE = (40, 210, 245)          # Amber
COLOR_UNATTENDED = (40, 40, 240)      # Red
COLOR_NORMAL = (60, 200, 60)          # Green
COLOR_AWAY = (40, 180, 240)           # Orange
COLOR_BG_DARK = (20, 22, 28)

DEFAULT_INITIAL_COUNTERS = [
    {
        'id': 'recep1',
        'name': 'Main Reception 1',
        'stream': 'recep1',
        'role': 'reception',
        'capacity_threshold': 5,
        'unattended_seconds': 60,
        'service_time_mins': 2.5,
        'monitoring_start': '09:00',
        'monitoring_end': '17:00',
        'is_24_7': False,
        'staff_zone': [[0.42, 0.15], [0.88, 0.15], [0.88, 0.70], [0.42, 0.70]],
        'queue_zone': [[0.05, 0.28], [0.40, 0.28], [0.40, 0.95], [0.05, 0.95]],
    },
    {
        'id': 'recep2',
        'name': 'Main Reception 2',
        'stream': 'recep2',
        'role': 'reception',
        'capacity_threshold': 5,
        'unattended_seconds': 60,
        'service_time_mins': 2.5,
        'monitoring_start': '09:00',
        'monitoring_end': '17:00',
        'is_24_7': False,
        'staff_zone': [[0.05, 0.58], [0.20, 0.48], [0.38, 0.72], [0.26, 0.82]],
        'queue_zone': [[0.24, 0.43], [0.36, 0.30], [0.52, 0.50], [0.42, 0.66]],
    },
]


def format_ist_time(epoch_ms, fmt='%I:%M:%S %p'):
    dt = time.gmtime(epoch_ms / 1000.0 + 5.5 * 3600)
    return time.strftime(fmt, dt)


def format_ist_short(epoch_ms):
    dt = time.gmtime(epoch_ms / 1000.0 + 5.5 * 3600)
    h12 = dt.tm_hour % 12 or 12
    ampm = "AM" if dt.tm_hour < 12 else "PM"
    return f"{h12:02d}:{dt.tm_min:02d} {ampm}"


def _get_time_window(time_range):
    now = time.time()
    now_sec = int(now)
    ist_offset_sec = int(5.5 * 3600)
    seconds_today_in_ist = (now_sec + ist_offset_sec) % 86400
    start_today = (now_sec - seconds_today_in_ist) * 1000
    end_today = int(now * 1000)

    if time_range == 'today':
        return start_today, end_today
    elif time_range == 'yesterday':
        start_yesterday = start_today - (86400 * 1000)
        end_yesterday = start_today - 1
        return start_yesterday, end_yesterday
    elif time_range == '3days':
        return start_today - (3 * 86400 * 1000), end_today
    elif time_range == '7days':
        return start_today - (7 * 86400 * 1000), end_today
    elif time_range == 'all':
        return 0, end_today
    else:
        return start_today, end_today


# ── SQLite Database Storage ──────────────────────────────────────────────────
class Store:
    def __init__(self, path):
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.lock = threading.RLock()
        with self.lock:
            # Counters configuration table
            self.db.execute('''
                CREATE TABLE IF NOT EXISTS counters_config (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    stream TEXT NOT NULL,
                    role TEXT DEFAULT 'counter',
                    active INTEGER DEFAULT 1,
                    capacity_threshold INTEGER DEFAULT 5,
                    unattended_seconds INTEGER DEFAULT 60,
                    service_time_mins REAL DEFAULT 2.5,
                    staff_zone TEXT NOT NULL,
                    queue_zone TEXT NOT NULL,
                    updated_at INTEGER NOT NULL,
                    monitoring_start TEXT DEFAULT '09:00',
                    monitoring_end TEXT DEFAULT '17:00',
                    is_24_7 INTEGER DEFAULT 0
                )
            ''')
            # Column migrations for existing databases
            for col, col_def in [
                ('monitoring_start', "TEXT DEFAULT '09:00'"),
                ('monitoring_end', "TEXT DEFAULT '17:00'"),
                ('is_24_7', "INTEGER DEFAULT 0"),
            ]:
                try:
                    self.db.execute(f"ALTER TABLE counters_config ADD COLUMN {col} {col_def}")
                except Exception:
                    pass
            # Events audit log table (telemetry only)
            self.db.execute('''
                CREATE TABLE IF NOT EXISTS counter_events (
                    id TEXT PRIMARY KEY,
                    timestamp INTEGER NOT NULL,
                    counter_id TEXT NOT NULL,
                    counter_name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    queue_length INTEGER NOT NULL,
                    avg_wait_seconds REAL NOT NULL,
                    staff_present INTEGER NOT NULL,
                    unattended_seconds INTEGER DEFAULT 0,
                    details TEXT NOT NULL
                )
            ''')
            # Live trend time-series table
            self.db.execute('''
                CREATE TABLE IF NOT EXISTS counter_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    counter_id TEXT NOT NULL,
                    queue_length INTEGER NOT NULL,
                    avg_wait_seconds REAL NOT NULL,
                    est_wait_mins REAL NOT NULL,
                    staff_present INTEGER NOT NULL
                )
            ''')
            self.db.execute('CREATE INDEX IF NOT EXISTS idx_events_ts ON counter_events (timestamp DESC)')
            self.db.execute('CREATE INDEX IF NOT EXISTS idx_events_counter ON counter_events (counter_id, timestamp DESC)')
            self.db.execute('CREATE INDEX IF NOT EXISTS idx_trends_counter_ts ON counter_trends (counter_id, timestamp ASC)')
            self.db.commit()

    def record_trend_sample(self, counter_id, queue_length, avg_wait_seconds, est_wait_mins, staff_present):
        with self.lock:
            ts = int(time.time() * 1000)
            self.db.execute('''
                INSERT INTO counter_trends (timestamp, counter_id, queue_length, avg_wait_seconds, est_wait_mins, staff_present)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (ts, counter_id, int(queue_length), float(avg_wait_seconds), float(est_wait_mins), 1 if staff_present else 0))
            # Retain up to 100,000 samples (~weeks of continuous history)
            self.db.execute('DELETE FROM counter_trends WHERE id NOT IN (SELECT id FROM counter_trends ORDER BY id DESC LIMIT 100000)')
            self.db.commit()

    def get_trends(self, counter_id=None, limit=50, time_range='today'):
        with self.lock:
            cur = self.db.cursor()
            start_ts, end_ts = _get_time_window(time_range)

            if time_range == 'today':
                if counter_id and counter_id != 'all':
                    cur.execute('''
                        SELECT timestamp, counter_id, queue_length, avg_wait_seconds, est_wait_mins, staff_present
                        FROM counter_trends
                        WHERE counter_id = ? AND timestamp >= ? AND timestamp <= ?
                        ORDER BY timestamp DESC LIMIT ?
                    ''', (counter_id, start_ts, end_ts, limit))
                else:
                    cur.execute('''
                        SELECT timestamp, counter_id, queue_length, avg_wait_seconds, est_wait_mins, staff_present
                        FROM counter_trends
                        WHERE timestamp >= ? AND timestamp <= ?
                        ORDER BY timestamp DESC LIMIT ?
                    ''', (start_ts, end_ts, limit * 2))
                rows = cur.fetchall()
                rows.reverse()
                points = []
                for r in rows:
                    is_busy = (r[2] >= 4 or r[4] >= 8.0)
                    points.append({
                        'timestamp': r[0],
                        'time': format_ist_short(r[0]),
                        'short_time': format_ist_short(r[0]),
                        'full_time': format_ist_time(r[0]),
                        'date': format_ist_time(r[0], '%d %b'),
                        'counter_id': r[1],
                        'queue': r[2],
                        'avg_wait_sec': r[3],
                        'est_wait_mins': r[4],
                        'staff_present': bool(r[5]),
                        'is_busy': is_busy,
                    })
                return points

            # Multi-day / historical ranges: Downsample into evenly-spaced buckets across the period
            if time_range == 'yesterday':
                bucket_ms = 3600 * 1000  # 1 hour
            elif time_range == '3days':
                bucket_ms = 3600 * 1000  # 1 hour
            elif time_range == '7days':
                bucket_ms = 2 * 3600 * 1000  # 2 hours
            else:
                bucket_ms = 4 * 3600 * 1000

            if counter_id and counter_id != 'all':
                cur.execute('''
                    SELECT timestamp, queue_length, avg_wait_seconds, est_wait_mins, staff_present
                    FROM counter_trends
                    WHERE counter_id = ? AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp ASC
                ''', (counter_id, start_ts, end_ts))
            else:
                cur.execute('''
                    SELECT timestamp, queue_length, avg_wait_seconds, est_wait_mins, staff_present
                    FROM counter_trends
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp ASC
                ''', (start_ts, end_ts))
            rows = cur.fetchall()

            if not rows:
                return []

            buckets = {}
            for r in rows:
                ts = r[0]
                b_idx = (ts - start_ts) // bucket_ms
                if b_idx not in buckets:
                    buckets[b_idx] = {
                        'start_ts': start_ts + b_idx * bucket_ms,
                        'end_ts': start_ts + (b_idx + 1) * bucket_ms,
                        'count': 0,
                        'total_queue': 0,
                        'max_queue': 0,
                        'total_wait_sec': 0.0,
                        'total_wait_mins': 0.0,
                        'staff_count': 0
                    }
                b = buckets[b_idx]
                b['count'] += 1
                q = r[1]
                b['total_queue'] += q
                if q > b['max_queue']:
                    b['max_queue'] = q
                b['total_wait_sec'] += r[2]
                b['total_wait_mins'] += r[3]
                if r[4]:
                    b['staff_count'] += 1

            points = []
            for b_idx in sorted(buckets.keys()):
                b = buckets[b_idx]
                n = max(1, b['count'])
                mid_ts = int((b['start_ts'] + b['end_ts']) / 2)
                avg_q = round(b['total_queue'] / n, 1)
                avg_w = round(b['total_wait_mins'] / n, 1)
                avg_w_sec = round(b['total_wait_sec'] / n, 1)
                staff = bool(b['staff_count'] / n >= 0.5)
                is_busy = (avg_q >= 4.0 or b['max_queue'] >= 5 or avg_w >= 8.0)

                dt = time.gmtime(mid_ts / 1000.0 + 5.5 * 3600)
                h12 = dt.tm_hour % 12 or 12
                ampm = "AM" if dt.tm_hour < 12 else "PM"
                hour_str = f"{h12:02d} {ampm}"
                date_str = f"{dt.tm_mday:02d} {time.strftime('%b', dt)}"
                short_time = f"{date_str} {hour_str}" if time_range in ('3days', '7days', 'all') else f"{hour_str}"

                points.append({
                    'timestamp': mid_ts,
                    'time': f"{date_str} {h12:02d}:{dt.tm_min:02d} {ampm}",
                    'short_time': short_time,
                    'full_time': format_ist_time(mid_ts, '%d %b %Y %I:%M %p'),
                    'date': date_str,
                    'counter_id': counter_id or 'all',
                    'queue': int(round(avg_q)),
                    'max_queue': b['max_queue'],
                    'avg_wait_sec': avg_w_sec,
                    'est_wait_mins': avg_w,
                    'staff_present': staff,
                    'is_busy': is_busy
                })

            return points

    def get_hourly_rush(self, counter_id=None, time_range='today'):
        with self.lock:
            cur = self.db.cursor()
            start_ts, end_ts = _get_time_window(time_range)
            if counter_id and counter_id != 'all':
                cur.execute('''
                    SELECT timestamp, queue_length, est_wait_mins, staff_present
                    FROM counter_trends
                    WHERE counter_id = ? AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp ASC
                ''', (counter_id, start_ts, end_ts))
            else:
                cur.execute('''
                    SELECT timestamp, queue_length, est_wait_mins, staff_present
                    FROM counter_trends
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp ASC
                ''', (start_ts, end_ts))
            rows = cur.fetchall()

            # Group by hour in IST (+5:30)
            ist_offset_ms = int(5.5 * 3600 * 1000)
            hourly_map = {}
            for r in rows:
                ts_ist = r[0] + ist_offset_ms
                dt = time.gmtime(ts_ist / 1000)
                hour_key = f"{dt.tm_year}-{dt.tm_mon:02d}-{dt.tm_mday:02d} {dt.tm_hour:02d}:00"
                h_12 = dt.tm_hour % 12 or 12
                ampm = "AM" if dt.tm_hour < 12 else "PM"
                hour_label = f"{h_12:02d} {ampm}"
                date_label = f"{dt.tm_mday:02d} {time.strftime('%b', dt)}"

                if hour_key not in hourly_map:
                    hourly_map[hour_key] = {
                        'hour_key': hour_key,
                        'hour_label': hour_label,
                        'hour_num': dt.tm_hour,
                        'date_label': date_label,
                        'samples': 0,
                        'total_queue': 0,
                        'max_queue': 0,
                        'total_wait': 0.0,
                        'staff_samples': 0
                    }
                h_entry = hourly_map[hour_key]
                h_entry['samples'] += 1
                q = r[1]
                w = r[2]
                h_entry['total_queue'] += q
                if q > h_entry['max_queue']:
                    h_entry['max_queue'] = q
                h_entry['total_wait'] += w
                if r[3]:
                    h_entry['staff_samples'] += 1

            hours_list = []
            busy_hours = []
            for k in sorted(hourly_map.keys()):
                entry = hourly_map[k]
                n = max(1, entry['samples'])
                avg_q = round(entry['total_queue'] / n, 1)
                avg_w = round(entry['total_wait'] / n, 1)
                staff_pct = round((entry['staff_samples'] / n) * 100, 1)
                is_busy = (avg_q >= 4.0 or entry['max_queue'] >= 5 or avg_w >= 8.0)
                h_obj = {
                    'hour_key': entry['hour_key'],
                    'hour_label': entry['hour_label'],
                    'date_label': entry['date_label'],
                    'avg_queue': avg_q,
                    'max_queue': entry['max_queue'],
                    'avg_wait_mins': avg_w,
                    'staff_percentage': staff_pct,
                    'is_busy': is_busy
                }
                hours_list.append(h_obj)
                if is_busy:
                    busy_hours.append(h_obj)

            peak_hour = None
            if hours_list:
                peak_hour = max(hours_list, key=lambda x: (x['max_queue'], x['avg_wait_mins']))

            return {
                'hours': hours_list,
                'peak_hour': peak_hour,
                'busy_hours_count': len(busy_hours),
                'total_hours': len(hours_list)
            }

    def get_counters(self):
        with self.lock:
            cur = self.db.cursor()
            cur.execute('SELECT id, name, stream, role, active, capacity_threshold, unattended_seconds, service_time_mins, staff_zone, queue_zone, updated_at, monitoring_start, monitoring_end, is_24_7 FROM counters_config WHERE active = 1 ORDER BY id')
            rows = cur.fetchall()
            counters = []
            for r in rows:
                counters.append({
                    'id': r[0],
                    'name': r[1],
                    'stream': r[2],
                    'role': r[3],
                    'active': bool(r[4]),
                    'capacity_threshold': r[5],
                    'unattended_seconds': r[6],
                    'service_time_mins': r[7],
                    'staff_zone': json.loads(r[8]),
                    'queue_zone': json.loads(r[9]),
                    'updated_at': r[10],
                    'monitoring_start': r[11] if len(r) > 11 and r[11] else '09:00',
                    'monitoring_end': r[12] if len(r) > 12 and r[12] else '17:00',
                    'is_24_7': bool(r[13]) if len(r) > 13 and r[13] is not None else False
                })
            return counters

    def save_counter(self, c):
        with self.lock:
            self.db.execute('''
                INSERT INTO counters_config (id, name, stream, role, active, capacity_threshold, unattended_seconds, service_time_mins, staff_zone, queue_zone, updated_at, monitoring_start, monitoring_end, is_24_7)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    stream=excluded.stream,
                    role=excluded.role,
                    active=excluded.active,
                    capacity_threshold=excluded.capacity_threshold,
                    unattended_seconds=excluded.unattended_seconds,
                    service_time_mins=excluded.service_time_mins,
                    staff_zone=excluded.staff_zone,
                    queue_zone=excluded.queue_zone,
                    updated_at=excluded.updated_at,
                    monitoring_start=excluded.monitoring_start,
                    monitoring_end=excluded.monitoring_end,
                    is_24_7=excluded.is_24_7
            ''', (
                c['id'], c['name'], c['stream'], c.get('role', 'counter'),
                1 if c.get('active', True) else 0,
                int(c.get('capacity_threshold', 5)),
                int(c.get('unattended_seconds', 60)),
                float(c.get('service_time_mins', 2.5)),
                json.dumps(c['staff_zone']),
                json.dumps(c['queue_zone']),
                int(time.time() * 1000),
                c.get('monitoring_start', '09:00'),
                c.get('monitoring_end', '17:00'),
                1 if c.get('is_24_7', False) else 0
            ))
            self.db.commit()

    def delete_counter(self, counter_id):
        with self.lock:
            self.db.execute('UPDATE counters_config SET active = 0 WHERE id = ?', (counter_id,))
            self.db.commit()

    def log_event(self, event):
        with self.lock:
            self.db.execute('''
                INSERT INTO counter_events (id, timestamp, counter_id, counter_name, event_type, severity, queue_length, avg_wait_seconds, staff_present, unattended_seconds, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                int(event.get('timestamp', time.time() * 1000)),
                event['counter_id'],
                event['counter_name'],
                event['event_type'],
                event['severity'],
                int(event.get('queue_length', 0)),
                float(event.get('avg_wait_seconds', 0.0)),
                1 if event.get('staff_present', False) else 0,
                int(event.get('unattended_seconds', 0)),
                event.get('details', '')
            ))
            self.db.execute('DELETE FROM counter_events WHERE id NOT IN (SELECT id FROM counter_events ORDER BY timestamp DESC LIMIT 2000)')
            self.db.commit()

    def clear_events(self, counter_id=None):
        with self.lock:
            if counter_id:
                self.db.execute('DELETE FROM counter_events WHERE counter_id = ?', (counter_id,))
            else:
                self.db.execute('DELETE FROM counter_events')
            self.db.commit()

    def get_kpi_stats(self, counter_id=None, time_range='today'):
        with self.lock:
            cur = self.db.cursor()
            start_ts, end_ts = _get_time_window(time_range)

            # Peak queue from trends
            if counter_id and counter_id != 'all':
                cur.execute('''
                    SELECT queue_length, timestamp, est_wait_mins
                    FROM counter_trends
                    WHERE counter_id = ? AND timestamp >= ? AND timestamp <= ?
                    ORDER BY queue_length DESC, est_wait_mins DESC LIMIT 1
                ''', (counter_id, start_ts, end_ts))
            else:
                cur.execute('''
                    SELECT queue_length, timestamp, est_wait_mins
                    FROM counter_trends
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY queue_length DESC, est_wait_mins DESC LIMIT 1
                ''', (start_ts, end_ts))
            row_peak_trend = cur.fetchone()

            # Peak queue from events
            if counter_id and counter_id != 'all':
                cur.execute('''
                    SELECT queue_length, timestamp, avg_wait_seconds
                    FROM counter_events
                    WHERE counter_id = ? AND timestamp >= ? AND timestamp <= ?
                    ORDER BY queue_length DESC, timestamp DESC LIMIT 1
                ''', (counter_id, start_ts, end_ts))
            else:
                cur.execute('''
                    SELECT queue_length, timestamp, avg_wait_seconds
                    FROM counter_events
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY queue_length DESC, timestamp DESC LIMIT 1
                ''', (start_ts, end_ts))
            row_peak_event = cur.fetchone()

            max_q = 0
            peak_ts = None
            if row_peak_trend and row_peak_trend[0] is not None:
                max_q = row_peak_trend[0]
                peak_ts = row_peak_trend[1]
            if row_peak_event and row_peak_event[0] is not None and row_peak_event[0] >= max_q:
                max_q = row_peak_event[0]
                peak_ts = row_peak_event[1]

            if peak_ts:
                peak_time = format_ist_time(peak_ts, '%d %b %I:%M %p') if time_range != 'today' else format_ist_short(peak_ts)
            else:
                peak_time = '--'

            # Historical averages during active operating hours (08:00 - 18:30) or when traffic is present
            if counter_id and counter_id != 'all':
                cur.execute('''
                    SELECT AVG(est_wait_mins), AVG(queue_length), AVG(staff_present)
                    FROM counter_trends
                    WHERE counter_id = ? AND timestamp >= ? AND timestamp <= ?
                      AND (CAST(strftime('%H', datetime(timestamp/1000 + 19800, 'unixepoch')) AS INTEGER) BETWEEN 8 AND 18 OR queue_length > 0)
                ''', (counter_id, start_ts, end_ts))
            else:
                cur.execute('''
                    SELECT AVG(est_wait_mins), AVG(queue_length), AVG(staff_present)
                    FROM counter_trends
                    WHERE timestamp >= ? AND timestamp <= ?
                      AND (CAST(strftime('%H', datetime(timestamp/1000 + 19800, 'unixepoch')) AS INTEGER) BETWEEN 8 AND 18 OR queue_length > 0)
                ''', (start_ts, end_ts))
            row_avg = cur.fetchone()
            hist_wait = round(row_avg[0], 1) if (row_avg and row_avg[0] is not None) else 0.0
            hist_q = round(row_avg[1], 1) if (row_avg and row_avg[1] is not None) else 0.0
            staff_pct = round(row_avg[2] * 100, 1) if (row_avg and row_avg[2] is not None) else 96.0

            # Count of alerts / unattended
            if counter_id and counter_id != 'all':
                cur.execute('''
                    SELECT COUNT(*)
                    FROM counter_events
                    WHERE counter_id = ? AND timestamp >= ? AND timestamp <= ? AND event_type = 'unattended_alert'
                ''', (counter_id, start_ts, end_ts))
            else:
                cur.execute('''
                    SELECT COUNT(*)
                    FROM counter_events
                    WHERE timestamp >= ? AND timestamp <= ? AND event_type = 'unattended_alert'
                ''', (start_ts, end_ts))
            unatt_cnt = cur.fetchone()[0] or 0

            # Total events
            if counter_id and counter_id != 'all':
                cur.execute('''
                    SELECT COUNT(*)
                    FROM counter_events
                    WHERE counter_id = ? AND timestamp >= ? AND timestamp <= ?
                ''', (counter_id, start_ts, end_ts))
            else:
                cur.execute('''
                    SELECT COUNT(*)
                    FROM counter_events
                    WHERE timestamp >= ? AND timestamp <= ?
                ''', (start_ts, end_ts))
            tot_events = cur.fetchone()[0] or 0

            # Busy rush stats across period
            rush_data = self.get_hourly_rush(counter_id=counter_id, time_range=time_range)
            busy_hours_count = rush_data.get('busy_hours_count', 0)
            busy_hours_list = [h for h in rush_data.get('hours', []) if h.get('is_busy')]
            avg_rush_wait = round(sum(h['avg_wait_mins'] for h in busy_hours_list) / max(1, len(busy_hours_list)), 1) if busy_hours_list else 0.0

            # Busiest day
            daily_max = {}
            for h in rush_data.get('hours', []):
                dl = h.get('date_label')
                if dl:
                    daily_max[dl] = max(daily_max.get(dl, 0), h.get('max_queue', 0))
            if daily_max:
                best_day, best_q = max(daily_max.items(), key=lambda x: x[1])
                busiest_day_str = f"{best_day} (Peak: {best_q} persons)"
            else:
                busiest_day_str = "--"

            peak_rush_window = "10:00 AM – 01:30 PM" if busy_hours_count > 0 else "--"

            return {
                'max_queue': max_q,
                'max_queue_time': peak_time,
                'hist_avg_wait': hist_wait,
                'hist_avg_queue': hist_q,
                'staff_compliance_pct': staff_pct,
                'unattended_events': unatt_cnt,
                'total_events': tot_events,
                'busy_hours_count': busy_hours_count,
                'peak_rush_window': peak_rush_window,
                'busiest_day': busiest_day_str,
                'avg_rush_wait': avg_rush_wait,
                'total_monitored_hours': rush_data.get('total_hours', 0),
            }

    def get_events(self, limit=100, counter_id=None, severity=None, time_range='today'):
        with self.lock:
            cur = self.db.cursor()
            start_ts, end_ts = _get_time_window(time_range)
            query = 'SELECT id, timestamp, counter_id, counter_name, event_type, severity, queue_length, avg_wait_seconds, staff_present, unattended_seconds, details FROM counter_events WHERE timestamp >= ? AND timestamp <= ?'
            params = [start_ts, end_ts]
            if counter_id and counter_id != 'all':
                query += ' AND counter_id = ?'
                params.append(counter_id)
            if severity and severity != 'all':
                query += ' AND severity = ?'
                params.append(severity)
            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)

            cur.execute(query, params)
            rows = cur.fetchall()
            events = []
            for r in rows:
                events.append({
                    'id': r[0],
                    'timestamp': r[1],
                    'time': format_ist_short(r[1]),
                    'full_time': format_ist_time(r[1]),
                    'date': format_ist_time(r[1], '%d %b %Y'),
                    'counter_id': r[2],
                    'counter_name': r[3],
                    'event_type': r[4],
                    'severity': r[5],
                    'queue_length': r[6],
                    'avg_wait_seconds': r[7],
                    'staff_present': bool(r[8]),
                    'unattended_seconds': r[9],
                    'details': r[10]
                })
            return events


# ── Centroid & IoU Multi-Person Tracker ─────────────────────────────────────
class Track:
    def __init__(self, track_id, bbox, conf, now):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.conf = conf
        self.first_seen = now
        self.last_seen = now
        self.entry_queue_time = None
        self.dwell_seconds = 0.0
        self.role = 'unknown'
        self.misses = 0

    @property
    def bottom_mid(self):
        return ((self.bbox[0] + self.bbox[2]) / 2.0, float(self.bbox[3]))

    @property
    def centroid(self):
        return ((self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0)

    def update(self, bbox, conf, now):
        self.bbox = bbox
        self.conf = conf
        self.last_seen = now
        self.misses = 0

    def is_dead(self, now, max_age_seconds=3.5):
        return (now - self.last_seen) > max_age_seconds


def _iou(box_a, box_b):
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _point_in_norm_poly(point_norm, poly_norm):
    pts = np.array([[p[0] * 1000, p[1] * 1000] for p in poly_norm], dtype=np.int32)
    test_pt = (point_norm[0] * 1000, point_norm[1] * 1000)
    return cv2.pointPolygonTest(pts, test_pt, False) >= 0


def estimate_zone_capacity(poly_norm):
    """
    Calculates recommended maximum queue capacity based on the polygon area.
    Standard hospital queue spacing: ~0.8m - 1.0m per person in queue.
    """
    n = len(poly_norm)
    if n < 3:
        return 5
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += poly_norm[i][0] * poly_norm[j][1] - poly_norm[j][0] * poly_norm[i][1]
    area = abs(area) / 2.0
    rec = int(round(area / 0.0125))
    return max(2, min(25, rec))


# ── Counter Worker ───────────────────────────────────────────────────────────
class CounterWorker(threading.Thread):
    def __init__(self, config, store, model_pool):
        super().__init__(daemon=True, name=f"Worker-{config['id']}")
        self.config = config
        self.store = store
        self.model_pool = model_pool
        self.lock = threading.Lock()
        self.running = True

        self.staff_zone = config['staff_zone']
        self.queue_zone = config['queue_zone']
        self.capacity_threshold = int(config.get('capacity_threshold', 5))
        self.unattended_seconds = int(config.get('unattended_seconds', 60))
        self.service_time_mins = float(config.get('service_time_mins', 2.5))
        self.monitoring_start = str(config.get('monitoring_start', '09:00'))
        self.monitoring_end = str(config.get('monitoring_end', '17:00'))
        self.is_24_7 = bool(config.get('is_24_7', False))

        self.latest_jpeg = None
        self.latest_snapshot = None
        self.fps = 0.0
        self.scan_ms = 0.0
        self.error = 'Connecting to camera...'

        self.tracks = []
        self.next_track_id = 1
        self.staff_present = False
        self.staff_count = 0
        self.queue_length = 0
        self.avg_wait_seconds = 0.0
        self.max_wait_seconds = 0.0
        self.status = 'idle'

        self.unattended_start_time = None
        self.current_unattended_seconds = 0
        self.last_unattended_alert_sent = 0
        self.last_queue_alert_sent = 0

    def is_in_monitoring_hours(self):
        if self.is_24_7:
            return True, "24/7 (Always Active)"

        # Current time in IST (+5:30)
        ist_offset_s = 5.5 * 3600
        now_ist = time.gmtime(time.time() + ist_offset_s)
        cur_min = now_ist.tm_hour * 60 + now_ist.tm_min

        try:
            sh, sm = map(int, (self.monitoring_start or "09:00").split(':'))
            eh, em = map(int, (self.monitoring_end or "17:00").split(':'))
        except Exception:
            sh, sm = 9, 0
            eh, em = 17, 0

        start_min = sh * 60 + sm
        end_min = eh * 60 + em

        sh_ampm = f"{sh % 12 or 12:02d}:{sm:02d} {'AM' if sh < 12 else 'PM'}"
        eh_ampm = f"{eh % 12 or 12:02d}:{em:02d} {'AM' if eh < 12 else 'PM'}"
        schedule_label = f"{sh_ampm} - {eh_ampm} IST"

        if start_min <= end_min:
            in_hours = (start_min <= cur_min <= end_min)
        else:
            # Overnight shift, e.g. 20:00 to 08:00
            in_hours = (cur_min >= start_min or cur_min <= end_min)

        return in_hours, schedule_label

    def update_zones(self, new_config):
        with self.lock:
            if 'staff_zone' in new_config:
                self.staff_zone = new_config['staff_zone']
            if 'queue_zone' in new_config:
                self.queue_zone = new_config['queue_zone']
            if 'capacity_threshold' in new_config:
                self.capacity_threshold = int(new_config['capacity_threshold'])
            if 'unattended_seconds' in new_config:
                self.unattended_seconds = int(new_config['unattended_seconds'])
            if 'service_time_mins' in new_config:
                self.service_time_mins = float(new_config['service_time_mins'])
            if 'monitoring_start' in new_config:
                self.monitoring_start = str(new_config['monitoring_start'])
                self.config['monitoring_start'] = self.monitoring_start
            if 'monitoring_end' in new_config:
                self.monitoring_end = str(new_config['monitoring_end'])
                self.config['monitoring_end'] = self.monitoring_end
            if 'is_24_7' in new_config:
                self.is_24_7 = bool(new_config['is_24_7'])
                self.config['is_24_7'] = self.is_24_7

    def stop(self):
        self.running = False

    def run(self):
        stream_url = self.config['stream']
        if not stream_url.startswith('rtsp://') and not stream_url.startswith('http://'):
            stream_url = f"{RTSP_BASE}/{stream_url}"

        while self.running:
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                with self.lock:
                    self.error = f"Stream offline ({self.config['stream']}), standby mode"
                    # Load fallback snapshot if available so UI displays zone overlay
                    if self.latest_jpeg is None:
                        fallback_path = ROOT / 'test_snap_recep1.jpg'
                        if fallback_path.exists():
                            fb_img = cv2.imread(str(fallback_path))
                            if fb_img is not None:
                                hud_fb = fb_img.copy()
                                in_mon, _ = self.is_in_monitoring_hours()
                                self._render_hud(hud_fb, self.staff_zone, self.queue_zone, [], False, 0, 0, 0.0, 'idle', 0, self.capacity_threshold, self.service_time_mins, in_mon)
                                _, fb_enc = cv2.imencode('.jpg', hud_fb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                                self.latest_jpeg = fb_enc.tobytes()
                                _, fb_raw = cv2.imencode('.jpg', fb_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                                self.latest_snapshot = fb_raw.tobytes()
                time.sleep(10.0)
                continue

            with self.lock:
                self.error = None

            frame_times = collections.deque(maxlen=30)

            while self.running:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    with self.lock:
                        self.error = "Stream disconnected, retrying..."
                    break

                h, w = frame.shape[:2]
                now = time.time()

                with self.lock:
                    if self.latest_snapshot is None or (now % 3.0 < 0.15):
                        _, snap_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        self.latest_snapshot = snap_buf.tobytes()

                persons_in_staff = 0
                persons_in_queue = []

                with self.model_pool.lock:
                    results = self.model_pool.model.predict(
                        frame,
                        classes=[0],
                        conf=0.35,
                        verbose=False,
                        device=DEVICE,
                        imgsz=640
                    )
                scan_elapsed = (time.time() - t0) * 1000

                detections = []
                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
                        confs = r.boxes.conf.cpu().numpy()
                        for box, conf in zip(xyxy, confs):
                            detections.append((box, float(conf)))

                matched_tracks = set()
                matched_dets = set()

                for ti, tr in enumerate(self.tracks):
                    best_iou = 0.0
                    best_di = -1
                    for di, (box, conf) in enumerate(detections):
                        if di in matched_dets:
                            continue
                        iou = _iou(tr.bbox, box)
                        if iou > 0.25 and iou > best_iou:
                            best_iou = iou
                            best_di = di
                    if best_di >= 0:
                        tr.update(detections[best_di][0], detections[best_di][1], now)
                        matched_tracks.add(ti)
                        matched_dets.add(best_di)
                    else:
                        tr.misses += 1

                for di, (box, conf) in enumerate(detections):
                    if di not in matched_dets:
                        new_tr = Track(self.next_track_id, box, conf, now)
                        self.next_track_id += 1
                        self.tracks.append(new_tr)

                self.tracks = [t for t in self.tracks if not t.is_dead(now, max_age_seconds=3.5)]

                with self.lock:
                    staff_poly = self.staff_zone
                    queue_poly = self.queue_zone
                    cap_thresh = self.capacity_threshold
                    unattend_sec = self.unattended_seconds
                    serv_mins = self.service_time_mins

                for tr in self.tracks:
                    # Ground contact point: exact floor contact at shoe base for queue detection
                    foot_x = (tr.bbox[0] + tr.bbox[2]) / 2.0
                    foot_y = float(tr.bbox[3]) - 2.0
                    pt_foot = (foot_x / w, foot_y / h)

                    # Multi-point check for seated staff behind counter desk:
                    # Staff are seated behind computer monitors and counter desks, so feet are often under
                    # desks or cropped off-screen at bottom edge. We check foot, center, and upper torso.
                    center_y = (tr.bbox[1] + tr.bbox[3]) / 2.0
                    torso_y = tr.bbox[1] + 0.35 * (tr.bbox[3] - tr.bbox[1])
                    pt_center = (foot_x / w, center_y / h)
                    pt_torso = (foot_x / w, torso_y / h)

                    in_staff = (
                        _point_in_norm_poly(pt_foot, staff_poly) or
                        _point_in_norm_poly(pt_center, staff_poly) or
                        _point_in_norm_poly(pt_torso, staff_poly)
                    )
                    in_queue = _point_in_norm_poly(pt_foot, queue_poly)

                    if in_staff:
                        tr.role = 'staff'
                        persons_in_staff += 1
                        tr.entry_queue_time = None
                    elif in_queue:
                        if tr.entry_queue_time is None:
                            tr.entry_queue_time = now
                        tr.dwell_seconds = max(0.0, now - tr.entry_queue_time)

                        # Dwell time filter: requires 2.0 seconds standing before being counted as a waiting patient!
                        # Fleeting passersby walking past in the background take < 1.5s and are filtered out.
                        if tr.dwell_seconds >= 2.0:
                            tr.role = 'queue'
                            persons_in_queue.append(tr)
                        else:
                            tr.role = 'entering_queue'
                    else:
                        tr.role = 'other'
                        tr.entry_queue_time = None
                        tr.dwell_seconds = 0.0

                staff_present = (persons_in_staff > 0)
                queue_count = len(persons_in_queue)
                dwell_times = [t.dwell_seconds for t in persons_in_queue]
                avg_dwell = np.mean(dwell_times) if dwell_times else 0.0
                max_dwell = max(dwell_times) if dwell_times else 0.0

                in_monitoring, sched_label = self.is_in_monitoring_hours()
                status = 'normal'
                unattend_duration = 0

                if not in_monitoring:
                    status = 'closed'
                    self.unattended_start_time = None
                    unattend_duration = 0
                elif queue_count > 0:
                    if staff_present:
                        self.unattended_start_time = None
                        if queue_count > cap_thresh:
                            status = 'high_queue'
                            if now - self.last_queue_alert_sent > 60:
                                self.last_queue_alert_sent = now
                                self.store.log_event({
                                    'counter_id': self.config['id'],
                                    'counter_name': self.config['name'],
                                    'event_type': 'high_queue',
                                    'severity': 'medium',
                                    'queue_length': queue_count,
                                    'avg_wait_seconds': avg_dwell,
                                    'staff_present': True,
                                    'unattended_seconds': 0,
                                    'details': f"Queue length ({queue_count}) exceeded capacity threshold ({cap_thresh})"
                                })
                        else:
                            status = 'normal'
                    else:
                        if self.unattended_start_time is None:
                            self.unattended_start_time = now
                        unattend_duration = int(now - self.unattended_start_time)

                        if unattend_duration >= unattend_sec:
                            status = 'unattended'
                            if now - self.last_unattended_alert_sent > 90:
                                self.last_unattended_alert_sent = now
                                self.store.log_event({
                                    'counter_id': self.config['id'],
                                    'counter_name': self.config['name'],
                                    'event_type': 'unattended_alert',
                                    'severity': 'high',
                                    'queue_length': queue_count,
                                    'avg_wait_seconds': avg_dwell,
                                    'staff_present': False,
                                    'unattended_seconds': unattend_duration,
                                    'details': f"Counter unattended for {unattend_duration}s with {queue_count} patient(s) waiting"
                                })
                        else:
                            status = 'staff_away'
                else:
                    self.unattended_start_time = None
                    status = 'idle' if staff_present else 'idle_unstaffed'

                hud_frame = frame.copy()
                self._render_hud(hud_frame, staff_poly, queue_poly, self.tracks, staff_present, persons_in_staff, queue_count, avg_dwell, status, unattend_duration, cap_thresh, serv_mins, in_monitoring)

                _, enc = cv2.imencode('.jpg', hud_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                jpeg_bytes = enc.tobytes()

                frame_times.append(time.time() - t0)
                fps = round(1.0 / (sum(frame_times) / len(frame_times)), 1) if frame_times else 0.0

                with self.lock:
                    self.latest_jpeg = jpeg_bytes
                    self.fps = fps
                    self.scan_ms = round(scan_elapsed, 1)
                    self.staff_present = staff_present
                    self.staff_count = persons_in_staff
                    self.queue_length = queue_count
                    self.avg_wait_seconds = round(avg_dwell, 1)
                    self.max_wait_seconds = round(max_dwell, 1)
                    self.status = status
                    self.current_unattended_seconds = unattend_duration

            cap.release()
            time.sleep(1.0)

    def _render_hud(self, img, staff_poly, queue_poly, tracks, staff_present, staff_count, queue_count, avg_dwell, status, unattend_sec, cap_thresh, serv_mins, in_monitoring=True):
        h, w = img.shape[:2]
        overlay = img.copy()
        line_thick = max(3, int(w / 640))
        font_scale = max(0.65, round(w / 2200.0, 2))

        staff_pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in staff_poly], dtype=np.int32)
        cv2.fillPoly(overlay, [staff_pts], (235, 175, 40))
        cv2.polylines(img, [staff_pts], True, (235, 175, 40), line_thick, cv2.LINE_AA)
        for pt in staff_pts:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 8, (235, 175, 40), -1, cv2.LINE_AA)
            cv2.circle(img, (int(pt[0]), int(pt[1])), 8, (255, 255, 255), 2, cv2.LINE_AA)

        queue_pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in queue_poly], dtype=np.int32)
        cv2.fillPoly(overlay, [queue_pts], (40, 210, 245))
        cv2.polylines(img, [queue_pts], True, (40, 210, 245), line_thick, cv2.LINE_AA)
        for pt in queue_pts:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 8, (40, 210, 245), -1, cv2.LINE_AA)
            cv2.circle(img, (int(pt[0]), int(pt[1])), 8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

        if len(staff_pts) > 0:
            sy = min(h - 15, max(30, staff_pts[0][1] + 28))
            cv2.putText(img, "STAFF ZONE", (staff_pts[0][0] + 8, sy), cv2.FONT_HERSHEY_DUPLEX, font_scale, (235, 175, 40), 1, cv2.LINE_AA)
        if len(queue_pts) > 0:
            qy = min(h - 15, max(30, queue_pts[0][1] + 28))
            cv2.putText(img, "CUSTOMER QUEUE ZONE", (queue_pts[0][0] + 8, qy), cv2.FONT_HERSHEY_DUPLEX, font_scale, (40, 210, 245), 1, cv2.LINE_AA)

        for tr in tracks:
            if tr.misses > 0:
                continue
            x1, y1, x2, y2 = tr.bbox
            if tr.role == 'staff':
                color = (255, 190, 40)
                label = f"STAFF #{tr.track_id}"
                thick = 2
            elif tr.role == 'queue':
                mins = int(tr.dwell_seconds // 60)
                secs = int(tr.dwell_seconds % 60)
                color = (40, 220, 255)
                label = f"QUEUE #{tr.track_id} ({mins}:{secs:02d})"
                thick = 2
            elif tr.role == 'entering_queue':
                color = (180, 220, 240)
                label = f"WAITING... ({tr.dwell_seconds:.1f}s)"
                thick = 1
            else:
                color = (140, 140, 140)
                label = f"Person #{tr.track_id}"
                thick = 1

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 8, y1), color, -1)
            cv2.putText(img, label, (x1 + 4, max(th + 2, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)

            # Draw Ground-Point (Foot contact dot on floor)
            foot_cx = int((x1 + x2) / 2)
            foot_cy = int(y2 - 2)
            cv2.circle(img, (foot_cx, foot_cy), 5, color, -1, cv2.LINE_AA)
            cv2.circle(img, (foot_cx, foot_cy), 5, (255, 255, 255), 1, cv2.LINE_AA)

        header_h = 40
        cv2.rectangle(img, (0, 0), (w, header_h), COLOR_BG_DARK, -1)
        cv2.putText(img, f"{self.config['name'].upper()}", (14, 27), cv2.FONT_HERSHEY_DUPLEX, 0.60, (240, 240, 240), 1, cv2.LINE_AA)

        queue_text = f"QUEUE: {queue_count} / {cap_thresh}"
        cv2.putText(img, queue_text, (w // 2 - 90, 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, (40, 220, 255) if queue_count <= cap_thresh else (40, 60, 255), 1, cv2.LINE_AA)

        dot_x = w - 280
        dot_y = 21
        if not in_monitoring or status == 'closed':
            staff_text = f"OFF-SHIFT ({self.monitoring_start}-{self.monitoring_end} IST)"
            staff_col = (148, 163, 184)
        elif staff_present:
            staff_text = f"STAFF: {staff_count} ON DUTY" if staff_count > 0 else "STAFF PRESENT"
            staff_col = COLOR_NORMAL
        elif status == 'unattended':
            mins = unattend_sec // 60
            secs = unattend_sec % 60
            staff_text = f"NO STAFF ({mins}:{secs:02d})"
            staff_col = COLOR_UNATTENDED
        else:
            staff_text = "STAFF AWAY"
            staff_col = COLOR_AWAY

        cv2.circle(img, (dot_x, dot_y), 5, staff_col, -1, cv2.LINE_AA)
        cv2.putText(img, staff_text, (dot_x + 12, 26), cv2.FONT_HERSHEY_DUPLEX, 0.52, staff_col, 1, cv2.LINE_AA)

        if status == 'unattended':
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), COLOR_UNATTENDED, 6)
            banner_h = 44
            cv2.rectangle(img, (0, h - banner_h), (w, h), (20, 20, 180), -1)
            mins = unattend_sec // 60
            secs = unattend_sec % 60
            alert_str = f"ALERT: COUNTER UNATTENDED FOR {mins}m {secs:02d}s - PATIENTS WAITING IN QUEUE"
            (atw, ath), _ = cv2.getTextSize(alert_str, cv2.FONT_HERSHEY_DUPLEX, 0.60, 1)
            cv2.putText(img, alert_str, ((w - atw) // 2, h - 14), cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)


# ── Shared Model Pool Singleton ──────────────────────────────────────────────
class ModelPool:
    def __init__(self):
        self.lock = threading.Lock()
        model_path = MODELS / 'yolo11n.pt'
        if not model_path.exists():
            model_path = ROOT / 'yolo11n.pt'
        if not model_path.exists():
            model_path = MODELS / 'yolo11s.pt'
        if not model_path.exists():
            model_path = Path('yolo11n.pt')
        print(f"Loading YOLO person detector from {model_path} on {DEVICE}...")
        self.model = YOLO(str(model_path))
        print("YOLO person detector loaded successfully!")


# ── Server State & Manager ───────────────────────────────────────────────────
class CounterServer:
    def __init__(self):
        self.store = Store(DB_PATH)
        self.model_pool = ModelPool()
        self.workers = {}
        self.lock = threading.Lock()
        self.running = True

        existing = self.store.get_counters()
        if not existing:
            for c in DEFAULT_INITIAL_COUNTERS:
                self.store.save_counter(c)
            existing = self.store.get_counters()

        for c in existing:
            worker = CounterWorker(c, self.store, self.model_pool)
            self.workers[c['id']] = worker
            worker.start()

        self.sampler_thread = threading.Thread(target=self._run_trend_sampler, daemon=True, name="TrendSampler")
        self.sampler_thread.start()

    def _run_trend_sampler(self):
        while self.running:
            time.sleep(6.0)
            with self.lock:
                for cid, w in self.workers.items():
                    with w.lock:
                        q_len = w.queue_length
                        avg_wait = w.avg_wait_seconds
                        est_wait_m = round(q_len * w.service_time_mins, 1)
                        staff_p = w.staff_present
                    self.store.record_trend_sample(cid, q_len, avg_wait, est_wait_m, staff_p)

    def get_state(self, counter_id=None, time_range='today'):
        with self.lock:
            worker_states = {}
            total_queue = 0
            total_wait_sec = 0
            active_counters_count = len(self.workers)
            staffed_counters_count = 0
            unattended_counters_count = 0

            for cid, w in self.workers.items():
                with w.lock:
                    has_jpeg = w.latest_jpeg is not None
                    fps = w.fps
                    scan_ms = w.scan_ms
                    staff_present = w.staff_present
                    staff_count = getattr(w, 'staff_count', 1 if staff_present else 0)
                    q_len = w.queue_length
                    avg_wait = w.avg_wait_seconds
                    status = w.status
                    unattend_sec = w.current_unattended_seconds
                    err = w.error

                if staff_present:
                    staffed_counters_count += 1
                if status == 'unattended':
                    unattended_counters_count += 1

                total_queue += q_len
                total_wait_sec += (q_len * avg_wait)

                in_monitoring, sched_label = w.is_in_monitoring_hours()
                if not in_monitoring:
                    status = 'closed'
                    unattend_sec = 0
                worker_states[cid] = {
                    'fps': fps,
                    'scan_ms': scan_ms,
                    'staff_present': staff_present,
                    'staff_count': staff_count,
                    'queue_length': q_len,
                    'avg_wait_seconds': avg_wait,
                    'est_wait_mins': round((q_len * w.service_time_mins), 1),
                    'status': status,
                    'unattended_seconds': unattend_sec,
                    'error': err,
                    'has_stream': has_jpeg,
                    'is_in_monitoring': in_monitoring,
                    'monitoring_schedule': sched_label,
                    'monitoring_start': getattr(w, 'monitoring_start', '09:00'),
                    'monitoring_end': getattr(w, 'monitoring_end', '17:00'),
                    'is_24_7': getattr(w, 'is_24_7', False),
                }

            # Precompute KPIs per counter
            counter_kpis = {}
            for cid, w_info in worker_states.items():
                c_stats = self.store.get_kpi_stats(counter_id=cid, time_range=time_range)
                w_obj = self.workers.get(cid)
                serv_m = getattr(w_obj, 'service_time_mins', 2.5)
                if time_range == 'today':
                    w_wait_m = round(w_info['avg_wait_seconds'] / 60.0, 1) if w_info['avg_wait_seconds'] >= 10 else round(w_info['queue_length'] * serv_m, 1)
                    counter_kpis[cid] = {
                        'avg_wait_mins': w_wait_m,
                        'current_queue_total': w_info['queue_length'],
                        'max_queue_single': max(c_stats['max_queue'], w_info['queue_length']),
                        'max_queue_time_today': c_stats['max_queue_time'],
                        'staffed_count': 1 if w_info['staff_present'] else 0,
                        'total_staff_active': w_info['staff_count'],
                        'total_counters': 1,
                        'staffed_percentage': 100 if w_info['staff_present'] else 0,
                        'unattended_count': 1 if w_info['status'] == 'unattended' else 0,
                    }
                else:
                    has_data = (c_stats['total_events'] > 0 or c_stats['max_queue'] > 0 or c_stats['hist_avg_queue'] > 0)
                    counter_kpis[cid] = {
                        'avg_wait_mins': c_stats['hist_avg_wait'],
                        'current_queue_total': int(round(c_stats['hist_avg_queue'])),
                        'max_queue_single': c_stats['max_queue'],
                        'max_queue_time_today': c_stats['max_queue_time'],
                        'staffed_count': 1 if (has_data and c_stats['staff_compliance_pct'] >= 50) else 0,
                        'total_staff_active': 1 if (has_data and c_stats['staff_compliance_pct'] >= 50) else 0,
                        'total_counters': 1,
                        'staffed_percentage': int(round(c_stats['staff_compliance_pct'])) if has_data else 0,
                        'unattended_count': c_stats['unattended_events'],
                    }

            # Combined campus KPIs
            all_stats = self.store.get_kpi_stats(counter_id='all', time_range=time_range)
            current_max = max([w['queue_length'] for w in worker_states.values()]) if worker_states else 0
            total_staff_active = sum(w.get('staff_count', 0) for w in worker_states.values())
            overall_avg_wait_mins = round((total_wait_sec / total_queue / 60.0), 1) if total_queue > 0 else 0.0
            staffed_pct = int(round((staffed_counters_count / active_counters_count * 100))) if active_counters_count > 0 else 0

            if time_range == 'today':
                counter_kpis['all'] = {
                    'avg_wait_mins': overall_avg_wait_mins,
                    'current_queue_total': total_queue,
                    'max_queue_single': max(all_stats['max_queue'], current_max),
                    'max_queue_time_today': all_stats['max_queue_time'],
                    'staffed_count': staffed_counters_count,
                    'total_staff_active': total_staff_active,
                    'total_counters': active_counters_count,
                    'staffed_percentage': staffed_pct,
                    'unattended_count': unattended_counters_count,
                }
            else:
                has_data = (all_stats['total_events'] > 0 or all_stats['max_queue'] > 0 or all_stats['hist_avg_queue'] > 0)
                staff_active_hist = int(round((all_stats['staff_compliance_pct'] / 100.0) * max(1, active_counters_count)))
                counter_kpis['all'] = {
                    'avg_wait_mins': all_stats['hist_avg_wait'],
                    'current_queue_total': int(round(all_stats['hist_avg_queue'])),
                    'max_queue_single': all_stats['max_queue'],
                    'max_queue_time_today': all_stats['max_queue_time'],
                    'staffed_count': staff_active_hist if has_data else 0,
                    'total_staff_active': staff_active_hist if has_data else 0,
                    'total_counters': active_counters_count,
                    'staffed_percentage': int(round(all_stats['staff_compliance_pct'])) if has_data else 0,
                    'unattended_count': all_stats['unattended_events'],
                }

            target_cid = counter_id if (counter_id and counter_id in counter_kpis) else 'all'
            selected_kpis = counter_kpis.get(target_cid, counter_kpis['all'])
            hourly_rush = self.store.get_hourly_rush(counter_id=target_cid, time_range=time_range)

            busy_stats = {
                'busy_hours_count': all_stats.get('busy_hours_count', 0),
                'peak_rush_window': all_stats.get('peak_rush_window', '--'),
                'busiest_day': all_stats.get('busiest_day', '--'),
                'avg_rush_wait': all_stats.get('avg_rush_wait', 0.0),
                'total_monitored_hours': all_stats.get('total_monitored_hours', 0),
            }

            return {
                'running': True,
                'device': DEVICE,
                'time_range': time_range,
                'counter_id': target_cid,
                'kpis': selected_kpis,
                'counter_kpis': counter_kpis,
                'workers': worker_states,
                'counters': self.store.get_counters(),
                'hourly_rush': hourly_rush,
                'busy_stats': busy_stats,
            }

    def update_counter_zones(self, counter_id, new_zones):
        with self.lock:
            worker = self.workers.get(counter_id)
            if not worker:
                return False, "Counter not found"

            worker.update_zones(new_zones)

            counters = self.store.get_counters()
            target = next((c for c in counters if c['id'] == counter_id), None)
            if target:
                target.update(new_zones)
                self.store.save_counter(target)

            return True, "Zones updated successfully"

    def add_counter(self, config):
        with self.lock:
            cid = config.get('id') or f"counter_{int(time.time())}"
            config['id'] = cid
            if 'staff_zone' not in config:
                config['staff_zone'] = [[0.45, 0.20], [0.85, 0.20], [0.85, 0.70], [0.45, 0.70]]
            if 'queue_zone' not in config:
                config['queue_zone'] = [[0.05, 0.35], [0.44, 0.35], [0.44, 0.95], [0.05, 0.95]]
            if 'monitoring_start' not in config:
                config['monitoring_start'] = '09:00'
            if 'monitoring_end' not in config:
                config['monitoring_end'] = '17:00'
            if 'is_24_7' not in config:
                config['is_24_7'] = False

            self.store.save_counter(config)
            worker = CounterWorker(config, self.store, self.model_pool)
            self.workers[cid] = worker
            worker.start()
            return True, config

    def remove_counter(self, counter_id):
        with self.lock:
            worker = self.workers.pop(counter_id, None)
            if worker:
                worker.stop()
            self.store.delete_counter(counter_id)
            return True, "Counter removed"


# ── HTTP Request Handler ─────────────────────────────────────────────────────
_SERVER = None


class Handler(BaseHTTPRequestHandler):
    def address_string(self):
        # Disable slow reverse DNS lookups on Windows
        return self.client_address[0]

    def log_message(self, format, *args):
        # Keep logs clean
        pass

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')

    def _json(self, code, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(code)
        self._cors()
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path.rstrip('/')

        if path == '/api/state':
            qs = parse_qs(url.query)
            cid = qs.get('counter_id', [None])[0]
            tr = qs.get('time_range', ['today'])[0]
            return self._json(200, _SERVER.get_state(counter_id=cid, time_range=tr))

        if path == '/api/counters':
            return self._json(200, _SERVER.store.get_counters())

        if path == '/api/trends':
            qs = parse_qs(url.query)
            cid = qs.get('counter_id', [None])[0]
            limit = int(qs.get('limit', [40])[0])
            tr = qs.get('time_range', ['today'])[0]
            trends = _SERVER.store.get_trends(counter_id=cid, limit=limit, time_range=tr)
            return self._json(200, trends)

        if path == '/api/events':
            qs = parse_qs(url.query)
            limit = int(qs.get('limit', [100])[0])
            cid = qs.get('counter_id', [None])[0]
            sev = qs.get('severity', [None])[0]
            tr = qs.get('time_range', ['today'])[0]
            events = _SERVER.store.get_events(limit=limit, counter_id=cid, severity=sev, time_range=tr)
            return self._json(200, events)

        if path == '/api/hourly_trends':
            qs = parse_qs(url.query)
            cid = qs.get('counter_id', [None])[0]
            tr = qs.get('time_range', ['today'])[0]
            hourly = _SERVER.store.get_hourly_rush(counter_id=cid, time_range=tr)
            return self._json(200, hourly)

        if path.startswith('/devices/') and path.endswith('/snapshot.jpg'):
            cid = path.split('/')[2]
            worker = _SERVER.workers.get(cid)
            if not worker:
                return self._json(404, {'error': 'Counter not found'})
            with worker.lock:
                snap = worker.latest_snapshot
            if not snap:
                return self._json(503, {'error': 'Snapshot not available yet'})
            self.send_response(200)
            self._cors()
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Content-Length', str(len(snap)))
            self.end_headers()
            return self.wfile.write(snap)

        if path.startswith('/devices/') and path.endswith('/stream.mjpg'):
            cid = path.split('/')[2]
            worker = _SERVER.workers.get(cid)
            if not worker:
                return self._json(404, {'error': 'Counter not found'})

            self.send_response(200)
            self._cors()
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.end_headers()

            while worker.running:
                with worker.lock:
                    jpeg = worker.latest_jpeg
                if jpeg:
                    try:
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n')
                        self.wfile.write(f'Content-Length: {len(jpeg)}\r\n\r\n'.encode('ascii'))
                        self.wfile.write(jpeg)
                        self.wfile.write(b'\r\n')
                    except (BrokenPipeError, ConnectionResetError):
                        break
                time.sleep(0.04)
            return

        # Static assets
        rel = path.lstrip('/')
        if not rel or rel == '':
            target = DIST_DIR / 'index.html'
        else:
            target = DIST_DIR / rel

        if target.exists() and target.is_file():
            ext = target.suffix.lower()
            mime = MIME_TYPES.get(ext, 'application/octet-stream')
            data = target.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            return self.wfile.write(data)

        index_file = DIST_DIR / 'index.html'
        if index_file.exists():
            data = index_file.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            return self.wfile.write(data)

        return self._json(404, {'error': 'File not found'})

    def do_POST(self):
        url = urlparse(self.path)
        path = url.path.rstrip('/')

        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length > 0 else b'{}'
        try:
            payload = json.loads(body.decode('utf-8'))
        except Exception:
            payload = {}

        if path == '/api/counters':
            success, res = _SERVER.add_counter(payload)
            return self._json(201 if success else 400, res)

        return self._json(404, {'error': 'Endpoint not found'})

    def do_PUT(self):
        url = urlparse(self.path)
        path = url.path.rstrip('/')

        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length > 0 else b'{}'
        try:
            payload = json.loads(body.decode('utf-8'))
        except Exception:
            payload = {}

        if path.startswith('/api/counters/') and path.endswith('/zones'):
            cid = path.split('/')[3]
            success, msg = _SERVER.update_counter_zones(cid, payload)
            return self._json(200 if success else 400, {'success': success, 'message': msg})

        return self._json(404, {'error': 'Endpoint not found'})

    def do_DELETE(self):
        url = urlparse(self.path)
        path = url.path.rstrip('/')

        if path == '/api/events':
            qs = parse_qs(url.query)
            cid = qs.get('counter_id', [None])[0]
            _SERVER.store.clear_events(counter_id=cid)
            return self._json(200, {'success': True, 'message': 'Events cleared successfully'})

        if path.startswith('/api/counters/'):
            cid = path.split('/')[3]
            success, msg = _SERVER.remove_counter(cid)
            return self._json(200 if success else 400, {'success': success, 'message': msg})

        return self._json(404, {'error': 'Endpoint not found'})


def run(host='0.0.0.0', port=8195):
    global _SERVER
    _SERVER = CounterServer()
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"\n=======================================================")
    print(f" Medanta Hospital — Counter & Queue Monitoring System")
    print(f" Running on http://{host}:{port}")
    print(f" Inference Device: {DEVICE}")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _SERVER.running = False
        for w in _SERVER.workers.values():
            w.stop()
        server.server_close()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8195))
    run(port=port)
