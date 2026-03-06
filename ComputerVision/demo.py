# ghostgrid — live dashboard
# connects to: camera.py (shared frame file) + energy_data.db

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sqlite3
import time
import os
import numpy as np
import cv2
from datetime import datetime

# ── CONFIG ──────────────────────────────────────────────────────────────────
DB_PATH        = os.environ.get("GHOSTGRID_DB", "energy_data.db")
FRAME_PATH     = os.environ.get("GHOSTGRID_FRAME", "ghostgrid_frame.jpg")   # camera.py writes here
REFRESH_SECS   = 2
COST_PER_KWH   = 15.0   # Rs per kWh (Nepal estimate)
CARBON_INTENSITY = 490  # gCO2 per kWh (Nepal grid)
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GhostGrid",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── THEME ───────────────────────────────────────────────────────────────────
NAVY     = "#2F4156"
TEAL     = "#567C8D"
SKY      = "#C8D9E6"
BEIGE    = "#F5EFEB"
WHITE    = "#FFFFFF"
RED      = "#C0392B"
RED_SOFT = "#FADBD8"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

  .stApp {{ background-color: {BEIGE}; font-family: 'DM Sans', sans-serif; }}
  .block-container {{ padding: 0 2rem 0 2rem !important; max-width: 100% !important; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  section[data-testid="stSidebar"] {{ display: none; }}

  .top_banner {{
    background: {NAVY}; padding: 14px 36px;
    display: flex; align-items: center; justify-content: space-between;
  }}
  .brand_logo {{ font-family:'DM Serif Display',serif; font-size:42px; color:{WHITE}; letter-spacing:1px; text-align:center; flex:1; }}
  .brand_logo span {{ color:{SKY}; }}
  .banner_right {{ display:flex; align-items:center; gap:20px; }}
  .live_time {{ font-size:13px; color:{SKY}; letter-spacing:1px; }}
  .status_pill {{ padding:5px 16px; border-radius:20px; font-size:12px; font-weight:600; letter-spacing:0.5px; }}
  .pill_occupied {{ background:rgba(86,124,141,0.3); color:{SKY}; border:1px solid {TEAL}; }}
  .pill_empty    {{ background:rgba(192,57,43,0.2);  color:#F1948A;  border:1px solid {RED}; }}

  .page_body {{ padding: 28px 36px; }}

  .metric_card {{
    background:{WHITE}; border-radius:12px; padding:20px 24px;
    border:1px solid {SKY}; box-shadow:0 2px 12px rgba(47,65,86,0.07);
  }}
  .metric_value {{ font-family:'DM Serif Display',serif; font-size:36px; color:{NAVY}; line-height:1; margin-bottom:4px; }}
  .metric_label {{ font-size:10px; color:{TEAL}; letter-spacing:1.5px; text-transform:uppercase; font-weight:500; margin-top:6px; }}
  .metric_icon  {{ font-size:18px; margin-bottom:6px; }}

  .section_title {{
    font-size:18px; font-weight:600; letter-spacing:1px; text-transform:uppercase;
    color:{TEAL}; margin-bottom:10px; padding-bottom:6px; border-bottom:1px solid {SKY};
  }}

  .alert_box_safe  {{ background:{WHITE};    border:1.5px solid {TEAL}; border-radius:12px; padding:22px; }}
  .alert_box_waste {{ background:{RED_SOFT}; border:2px   solid {RED};  border-radius:12px; padding:22px; animation:pulsered 1.2s infinite; }}
  @keyframes pulsered {{
    0%,100% {{ box-shadow:0 0 0 0 rgba(192,57,43,0.3); }}
    50%      {{ box-shadow:0 0 0 8px rgba(192,57,43,0); }}
  }}
  .alert_title_safe  {{ font-family:'DM Serif Display',serif; font-size:22px; color:{TEAL}; margin-bottom:6px; }}
  .alert_title_waste {{ font-family:'DM Serif Display',serif; font-size:22px; color:{RED};  margin-bottom:6px; }}
  .alert_detail {{ font-size:12px; color:#555; line-height:2; margin-top:8px; }}

  .stButton > button {{
    background:{NAVY} !important; color:{WHITE} !important; border:none !important;
    border-radius:8px !important; padding:12px 0 !important; font-size:14px !important;
    font-weight:600 !important; width:100% !important;
    font-family:'DM Sans',sans-serif !important; transition:background 0.2s !important;
  }}
  .stButton > button:hover {{ background:{TEAL} !important; }}

  /* sortable log table */
  .log_table {{ width:100%; border-collapse:collapse; font-size:12px; background:{WHITE}; border-radius:12px; overflow:hidden; box-shadow:0 2px 12px rgba(47,65,86,0.07); }}
  .log_table th {{
    background:{NAVY}; color:{WHITE}; padding:11px 16px;
    text-align:left; font-size:10px; letter-spacing:1.5px; text-transform:uppercase;
    cursor:pointer; user-select:none; white-space:nowrap;
  }}
  .log_table th:hover {{ background:{TEAL}; }}
  .log_table th .sort_arrow {{ margin-left:4px; opacity:0.6; }}
  .log_table th.active_sort .sort_arrow {{ opacity:1; }}
  .log_table td {{ padding:10px 16px; border-bottom:1px solid {SKY}; color:{NAVY}; }}
  .log_table tr:last-child td {{ border-bottom:none; }}
  .log_table tr:hover td {{ background:{BEIGE}; }}
  .td-device {{ background:{SKY}; color:{NAVY}; padding:2px 8px; border-radius:4px; font-size:10px; font-weight:600; letter-spacing:0.5px; }}
  .td-red    {{ color:{RED}; font-weight:600; }}

  .camera_box {{
    background:{SKY}; border-radius:12px; border:2px solid {SKY};
    aspect-ratio:4/3; display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:10px; position:relative; overflow:hidden;
  }}
  .camera_box_waste {{ background:#f0dada; border-color:{RED}; }}
  .camera_badge {{
    position:absolute; top:12px; left:12px;
    background:rgba(47,65,86,0.8); color:{SKY};
    font-size:10px; font-weight:600; padding:4px 10px; border-radius:20px; letter-spacing:1px;
  }}
  .detect_box_green {{
    border:2px solid #27AE60; background:rgba(39,174,96,0.08);
    padding:8px 20px; border-radius:6px; font-size:11px; color:#27AE60; font-weight:600;
  }}
  .detect_box_red {{
    border:2px solid {RED}; background:rgba(192,57,43,0.08);
    padding:8px 20px; border-radius:6px; font-size:11px; color:{RED}; font-weight:600;
    animation:pulsered 1.2s infinite;
  }}
  .detect_item {{
    display:inline-block; background:{SKY}; color:{NAVY};
    padding:3px 10px; border-radius:20px; font-size:10px; font-weight:600; margin:2px;
  }}
  .detect_item_waste {{
    background:#fadbd8; color:{RED};
  }}

  .empty_state {{ text-align:center; padding:32px; color:{TEAL}; font-size:13px; background:{WHITE}; border-radius:12px; border:1px dashed {SKY}; }}
  .row_divider {{ height:1px; background:{SKY}; margin:28px 0; }}
  .toast_msg   {{ background:{TEAL}; color:{WHITE}; padding:12px 20px; border-radius:8px; font-size:13px; font-weight:500; text-align:center; margin-top:8px; }}
  [data-testid="column"] {{ padding: 0 8px !important; }}
</style>
""", unsafe_allow_html=True)


# ── SESSION STATE ────────────────────────────────────────────────────────────
def _init(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_init("sort_col",   "date")
_init("sort_asc",   False)
_init("tab",        "live")


# ── DB HELPERS ───────────────────────────────────────────────────────────────
@st.cache_resource
def get_conn():
    # check-same-thread=False so streamlit can reuse across reruns
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def query(sql, params=()):
    try:
        conn = get_conn()
        return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()

def device_lookup(raw: str) -> str:
    """Strip trailing _NNN and look up display name."""
    prefix = raw.rsplit('_', 1)[0] if '_' in raw else raw
    df = query("SELECT display_name FROM DeviceReferenceTable WHERE raw_prefix = ?", (prefix,))
    if not df.empty:
        return df.iloc[0, 0]
    # fallback: title-case the raw prefix
    return prefix.replace('_', ' ').title()


# ── LOAD DB STATS ────────────────────────────────────────────────────────────
def load_totals():
    df = query("SELECT total_energy_wasted, carbon_footprint FROM waste_logs")
    if df.empty:
        return 0.0, 0.0, 0.0
    total_energy = df["total_energy_wasted"].sum()           # Wh
    total_carbon = df["carbon_footprint"].sum() * 1000       # g  (stored as kg)
    total_cost   = (total_energy / 1000) * COST_PER_KWH     # Rs
    return total_energy, total_carbon, total_cost

def load_log() -> pd.DataFrame:
    df = query("""
        SELECT
            id,
            date             AS raw_date,
            device_id,
            total_time_wasted,
            total_energy_wasted,
            carbon_footprint
        FROM waste_logs
        ORDER BY id DESC
    """)
    if df.empty:
        return df

    # parse date → separate Date and Time columns
    def split_dt(s):
        try:
            # ctime format: "Mon Jan  1 00:00:00 2025"
            dt = datetime.strptime(s.strip(), "%a %b %d %H:%M:%S %Y")
            return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S"), dt
        except Exception:
            return s, "", datetime.min

    parsed     = df["raw_date"].apply(split_dt)
    df["Date"] = parsed.apply(lambda x: x[0])
    df["Time"] = parsed.apply(lambda x: x[1])
    df["_dt"]  = parsed.apply(lambda x: x[2])

    df["Device"]       = df["device_id"].apply(device_lookup)
    df["Wasted (s)"]   = df["total_time_wasted"].round(1)
    df["Energy (Wh)"]  = df["total_energy_wasted"].round(4)
    df["Carbon (g)"]   = (df["carbon_footprint"] * 1000).round(4)
    df["Cost (Rs)"]    = ((df["total_energy_wasted"] / 1000) * COST_PER_KWH).round(4)

    return df[["Date","Time","Device","Wasted (s)","Energy (Wh)","Carbon (g)","Cost (Rs)","_dt"]]


# ── LIVE CAMERA FRAME ─────────────────────────────────────────────────────
def read_frame():
    """Return latest annotated frame as numpy array, or None."""
    if not os.path.exists(FRAME_PATH):
        return None
    try:
        img = cv2.imread(FRAME_PATH)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return None

def parse_detections_from_frame(frame_rgb):
    """
    Very lightweight heuristic: look for green/red/grey bounding-box colours
    drawn by camera.py and return rough labels.
    This is a best-effort visual parser — camera.py is the authoritative source.
    Returns: person_present (bool), wasting_devices (list), in_use_devices (list)
    """
    if frame_rgb is None:
        return False, [], []
    # convert to BGR for colour checks
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # pure green mask (person / in-use)  H~60  in OpenCV (0-180 scale)
    green_mask = cv2.inRange(hsv, (55, 180, 180), (65, 255, 255))
    # pure red mask (wasting)
    red_mask1  = cv2.inRange(hsv, (0,  150, 150), (5,  255, 255))
    red_mask2  = cv2.inRange(hsv, (175,150, 150), (180,255, 255))
    red_mask   = red_mask1 | red_mask2

    person_present = cv2.countNonZero(green_mask) > 500
    wasting        = cv2.countNonZero(red_mask)   > 200
    return person_present, wasting, []


# ── HEADER ───────────────────────────────────────────────────────────────────
total_energy, total_carbon, total_cost = load_totals()
log_df = load_log()

# Detect live state for pill
frame_rgb = read_frame()
person_present, waste_active, _ = parse_detections_from_frame(frame_rgb)
room_label = "🟢 OCCUPIED" if person_present else ("🔴 WASTING" if waste_active else "⚪ EMPTY")
pill_class = "pill_occupied" if person_present else "pill_empty"

st.markdown(f"""
<div class="top_banner">
  <div class="brand_logo">👻 Ghost<span>Grid</span></div>
  <div class="banner_right">
    <div class="live_time">⏰ {datetime.now().strftime("%H:%M:%S")}</div>
    <div class="status_pill {pill_class}">{room_label}</div>
  </div>
</div>
<div class="page_body">
""", unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────
tab_live, tab_history, tab_graphs = st.tabs(["📷  Live Monitor", "📋  Incident Log", "📊  Analytics"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:

    # metric cards
    st.markdown('<div class="section_title">Session Totals (All-time from DB)</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""<div class="metric_card">
            <div class="metric_icon">⚡</div>
            <div class="metric_value">{CARBON_INTENSITY}</div>
            <div class="metric_label">gCO₂ / kWh — Nepal Grid</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric_card">
            <div class="metric_icon">💨</div>
            <div class="metric_value">{total_carbon:.1f}g</div>
            <div class="metric_label">Total CO₂ from Waste</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric_card">
            <div class="metric_icon">💸</div>
            <div class="metric_value">Rs {total_cost:.2f}</div>
            <div class="metric_label">Money Wasted (All Sessions)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_feed, col_right = st.columns([6, 4], gap="large")

    with col_feed:
        st.markdown('<div class="section_title">📷 Live Sentinel Feed</div>', unsafe_allow_html=True)

        if frame_rgb is not None:
            # Show real camera frame with annotations from camera.py
            st.image(frame_rgb, use_container_width=True,
                     caption=f"Last frame — {datetime.now().strftime('%H:%M:%S')}")

            # overlay badge via HTML (appears below the image in streamlit)
            if person_present:
                st.markdown('<div class="detect_box_green">✅ PERSON DETECTED — Monitoring</div>',
                            unsafe_allow_html=True)
            elif waste_active:
                st.markdown('<div class="detect_box_red">⚠ DEVICE ON — ROOM EMPTY — Vampire Power!</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="color:{TEAL};font-size:12px;margin-top:4px;">Room empty — no active waste</div>',
                            unsafe_allow_html=True)
        else:
            # camera.py not running / frame not yet written
            st.markdown(f"""
            <div class="camera_box">
              <div class="camera_badge">WAITING FOR FEED</div>
              <div style="font-size:48px">📡</div>
              <div style="font-size:13px;color:{TEAL};">camera.py not running or frame not found</div>
              <div style="font-size:11px;color:{TEAL};">Expected: <code>{FRAME_PATH}</code></div>
            </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section_title">🚨 Status</div>', unsafe_allow_html=True)

        if waste_active:
            # get last wasted device from DB
            last_dev = log_df.iloc[0]["Device"] if not log_df.empty else "DEVICE"
            st.markdown(f"""
            <div class="alert_box_waste">
                <div class="alert_title_waste">🚨 Waste Detected!</div>
                <div class="alert_detail">
                    Room is <b style="color:{RED}">EMPTY</b> but device is ON<br>
                    Last logged device: <b>{last_dev}</b><br>
                    ⚡ Grid: <b>{CARBON_INTENSITY} gCO₂/kWh</b>
                </div>
            </div>""", unsafe_allow_html=True)

        elif person_present:
            st.markdown(f"""
            <div class="alert_box_safe">
                <div class="alert_title_safe">✅ All Good</div>
                <div class="alert_detail" style="text-align:center;color:{TEAL}">
                    Room is occupied.<br>No vampire power detected.<br>Monitoring continuously...
                </div>
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="alert_box_safe">
                <div class="alert_title_safe">👀 Watching</div>
                <div class="alert_detail" style="text-align:center;color:{TEAL}">
                    Room empty — no devices on.<br>Nothing wasted right now.
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── recent detections from DB ────────────────────────────────────────
        st.markdown('<div class="section_title">🔍 Recent Devices Seen</div>', unsafe_allow_html=True)
        recent = query("""
            SELECT device_id, MAX(date) as last_seen, SUM(total_time_wasted) as tw
            FROM waste_logs
            GROUP BY device_id
            ORDER BY last_seen DESC
            LIMIT 8
        """)
        if not recent.empty:
            chips = ""
            for _, row in recent.iterrows():
                name  = device_lookup(row["device_id"])
                waste = row["tw"]
                cls   = "detect_item_waste" if waste > 30 else ""
                chips += f'<span class="detect_item {cls}">{name}</span> '
            st.markdown(f'<div style="margin-top:8px">{chips}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="font-size:12px;color:#999">No data yet.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INCIDENT LOG  (sortable)
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown('<div class="section_title">📋 Incident Log</div>', unsafe_allow_html=True)

    COLUMNS = ["Date", "Time", "Device", "Wasted (s)", "Energy (Wh)", "Carbon (g)", "Cost (Rs)"]
    SORT_ARROWS = {"asc": "▲", "desc": "▼", "none": "⇅"}

    # ── sort controls ────────────────────────────────────────────────────────
    sort_cols = st.columns(len(COLUMNS))
    for i, col_name in enumerate(COLUMNS):
        with sort_cols[i]:
            is_active = (st.session_state.sort_col == col_name)
            arrow = (SORT_ARROWS["asc"] if st.session_state.sort_asc else SORT_ARROWS["desc"]) if is_active else SORT_ARROWS["none"]
            label = f"{col_name} {arrow}"
            if st.button(label, key=f"sort_{col_name}"):
                if st.session_state.sort_col == col_name:
                    st.session_state.sort_asc = not st.session_state.sort_asc
                else:
                    st.session_state.sort_col = col_name
                    st.session_state.sort_asc = True
                st.rerun()

    if not log_df.empty:
        display = log_df.copy()

        # sort
        col = st.session_state.sort_col
        if col in display.columns:
            sort_key = "_dt" if col in ("Date", "Time") else col
            display = display.sort_values(sort_key, ascending=st.session_state.sort_asc)

        rows_html = ""
        for _, row in display.iterrows():
            rows_html += f"""<tr>
                <td>{row['Date']}</td>
                <td>{row['Time']}</td>
                <td><span class="td-device">{row['Device'].upper()}</span></td>
                <td class="td-red">{row['Wasted (s)']}s</td>
                <td class="td-red">{row['Energy (Wh)']} Wh</td>
                <td class="td-red">{row['Carbon (g)']}g</td>
                <td class="td-red">Rs {row['Cost (Rs)']}</td>
            </tr>"""

        st.markdown(f"""
        <table class="log_table">
          <thead><tr>
            {''.join(f'<th>{c}</th>' for c in COLUMNS)}
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="empty_state">👀 No incidents yet — run camera.py to start logging...</div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS / GRAPHS
# ══════════════════════════════════════════════════════════════════════════════
with tab_graphs:
    st.markdown('<div class="section_title">📊 Analytics</div>', unsafe_allow_html=True)

    if log_df.empty:
        st.markdown(
            '<div class="empty_state">📈 No data yet — run camera.py to populate the DB.</div>',
            unsafe_allow_html=True
        )
    else:
        chart_df = log_df.copy().sort_values("_dt")

        # ── 1. Cumulative Carbon over time ─────────────────────────────────
        st.markdown("##### Cumulative CO₂ Leaked Over Time")
        chart_df["Cumulative CO₂ (g)"] = chart_df["Carbon (g)"].cumsum()

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=chart_df["_dt"],
            y=chart_df["Cumulative CO₂ (g)"],
            mode="lines+markers",
            line=dict(color=TEAL, width=2.5),
            marker=dict(color=NAVY, size=6, line=dict(color=WHITE, width=1.5)),
            fill="tozeroy",
            fillcolor="rgba(86,124,141,0.1)",
        ))
        fig1.update_layout(
            paper_bgcolor=BEIGE, plot_bgcolor=WHITE,
            font=dict(color=NAVY, family="DM Sans"),
            margin=dict(l=40, r=20, t=20, b=40), height=260,
            xaxis=dict(gridcolor=SKY, title="Time",           tickfont=dict(size=10, color=TEAL), linecolor=SKY),
            yaxis=dict(gridcolor=SKY, title="Cumulative g CO₂", tickfont=dict(size=10, color=TEAL), linecolor=SKY),
            showlegend=False,
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        g1, g2 = st.columns(2)

        # ── 2. Wasted time by device ───────────────────────────────────────
        with g1:
            st.markdown("##### Wasted Time per Device")
            by_dev = (
                log_df.groupby("Device")["Wasted (s)"]
                .sum()
                .reset_index()
                .sort_values("Wasted (s)", ascending=True)
            )
            fig2 = go.Figure(go.Bar(
                x=by_dev["Wasted (s)"],
                y=by_dev["Device"],
                orientation='h',
                marker_color=TEAL,
                text=by_dev["Wasted (s)"].apply(lambda v: f"{v:.1f}s"),
                textposition="outside",
            ))
            fig2.update_layout(
                paper_bgcolor=BEIGE, plot_bgcolor=WHITE,
                font=dict(color=NAVY, family="DM Sans"),
                margin=dict(l=10, r=40, t=20, b=40), height=300,
                xaxis=dict(gridcolor=SKY, title="Seconds", tickfont=dict(size=10, color=TEAL)),
                yaxis=dict(gridcolor=SKY, tickfont=dict(size=10, color=TEAL)),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── 3. Carbon share pie ────────────────────────────────────────────
        with g2:
            st.markdown("##### CO₂ Share by Device")
            by_co2 = log_df.groupby("Device")["Carbon (g)"].sum().reset_index()
            fig3 = go.Figure(go.Pie(
                labels=by_co2["Device"],
                values=by_co2["Carbon (g)"],
                hole=0.42,
                marker=dict(colors=px.colors.sequential.Teal),
                textinfo="label+percent",
                textfont_size=11,
            ))
            fig3.update_layout(
                paper_bgcolor=BEIGE,
                font=dict(color=NAVY, family="DM Sans"),
                margin=dict(l=20, r=20, t=20, b=20), height=300,
                showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── 4. Energy wasted per day bar ───────────────────────────────────
        st.markdown("##### Daily Energy Wasted (Wh)")
        daily = log_df.groupby("Date")["Energy (Wh)"].sum().reset_index()
        fig4 = go.Figure(go.Bar(
            x=daily["Date"],
            y=daily["Energy (Wh)"],
            marker_color=NAVY,
            text=daily["Energy (Wh)"].apply(lambda v: f"{v:.3f} Wh"),
            textposition="outside",
        ))
        fig4.update_layout(
            paper_bgcolor=BEIGE, plot_bgcolor=WHITE,
            font=dict(color=NAVY, family="DM Sans"),
            margin=dict(l=40, r=20, t=20, b=60), height=280,
            xaxis=dict(gridcolor=SKY, title="Date", tickfont=dict(size=10, color=TEAL), linecolor=SKY),
            yaxis=dict(gridcolor=SKY, title="Energy (Wh)", tickfont=dict(size=10, color=TEAL), linecolor=SKY),
            showlegend=False,
        )
        st.plotly_chart(fig4, use_container_width=True)

        # ── 5. Cost per device ─────────────────────────────────────────────
        st.markdown("##### Cumulative Cost per Device (Rs)")
        by_cost = (
            log_df.groupby("Device")["Cost (Rs)"]
            .sum()
            .reset_index()
            .sort_values("Cost (Rs)", ascending=False)
        )
        fig5 = go.Figure(go.Bar(
            x=by_cost["Device"],
            y=by_cost["Cost (Rs)"],
            marker_color=[TEAL if i % 2 == 0 else NAVY for i in range(len(by_cost))],
            text=by_cost["Cost (Rs)"].apply(lambda v: f"Rs {v:.4f}"),
            textposition="outside",
        ))
        fig5.update_layout(
            paper_bgcolor=BEIGE, plot_bgcolor=WHITE,
            font=dict(color=NAVY, family="DM Sans"),
            margin=dict(l=40, r=20, t=20, b=80), height=300,
            xaxis=dict(gridcolor=SKY, tickfont=dict(size=10, color=TEAL), tickangle=-30),
            yaxis=dict(gridcolor=SKY, title="Rs", tickfont=dict(size=10, color=TEAL)),
            showlegend=False,
        )
        st.plotly_chart(fig5, use_container_width=True)


st.markdown("</div>", unsafe_allow_html=True)

# ── AUTO-REFRESH ─────────────────────────────────────────────────────────────
time.sleep(REFRESH_SECS)
st.rerun()