"""
demo.py — GhostGrid dashboard
Run with:  streamlit run demo.py --server.address 0.0.0.0 --server.port 8501
camera.py must be in the same directory.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import atexit
from datetime import datetime

from camera import CameraProcessor, init_db, DB_PATH

# ── CONFIG ───────────────────────────────────────────────────────────────────
REFRESH_SECS     = 2
COST_PER_KWH     = 15.0
CARBON_INTENSITY = 490
MJPEG_PORT       = int(os.environ.get("GHOSTGRID_MJPEG_PORT", "5050"))
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="GhostGrid", page_icon="👻",
                   layout="wide", initial_sidebar_state="collapsed")

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
  .stApp {{ background-color:{BEIGE}; font-family:'DM Sans',sans-serif; }}
  .block-container {{ padding:0 2rem 0 2rem !important; max-width:100% !important; }}
  #MainMenu,footer,header {{ visibility:hidden; }}
  section[data-testid="stSidebar"] {{ display:none; }}

  .top_banner {{ background:{NAVY}; padding:14px 36px; display:flex; align-items:center; justify-content:space-between; }}
  .brand_logo {{ font-family:'DM Serif Display',serif; font-size:42px; color:{WHITE}; letter-spacing:1px; text-align:center; flex:1; }}
  .brand_logo span {{ color:{SKY}; }}
  .banner_right {{ display:flex; align-items:center; gap:20px; }}
  .live_time {{ font-size:13px; color:{SKY}; letter-spacing:1px; }}
  .status_pill {{ padding:5px 16px; border-radius:20px; font-size:12px; font-weight:600; letter-spacing:0.5px; }}
  .pill_occupied {{ background:rgba(86,124,141,0.3); color:{SKY}; border:1px solid {TEAL}; }}
  .pill_empty    {{ background:rgba(192,57,43,0.2);  color:#F1948A; border:1px solid {RED}; }}
  .pill_loading  {{ background:rgba(200,200,200,0.3); color:#aaa; border:1px solid #ccc; }}

  .page_body {{ padding:28px 36px; }}
  .metric_card {{ background:{WHITE}; border-radius:12px; padding:20px 24px; border:1px solid {SKY}; box-shadow:0 2px 12px rgba(47,65,86,0.07); }}
  .metric_value {{ font-family:'DM Serif Display',serif; font-size:36px; color:{NAVY}; line-height:1; margin-bottom:4px; }}
  .metric_label {{ font-size:10px; color:{TEAL}; letter-spacing:1.5px; text-transform:uppercase; font-weight:500; margin-top:6px; }}
  .metric_icon  {{ font-size:18px; margin-bottom:6px; }}

  .section_title {{ font-size:18px; font-weight:600; letter-spacing:1px; text-transform:uppercase; color:{TEAL}; margin-bottom:10px; padding-bottom:6px; border-bottom:1px solid {SKY}; }}

  .alert_box_safe  {{ background:{WHITE};    border:1.5px solid {TEAL}; border-radius:12px; padding:22px; }}
  .alert_box_waste {{ background:{RED_SOFT}; border:2px   solid {RED};  border-radius:12px; padding:22px; animation:pulsered 1.2s infinite; }}
  @keyframes pulsered {{ 0%,100%{{box-shadow:0 0 0 0 rgba(192,57,43,0.3)}} 50%{{box-shadow:0 0 0 8px rgba(192,57,43,0)}} }}
  .alert_title_safe  {{ font-family:'DM Serif Display',serif; font-size:22px; color:{TEAL}; margin-bottom:6px; }}
  .alert_title_waste {{ font-family:'DM Serif Display',serif; font-size:22px; color:{RED};  margin-bottom:6px; }}
  .alert_detail {{ font-size:12px; color:#555; line-height:2; margin-top:8px; }}

  .stButton > button {{ background:{NAVY} !important; color:{WHITE} !important; border:none !important; border-radius:8px !important; padding:12px 0 !important; font-size:14px !important; font-weight:600 !important; width:100% !important; font-family:'DM Sans',sans-serif !important; transition:background 0.2s !important; }}
  .stButton > button:hover {{ background:{TEAL} !important; }}

  .log_table {{ width:100%; border-collapse:collapse; font-size:12px; background:{WHITE}; border-radius:12px; overflow:hidden; box-shadow:0 2px 12px rgba(47,65,86,0.07); }}
  .log_table th {{ background:{NAVY}; color:{WHITE}; padding:11px 16px; text-align:left; font-size:10px; letter-spacing:1.5px; text-transform:uppercase; cursor:pointer; user-select:none; white-space:nowrap; }}
  .log_table th:hover {{ background:{TEAL}; }}
  .log_table td {{ padding:10px 16px; border-bottom:1px solid {SKY}; color:{NAVY}; vertical-align:middle; }}
  .log_table tr:last-child td {{ border-bottom:none; }}
  .log_table tr:hover td {{ background:{BEIGE}; }}
  .td-device {{ background:{SKY}; color:{NAVY}; padding:2px 8px; border-radius:4px; font-size:10px; font-weight:600; letter-spacing:0.5px; }}
  .td-red    {{ color:{RED}; font-weight:600; }}

  .session_table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  .session_table td {{ padding:7px 4px; border-bottom:1px solid {SKY}; color:{NAVY}; vertical-align:middle; }}
  .session_table tr:last-child td {{ border-bottom:none; }}
  .session_bar_bg   {{ background:#eee; border-radius:4px; height:6px; }}
  .session_bar_fill {{ background:{RED}; border-radius:4px; height:6px; }}

  .detect_chip  {{ display:inline-block; padding:3px 10px; border-radius:20px; font-size:10px; font-weight:600; margin:2px; letter-spacing:0.3px; }}
  .chip_in_use  {{ background:#d4efdf; color:#1e8449; }}
  .chip_wasting {{ background:{RED_SOFT}; color:{RED}; }}
  .chip_off     {{ background:#eee; color:#888; }}

  .camera_box {{ background:{SKY}; border-radius:12px; border:2px solid {SKY}; aspect-ratio:16/9; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:12px; }}
  .empty_state {{ text-align:center; padding:32px; color:{TEAL}; font-size:13px; background:{WHITE}; border-radius:12px; border:1px dashed {SKY}; }}
  .row_divider {{ height:1px; background:{SKY}; margin:28px 0; }}
  [data-testid="column"] {{ padding:0 8px !important; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PROCESSOR — singleton for lifetime of Streamlit server
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_processor():
    db  = init_db()
    cpu = CameraProcessor(db)
    cpu.start()
    def _on_exit():
        cpu.stop()
        cpu.flush_final_to_db()
        db.close()
    atexit.register(_on_exit)
    return cpu

processor = get_processor()


# ═══════════════════════════════════════════════════════════════════════════
#  DB HELPERS  — ttl=REFRESH_SECS so every rerun gets fresh data
# ═══════════════════════════════════════════════════════════════════════════

def _raw_query(sql, params=()):
    """Direct query, no cache — used internally."""
    try:
        return pd.read_sql_query(sql, processor.db_conn, params=params)
    except Exception:
        return pd.DataFrame()

# Build a lookup dict once per refresh (avoids N+1 queries per table render)
@st.cache_data(ttl=REFRESH_SECS)
def get_ref_table() -> dict:
    df = _raw_query("SELECT raw_prefix, display_name FROM DeviceReferenceTable")
    if df.empty:
        return {}
    return dict(zip(df["raw_prefix"], df["display_name"]))

def device_lookup(raw: str, ref: dict) -> str:
    """
    Strip trailing _<digits> track ID, then look up in ref table.
    Falls back to title-casing the prefix if not found.
    """
    if not raw or not isinstance(raw, str):
        return "Unknown"
    # strip trailing _NNN where NNN is purely digits
    import re
    prefix = re.sub(r'_\d+$', '', raw).strip()
    name = ref.get(prefix)
    if name:
        return name
    # partial match — find longest key that is a prefix of raw
    for key in sorted(ref.keys(), key=len, reverse=True):
        if raw.lower().startswith(key.lower()):
            return ref[key]
    return prefix.replace('_', ' ').title()

@st.cache_data(ttl=REFRESH_SECS)
def load_totals():
    df = _raw_query("SELECT total_energy_wasted, carbon_footprint FROM waste_logs")
    if df.empty:
        return 0.0, 0.0, 0.0
    total_energy = df["total_energy_wasted"].sum()
    total_carbon = df["carbon_footprint"].sum() * 1000
    total_cost   = (total_energy / 1000) * COST_PER_KWH
    return total_energy, total_carbon, total_cost

@st.cache_data(ttl=REFRESH_SECS)
def load_log() -> pd.DataFrame:
    df = _raw_query("""
        SELECT id, date AS raw_date, device_id,
               total_time_wasted, total_energy_wasted, carbon_footprint
        FROM waste_logs ORDER BY id DESC
    """)
    if df.empty:
        return pd.DataFrame()

    ref = get_ref_table()

    def split_dt(s):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%a %b %d %H:%M:%S %Y"):
            try:
                dt = datetime.strptime(str(s).strip(), fmt)
                return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S"), dt
            except Exception:
                pass
        return str(s), "", datetime.min

    parsed     = df["raw_date"].apply(split_dt)
    df["Date"] = parsed.apply(lambda x: x[0])
    df["Time"] = parsed.apply(lambda x: x[1])
    df["_dt"]  = parsed.apply(lambda x: x[2])
    df["Device"]      = df["device_id"].apply(lambda x: device_lookup(x, ref))
    df["Wasted (s)"]  = df["total_time_wasted"].round(1)
    df["Energy (Wh)"] = df["total_energy_wasted"].round(4)
    df["Carbon (g)"]  = (df["carbon_footprint"] * 1000).round(4)
    df["Cost (Rs)"]   = ((df["total_energy_wasted"] / 1000) * COST_PER_KWH).round(4)
    return df[["Date","Time","Device","Wasted (s)","Energy (Wh)","Carbon (g)","Cost (Rs)","_dt"]]

@st.cache_data(ttl=REFRESH_SECS)
def load_daily() -> pd.DataFrame:
    """
    Aggregate daily_summary by (date, device prefix) — one row per device
    type per day. Shows true end-of-day totals.
    Fields: Date, Device, Total ON Time (h), Wasted Time (h),
            Total Energy (Wh), Wasted Energy (Wh), Est. Cost (Rs), Carbon (g)
    """
    df = _raw_query("""
        SELECT date,
               device_id,
               total_time_used,
               total_time_wasted,
               total_energy_used,
               total_energy_wasted,
               carbon_footprint
        FROM daily_summary
    """)
    if df.empty:
        return pd.DataFrame()

    ref = get_ref_table()

    import re
    df["prefix"] = df["device_id"].apply(lambda x: re.sub(r'_\d+$', '', str(x)).strip())

    # group by (date, prefix) — sum all track instances together
    agg = df.groupby(["date", "prefix"]).agg(
        t_used   =("total_time_used",     "sum"),
        t_wasted =("total_time_wasted",   "sum"),
        e_used   =("total_energy_used",   "sum"),
        e_wasted =("total_energy_wasted", "sum"),
        carbon   =("carbon_footprint",    "sum"),
    ).reset_index()

    agg["Device"]             = agg["prefix"].apply(lambda x: device_lookup(x, ref))
    agg["Total ON Time (h)"]  = (agg["t_used"] / 3600).round(3)
    agg["Wasted Time (h)"]    = (agg["t_wasted"] / 3600).round(3)
    agg["Total Energy (Wh)"]  = (agg["e_used"] + agg["e_wasted"]).round(4)
    agg["Wasted Energy (Wh)"] = agg["e_wasted"].round(4)
    agg["Est. Cost (Rs)"]     = ((agg["e_used"] + agg["e_wasted"]) / 1000 * COST_PER_KWH).round(4)
    agg["Carbon (g)"]         = (agg["carbon"] * 1000).round(4)

    return agg[["date","Device","Total ON Time (h)","Wasted Time (h)",
                "Total Energy (Wh)","Wasted Energy (Wh)","Est. Cost (Rs)","Carbon (g)"]].sort_values(
        ["date","Device"], ascending=[False, True]
    )


# ═══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

for k, v in [("sort_col","Date"), ("sort_asc",False),
             ("dsort_col","date"), ("dsort_asc",False)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════
#  READ LIVE STATE
# ═══════════════════════════════════════════════════════════════════════════

with processor._lock:
    cam_running     = processor.state["running"]
    person_present  = processor.state["person_present"]
    waste_active    = processor.state["waste_active"]
    detections      = dict(processor.state["detections"])
    total_waste_now = dict(processor.state["total_waste"])
    total_usage_now = dict(processor.state["total_usage"])
    fps             = processor.state["fps"]


# ═══════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════

total_energy, total_carbon, total_cost = load_totals()

if not cam_running:
    room_label, pill_class = "⏳ LOADING", "pill_loading"
elif person_present:
    room_label, pill_class = "🟢 OCCUPIED", "pill_occupied"
elif waste_active:
    room_label, pill_class = "🔴 WASTING", "pill_empty"
else:
    room_label, pill_class = "⚪ EMPTY", "pill_empty"

st.markdown(f"""
<div class="top_banner">
  <div class="brand_logo">👻 Ghost<span>Grid</span></div>
  <div class="banner_right">
    <div class="live_time">⏰ {datetime.now().strftime("%H:%M:%S")}  &nbsp;|&nbsp;  {fps} FPS</div>
    <div class="status_pill {pill_class}">{room_label}</div>
  </div>
</div>
<div class="page_body">
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════

tab_live, tab_history, tab_daily, tab_graphs = st.tabs([
    "📷  Live Monitor", "📋  Incident Log", "📅  Daily Summary", "📊  Analytics"
])


# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — LIVE MONITOR
# ───────────────────────────────────────────────────────────────────────────
with tab_live:

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

        if not cam_running:
            st.markdown(f"""
            <div class="camera_box">
              <div style="font-size:48px">⏳</div>
              <div style="font-size:13px;color:{TEAL};">Model loading, please wait...</div>
            </div>""", unsafe_allow_html=True)
        else:
            import socket
            try:
                host_ip = socket.gethostbyname(socket.gethostname())
            except Exception:
                host_ip = "localhost"
            feed_url = f"http://{host_ip}:{MJPEG_PORT}/feed"
            st.markdown(f'''
            <div style="border-radius:12px;overflow:hidden;background:#000;line-height:0;">
              <img src="{feed_url}"
                   style="width:100%;border-radius:12px;display:block;"
                   onerror="this.style.opacity='0.3'">
            </div>''', unsafe_allow_html=True)

        if detections:
            ref   = get_ref_table()
            chips = ""
            for label, d in detections.items():
                if d["is_on"] and not d["being_used"]:
                    cls, icon = "chip_wasting", "⚠"
                elif d["is_on"] and d["being_used"]:
                    cls, icon = "chip_in_use",  "✅"
                else:
                    cls, icon = "chip_off", "◉"
                name = device_lookup(label, ref)
                chips += f'<span class="detect_chip {cls}">{icon} {name}</span>'
            st.markdown(f'<div style="margin-top:8px">{chips}</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section_title">🚨 Status</div>', unsafe_allow_html=True)

        if not cam_running:
            st.markdown(f"""
            <div class="alert_box_safe">
                <div class="alert_title_safe">⏳ Initialising</div>
                <div class="alert_detail" style="text-align:center;color:{TEAL}">
                    Loading YOLO model and connecting to camera...
                </div>
            </div>""", unsafe_allow_html=True)
        elif waste_active:
            ref      = get_ref_table()
            wasting  = [device_lookup(d["key"], ref) for d in detections.values()
                        if d["is_on"] and not d["being_used"]]
            dev_list = ", ".join(wasting) if wasting else "DEVICE"
            st.markdown(f"""
            <div class="alert_box_waste">
                <div class="alert_title_waste">🚨 Waste Detected!</div>
                <div class="alert_detail">
                    Room is <b style="color:{RED}">EMPTY</b> but device(s) are ON<br>
                    <b>{dev_list}</b><br>
                    ⚡ Grid: <b>{CARBON_INTENSITY} gCO₂/kWh</b>
                </div>
            </div>""", unsafe_allow_html=True)
        elif person_present:
            st.markdown(f"""
            <div class="alert_box_safe">
                <div class="alert_title_safe">✅ All Good</div>
                <div class="alert_detail" style="text-align:center;color:{TEAL}">
                    Room is occupied.<br>No vampire power detected.
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

        # ── This Session — one row per device prefix ───────────────────────
        all_keys = set(list(total_waste_now.keys()) + list(total_usage_now.keys()))
        if all_keys:
            import re as _re
            ref = get_ref_table()
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section_title">⏱ This Session</div>', unsafe_allow_html=True)

            by_prefix = {}
            for key in all_keys:
                prefix = _re.sub(r'_\d+$', '', key).strip()
                w = total_waste_now.get(key, 0)
                u = total_usage_now.get(key, 0)
                if prefix not in by_prefix:
                    by_prefix[prefix] = {"waste": 0.0, "usage": 0.0}
                by_prefix[prefix]["waste"] += w
                by_prefix[prefix]["usage"] += u

            max_waste = max((v["waste"] for v in by_prefix.values()), default=1) or 1
            rows_html = ""
            for prefix, vals in sorted(by_prefix.items(), key=lambda x: -x[1]["waste"]):
                name    = device_lookup(prefix, ref)
                w, u    = vals["waste"], vals["usage"]
                bar_pct = min(int((w / max_waste) * 100), 100)
                rows_html += f"""<tr>
                  <td style="font-weight:600;white-space:nowrap">{name}</td>
                  <td style="color:{RED};font-weight:600;white-space:nowrap">{w:.1f}s wasted</td>
                  <td style="color:#27ae60;white-space:nowrap">{u:.1f}s used</td>
                  <td style="width:80px">
                    <div class="session_bar_bg">
                      <div class="session_bar_fill" style="width:{bar_pct}%"></div>
                    </div>
                  </td>
                </tr>"""
            st.markdown(f'<table class="session_table">{rows_html}</table>',
                        unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — INCIDENT LOG
# ───────────────────────────────────────────────────────────────────────────
with tab_history:
    st.markdown('<div class="section_title">📋 Incident Log</div>', unsafe_allow_html=True)
    st.caption("Each row = one waste episode (device left on while room was empty). Written when person re-enters.")

    log_df  = load_log()
    COLUMNS = ["Date", "Time", "Device", "Wasted (s)", "Energy (Wh)", "Carbon (g)", "Cost (Rs)"]
    ARROWS  = {"asc": "▲", "desc": "▼", "none": "⇅"}

    sort_cols = st.columns(len(COLUMNS))
    for i, col_name in enumerate(COLUMNS):
        with sort_cols[i]:
            is_active = (st.session_state.sort_col == col_name)
            arrow     = (ARROWS["asc"] if st.session_state.sort_asc else ARROWS["desc"]) if is_active else ARROWS["none"]
            if st.button(f"{col_name} {arrow}", key=f"sort_{col_name}"):
                if st.session_state.sort_col == col_name:
                    st.session_state.sort_asc = not st.session_state.sort_asc
                else:
                    st.session_state.sort_col = col_name
                    st.session_state.sort_asc = True
                st.rerun()

    if not log_df.empty:
        sort_key = "_dt" if st.session_state.sort_col in ("Date","Time") else st.session_state.sort_col
        display  = log_df.sort_values(sort_key, ascending=st.session_state.sort_asc)

        rows_html = ""
        for _, row in display.iterrows():
            dev = str(row['Device']) if row['Device'] else "Unknown"
            rows_html += f"""<tr>
                <td>{row['Date']}</td>
                <td>{row['Time']}</td>
                <td><span class="td-device">{dev.upper()}</span></td>
                <td class="td-red">{row['Wasted (s)']}s</td>
                <td class="td-red">{row['Energy (Wh)']} Wh</td>
                <td class="td-red">{row['Carbon (g)']}g</td>
                <td class="td-red">Rs {row['Cost (Rs)']}</td>
            </tr>"""

        st.markdown(f"""
        <table class="log_table">
          <thead><tr>{''.join(f'<th>{c}</th>' for c in COLUMNS)}</tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="empty_state">👀 No incidents yet — written each time a person re-enters the room.</div>',
            unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — DAILY SUMMARY
# ───────────────────────────────────────────────────────────────────────────
with tab_daily:
    st.markdown('<div class="section_title">📅 Daily Summary</div>', unsafe_allow_html=True)
    st.caption("One row per device per day. Totals include both used and wasted time. Updated continuously.")

    daily_df = load_daily()
    DCOLS    = ["date","Device","Total ON Time (h)","Wasted Time (h)",
                "Total Energy (Wh)","Wasted Energy (Wh)","Est. Cost (Rs)","Carbon (g)"]
    DARROWS  = {"asc": "▲", "desc": "▼", "none": "⇅"}

    dsort_cols = st.columns(len(DCOLS))
    for i, col_name in enumerate(DCOLS):
        with dsort_cols[i]:
            is_active = (st.session_state.dsort_col == col_name)
            arrow     = (DARROWS["asc"] if st.session_state.dsort_asc else DARROWS["desc"]) if is_active else DARROWS["none"]
            if st.button(f"{col_name} {arrow}", key=f"dsort_{col_name}"):
                if st.session_state.dsort_col == col_name:
                    st.session_state.dsort_asc = not st.session_state.dsort_asc
                else:
                    st.session_state.dsort_col = col_name
                    st.session_state.dsort_asc = True
                st.rerun()

    if not daily_df.empty:
        try:
            display = daily_df.sort_values(
                st.session_state.dsort_col,
                ascending=st.session_state.dsort_asc
            )
        except KeyError:
            display = daily_df

        rows_html = ""
        for _, row in display.iterrows():
            dev = str(row['Device']) if row['Device'] else "Unknown"
            rows_html += f"""<tr>
                <td>{row['date']}</td>
                <td><span class="td-device">{dev.upper()}</span></td>
                <td>{row['Total ON Time (h)']}h</td>
                <td class="td-red">{row['Wasted Time (h)']}h</td>
                <td>{row['Total Energy (Wh)']} Wh</td>
                <td class="td-red">{row['Wasted Energy (Wh)']} Wh</td>
                <td class="td-red">Rs {row['Est. Cost (Rs)']}</td>
                <td class="td-red">{row['Carbon (g)']}g</td>
            </tr>"""

        st.markdown(f"""
        <table class="log_table">
          <thead><tr>{''.join(f'<th>{c}</th>' for c in DCOLS)}</tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="empty_state">📅 Daily summary will appear here. Data accumulates throughout the day.</div>',
            unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 4 — ANALYTICS  (uses live DB data, refreshes every rerun)
# ───────────────────────────────────────────────────────────────────────────
with tab_graphs:
    st.markdown('<div class="section_title">📊 Analytics</div>', unsafe_allow_html=True)

    log_df = load_log()

    if log_df.empty:
        st.markdown('<div class="empty_state">📈 No data yet — analytics appear after the first waste episode.</div>',
                    unsafe_allow_html=True)
    else:
        chart_df = log_df.sort_values("_dt")

        # 1. Cumulative CO2
        st.markdown("##### Cumulative CO₂ Leaked Over Time")
        chart_df = chart_df.copy()
        chart_df["Cumulative CO₂ (g)"] = chart_df["Carbon (g)"].cumsum()
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=chart_df["_dt"], y=chart_df["Cumulative CO₂ (g)"],
            mode="lines+markers",
            line=dict(color=TEAL, width=2.5),
            marker=dict(color=NAVY, size=6, line=dict(color=WHITE, width=1.5)),
            fill="tozeroy", fillcolor="rgba(86,124,141,0.1)",
        ))
        fig1.update_layout(
            paper_bgcolor=BEIGE, plot_bgcolor=WHITE,
            font=dict(color=NAVY, family="DM Sans"),
            margin=dict(l=40,r=20,t=20,b=40), height=260,
            xaxis=dict(gridcolor=SKY, title="Time",             tickfont=dict(size=10,color=TEAL)),
            yaxis=dict(gridcolor=SKY, title="Cumulative g CO₂", tickfont=dict(size=10,color=TEAL)),
            showlegend=False,
        )
        st.plotly_chart(fig1, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        g1, g2 = st.columns(2)

        with g1:
            st.markdown("##### Wasted Time per Device")
            by_dev = (log_df.groupby("Device")["Wasted (s)"].sum()
                      .reset_index().sort_values("Wasted (s)", ascending=True))
            fig2 = go.Figure(go.Bar(
                x=by_dev["Wasted (s)"], y=by_dev["Device"], orientation='h',
                marker_color=TEAL,
                text=by_dev["Wasted (s)"].apply(lambda v: f"{v:.1f}s"),
                textposition="outside",
            ))
            fig2.update_layout(
                paper_bgcolor=BEIGE, plot_bgcolor=WHITE,
                font=dict(color=NAVY, family="DM Sans"),
                margin=dict(l=10,r=60,t=20,b=40), height=320,
                xaxis=dict(gridcolor=SKY, title="Seconds", tickfont=dict(size=10,color=TEAL)),
                yaxis=dict(gridcolor=SKY, tickfont=dict(size=10,color=TEAL)),
                showlegend=False,
            )
            st.plotly_chart(fig2, width='stretch')

        with g2:
            st.markdown("##### CO₂ Share by Device")
            by_co2 = log_df.groupby("Device")["Carbon (g)"].sum().reset_index()
            by_co2 = by_co2[by_co2["Carbon (g)"] > 0]
            if not by_co2.empty:
                fig3 = go.Figure(go.Pie(
                    labels=by_co2["Device"], values=by_co2["Carbon (g)"],
                    hole=0.42, marker=dict(colors=px.colors.sequential.Teal),
                    textinfo="label+percent", textfont_size=11,
                ))
                fig3.update_layout(
                    paper_bgcolor=BEIGE, font=dict(color=NAVY,family="DM Sans"),
                    margin=dict(l=20,r=20,t=20,b=20), height=300, showlegend=False,
                )
                st.plotly_chart(fig3, width='stretch')

        st.markdown("##### Daily Wasted vs Used Energy (Wh)")
        daily_chart = load_daily()
        if not daily_chart.empty:
            dc = daily_chart.groupby("date").agg(
                wasted=("Wasted Energy (Wh)", "sum"),
                used  =("Total Energy (Wh)",  "sum"),
            ).reset_index()
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(name="Wasted", x=dc["date"], y=dc["wasted"], marker_color=RED))
            fig4.add_trace(go.Bar(name="Used",   x=dc["date"], y=dc["used"],   marker_color=TEAL))
            fig4.update_layout(
                barmode="group",
                paper_bgcolor=BEIGE, plot_bgcolor=WHITE,
                font=dict(color=NAVY, family="DM Sans"),
                margin=dict(l=40,r=20,t=20,b=60), height=300,
                xaxis=dict(gridcolor=SKY, tickfont=dict(size=10,color=TEAL)),
                yaxis=dict(gridcolor=SKY, title="Wh", tickfont=dict(size=10,color=TEAL)),
                legend=dict(bgcolor=BEIGE, font=dict(color=NAVY)),
            )
            st.plotly_chart(fig4, width='stretch')

        st.markdown("##### Cumulative Cost per Device (Rs)")
        by_cost = (log_df.groupby("Device")["Cost (Rs)"].sum()
                   .reset_index().sort_values("Cost (Rs)", ascending=False))
        fig5 = go.Figure(go.Bar(
            x=by_cost["Device"], y=by_cost["Cost (Rs)"],
            marker_color=[TEAL if i%2==0 else NAVY for i in range(len(by_cost))],
            text=by_cost["Cost (Rs)"].apply(lambda v: f"Rs {v:.4f}"),
            textposition="outside",
        ))
        fig5.update_layout(
            paper_bgcolor=BEIGE, plot_bgcolor=WHITE,
            font=dict(color=NAVY, family="DM Sans"),
            margin=dict(l=40,r=20,t=20,b=80), height=300,
            xaxis=dict(gridcolor=SKY, tickfont=dict(size=10,color=TEAL), tickangle=-30),
            yaxis=dict(gridcolor=SKY, title="Rs", tickfont=dict(size=10,color=TEAL)),
            showlegend=False,
        )
        st.plotly_chart(fig5, width='stretch')


st.markdown("</div>", unsafe_allow_html=True)
time.sleep(REFRESH_SECS)
st.rerun()