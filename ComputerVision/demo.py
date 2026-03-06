"""
demo.py — GhostGrid dashboard
Run with:  streamlit run demo.py
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

# ── import camera module directly ────────────────────────────────────────────
from camera import CameraProcessor, init_db, DB_PATH

# ── CONFIG ───────────────────────────────────────────────────────────────────
REFRESH_SECS     = 2
COST_PER_KWH     = 15.0    # Rs per kWh (Nepal)
CARBON_INTENSITY = 490     # gCO2 per kWh (Nepal grid)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GhostGrid",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── THEME ─────────────────────────────────────────────────────────────────────
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
  .pill_loading  {{ background:rgba(200,200,200,0.3); color:#aaa; border:1px solid #ccc; }}

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
  .alert_box_waste {{ background:{RED_SOFT}; border:2px solid {RED};    border-radius:12px; padding:22px; animation:pulsered 1.2s infinite; }}
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

  .log_table {{ width:100%; border-collapse:collapse; font-size:12px; background:{WHITE}; border-radius:12px; overflow:hidden; box-shadow:0 2px 12px rgba(47,65,86,0.07); }}
  .log_table th {{
    background:{NAVY}; color:{WHITE}; padding:11px 16px;
    text-align:left; font-size:10px; letter-spacing:1.5px; text-transform:uppercase;
    cursor:pointer; user-select:none; white-space:nowrap;
  }}
  .log_table th:hover {{ background:{TEAL}; }}
  .log_table td {{ padding:10px 16px; border-bottom:1px solid {SKY}; color:{NAVY}; }}
  .log_table tr:last-child td {{ border-bottom:none; }}
  .log_table tr:hover td {{ background:{BEIGE}; }}
  .td-device {{ background:{SKY}; color:{NAVY}; padding:2px 8px; border-radius:4px; font-size:10px; font-weight:600; letter-spacing:0.5px; }}
  .td-red    {{ color:{RED}; font-weight:600; }}

  .detect_chip {{
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:10px; font-weight:600; margin:2px; letter-spacing:0.3px;
  }}
  .chip_in_use  {{ background:#d4efdf; color:#1e8449; }}
  .chip_wasting {{ background:{RED_SOFT}; color:{RED}; }}
  .chip_off     {{ background:#eee; color:#888; }}

  .camera_box {{
    background:{SKY}; border-radius:12px; border:2px solid {SKY};
    aspect-ratio:16/9; display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:12px;
  }}

  .empty_state {{ text-align:center; padding:32px; color:{TEAL}; font-size:13px; background:{WHITE}; border-radius:12px; border:1px dashed {SKY}; }}
  .row_divider {{ height:1px; background:{SKY}; margin:28px 0; }}
  [data-testid="column"] {{ padding: 0 8px !important; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SPIN UP CAMERA PROCESSOR ONCE (cached for the lifetime of the server)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_processor():
    db  = init_db()
    cpu = CameraProcessor(db)
    cpu.start()

    # flush DB on server shutdown
    def _on_exit():
        cpu.stop()
        cpu.flush_to_db()
        db.close()
    atexit.register(_on_exit)

    return cpu

processor = get_processor()


# ═══════════════════════════════════════════════════════════════════════════
#  DB QUERY HELPERS  (use same connection from processor)
# ═══════════════════════════════════════════════════════════════════════════

def query(sql, params=()):
    try:
        return pd.read_sql_query(sql, processor.db_conn, params=params)
    except Exception:
        return pd.DataFrame()

def device_lookup(raw: str) -> str:
    prefix = raw.rsplit('_', 1)[0] if '_' in raw else raw
    df = query("SELECT display_name FROM DeviceReferenceTable WHERE raw_prefix = ?", (prefix,))
    return df.iloc[0, 0] if not df.empty else prefix.replace('_', ' ').title()

def load_totals():
    df = query("SELECT total_energy_wasted, carbon_footprint FROM waste_logs")
    if df.empty:
        return 0.0, 0.0, 0.0
    total_energy = df["total_energy_wasted"].sum()
    total_carbon = df["carbon_footprint"].sum() * 1000
    total_cost   = (total_energy / 1000) * COST_PER_KWH
    return total_energy, total_carbon, total_cost

def load_log() -> pd.DataFrame:
    df = query("""
        SELECT id, date AS raw_date, device_id,
               total_time_wasted, total_energy_wasted, carbon_footprint
        FROM waste_logs ORDER BY id DESC
    """)
    if df.empty:
        return df

    def split_dt(s):
        try:
            dt = datetime.strptime(s.strip(), "%a %b %d %H:%M:%S %Y")
            return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S"), dt
        except Exception:
            return s, "", datetime.min

    parsed     = df["raw_date"].apply(split_dt)
    df["Date"] = parsed.apply(lambda x: x[0])
    df["Time"] = parsed.apply(lambda x: x[1])
    df["_dt"]  = parsed.apply(lambda x: x[2])

    df["Device"]      = df["device_id"].apply(device_lookup)
    df["Wasted (s)"]  = df["total_time_wasted"].round(1)
    df["Energy (Wh)"] = df["total_energy_wasted"].round(4)
    df["Carbon (g)"]  = (df["carbon_footprint"] * 1000).round(4)
    df["Cost (Rs)"]   = ((df["total_energy_wasted"] / 1000) * COST_PER_KWH).round(4)

    return df[["Date","Time","Device","Wasted (s)","Energy (Wh)","Carbon (g)","Cost (Rs)","_dt"]]


# ═══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

if "sort_col" not in st.session_state: st.session_state.sort_col = "Date"
if "sort_asc" not in st.session_state: st.session_state.sort_asc = False


# ═══════════════════════════════════════════════════════════════════════════
#  READ LIVE STATE FROM PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

with processor._lock:
    cam_running     = processor.state["running"]
    person_present  = processor.state["person_present"]
    waste_active    = processor.state["waste_active"]
    detections      = list(processor.state["detections"])
    total_waste_now = dict(processor.state["total_waste"])
    fps             = processor.state["fps"]
    raw_frame       = processor.latest_frame   # BGR ndarray or None


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

tab_live, tab_history, tab_graphs = st.tabs(["📷  Live Monitor", "📋  Incident Log", "📊  Analytics"])


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

        MJPEG_PORT = int(os.environ.get("GHOSTGRID_MJPEG_PORT", "5050"))
        if not cam_running:
            st.markdown(f"""
            <div class="camera_box">
              <div style="font-size:48px">⏳</div>
              <div style="font-size:13px;color:{TEAL};">Model loading, please wait...</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="border-radius:12px;overflow:hidden;background:#000;line-height:0;">
              <img src="http://localhost:{MJPEG_PORT}/feed"
                   style="width:100%;border-radius:12px;display:block;"
                   onerror="this.style.display='none'">
            </div>""", unsafe_allow_html=True)

        # live detection chips below the frame
        if detections:
            chips = ""
            for d in detections:
                if d["being_used"]:
                    cls, icon = "chip_in_use", "✅"
                elif d["is_on"] and not d["being_used"]:
                    cls, icon = "chip_wasting", "⚠"
                else:
                    cls, icon = "chip_off", "◉"
                name = device_lookup(d["key"])
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
            wasting_devs = [device_lookup(d["key"]) for d in detections
                            if d["is_on"] and not d["being_used"]]
            dev_list = ", ".join(wasting_devs) if wasting_devs else "DEVICE"
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

        # ── live session waste timers ──────────────────────────────────────
        if total_waste_now:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section_title">⏱ This Session</div>', unsafe_allow_html=True)
            for key, secs in sorted(total_waste_now.items(), key=lambda x: -x[1]):
                name = device_lookup(key)
                st.markdown(
                    f'<div style="font-size:12px;color:{NAVY};padding:4px 0;">'
                    f'<b>{name}</b> — wasted <span style="color:{RED};font-weight:600">'
                    f'{secs:.1f}s</span></div>',
                    unsafe_allow_html=True
                )


# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — INCIDENT LOG (sortable)
# ───────────────────────────────────────────────────────────────────────────
with tab_history:
    st.markdown('<div class="section_title">📋 Incident Log</div>', unsafe_allow_html=True)

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
        display = log_df.copy()
        col     = st.session_state.sort_col
        sort_key = "_dt" if col in ("Date", "Time") else col
        display  = display.sort_values(sort_key, ascending=st.session_state.sort_asc)

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
          <thead><tr>{''.join(f'<th>{c}</th>' for c in COLUMNS)}</tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="empty_state">👀 No incidents yet — waiting for first session to complete...</div>',
            unsafe_allow_html=True
        )


# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — ANALYTICS
# ───────────────────────────────────────────────────────────────────────────
with tab_graphs:
    st.markdown('<div class="section_title">📊 Analytics</div>', unsafe_allow_html=True)

    log_df = load_log()

    if log_df.empty:
        st.markdown(
            '<div class="empty_state">📈 No data yet — data appears after a session ends.</div>',
            unsafe_allow_html=True
        )
    else:
        chart_df = log_df.sort_values("_dt")

        # 1. Cumulative CO2
        st.markdown("##### Cumulative CO₂ Leaked Over Time")
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
            margin=dict(l=40, r=20, t=20, b=40), height=260,
            xaxis=dict(gridcolor=SKY, title="Time",             tickfont=dict(size=10, color=TEAL)),
            yaxis=dict(gridcolor=SKY, title="Cumulative g CO₂", tickfont=dict(size=10, color=TEAL)),
            showlegend=False,
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        g1, g2 = st.columns(2)

        # 2. Wasted time per device
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
                margin=dict(l=10, r=40, t=20, b=40), height=300,
                xaxis=dict(gridcolor=SKY, title="Seconds", tickfont=dict(size=10, color=TEAL)),
                yaxis=dict(gridcolor=SKY, tickfont=dict(size=10, color=TEAL)),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # 3. CO2 share pie
        with g2:
            st.markdown("##### CO₂ Share by Device")
            by_co2 = log_df.groupby("Device")["Carbon (g)"].sum().reset_index()
            fig3 = go.Figure(go.Pie(
                labels=by_co2["Device"], values=by_co2["Carbon (g)"],
                hole=0.42,
                marker=dict(colors=px.colors.sequential.Teal),
                textinfo="label+percent", textfont_size=11,
            ))
            fig3.update_layout(
                paper_bgcolor=BEIGE, font=dict(color=NAVY, family="DM Sans"),
                margin=dict(l=20, r=20, t=20, b=20), height=300, showlegend=False,
            )
            st.plotly_chart(fig3, use_container_width=True)

        # 4. Daily energy wasted
        st.markdown("##### Daily Energy Wasted (Wh)")
        daily = log_df.groupby("Date")["Energy (Wh)"].sum().reset_index()
        fig4 = go.Figure(go.Bar(
            x=daily["Date"], y=daily["Energy (Wh)"], marker_color=NAVY,
            text=daily["Energy (Wh)"].apply(lambda v: f"{v:.3f} Wh"),
            textposition="outside",
        ))
        fig4.update_layout(
            paper_bgcolor=BEIGE, plot_bgcolor=WHITE,
            font=dict(color=NAVY, family="DM Sans"),
            margin=dict(l=40, r=20, t=20, b=60), height=280,
            xaxis=dict(gridcolor=SKY, title="Date",        tickfont=dict(size=10, color=TEAL)),
            yaxis=dict(gridcolor=SKY, title="Energy (Wh)", tickfont=dict(size=10, color=TEAL)),
            showlegend=False,
        )
        st.plotly_chart(fig4, use_container_width=True)

        # 5. Cost per device
        st.markdown("##### Cumulative Cost per Device (Rs)")
        by_cost = (log_df.groupby("Device")["Cost (Rs)"].sum()
                   .reset_index().sort_values("Cost (Rs)", ascending=False))
        fig5 = go.Figure(go.Bar(
            x=by_cost["Device"], y=by_cost["Cost (Rs)"],
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

# auto-refresh
time.sleep(REFRESH_SECS)
st.rerun()