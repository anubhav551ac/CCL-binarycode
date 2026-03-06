# ghostgrid demo 

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import random
from datetime import datetime

# page config
st.set_page_config(
    page_title="GhostGrid Demo",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# colors theme
NAVY     = "#2F4156"
TEAL     = "#567C8D"
SKY      = "#C8D9E6"
BEIGE    = "#F5EFEB"
WHITE    = "#FFFFFF"
RED      = "#C0392B"
RED_SOFT = "#FADBD8"

# the css for custom theme
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

  .stApp {{ background-color: {BEIGE}; font-family: 'DM Sans', sans-serif; }}
  .block-container {{ padding: 0  2rem 0 2rem !important; max-width: 100% !important; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  section[data-testid="stSidebar"] {{ display: none; }}

  .top_banner {{
    background: {NAVY};
    padding: 14px 36px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .brand_logo {{ font-family: 'DM Serif Display', serif; font-size: 42px; color: {WHITE}; letter-spacing: 1px; text-align: center; flex: 1; }}
  .brand_logo span {{ color: {SKY}; }}
  .banner_right {{ display: flex; align-items: center; gap: 20px; }}
  .live_time {{ font-size: 13px; color: {SKY}; letter-spacing: 1px; }}
  .status_pill {{ padding: 5px 16px; border-radius: 20px; font-size: 12px; font-weight: 600; letter-spacing: 0.5px; }}
  .pill_occupied {{ background: rgba(86,124,141,0.3); color: {SKY}; border: 1px solid {TEAL}; }}
  .pill_empty    {{ background: rgba(192,57,43,0.2); color: #F1948A; border: 1px solid {RED}; }}

  .page_body {{ padding: 28px 36px; }}

  .metric_card {{
    background: {WHITE}; border-radius: 12px; padding: 20px 24px;
    border: 1px solid {SKY}; box-shadow: 0 2px 12px rgba(47,65,86,0.07);
  }}
  .metric_value {{ font-family: 'DM Serif Display', serif; font-size: 36px; color: {NAVY}; line-height: 1; margin-bottom: 4px; }}
  .metric_label {{ font-size: 10px; color: {TEAL}; letter-spacing: 1.5px; text-transform: uppercase; font-weight: 500; margin-top: 6px; }}
  .metric_icon {{ font-size: 18px; margin-bottom: 6px; }}

  .section_title {{
    font-size: 18px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; padding 10px 10px;
    color: {TEAL}; margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px solid {SKY};
  }}

  .alert_box_safe  {{ background: {WHITE}; border: 1.5px solid {TEAL}; border-radius: 12px; padding: 22px; }}
  .alert_box_waste {{ background: {RED_SOFT}; border: 2px solid {RED}; border-radius: 12px; padding: 22px; animation: pulsered 1.2s infinite; }}
  @keyframes pulsered {{
    0%,100% {{ box-shadow: 0 0 0 0 rgba(192,57,43,0.3); }}
    50%      {{ box-shadow: 0 0 0 8px rgba(192,57,43,0); }}
  }}
  .alert_title_safe  {{ font-family:'DM Serif Display',serif; font-size:22px; color:{TEAL}; margin-bottom:6px; }}
  .alert_title_waste {{ font-family:'DM Serif Display',serif; font-size:22px; color:{RED}; margin-bottom:6px; }}
  .alert_detail {{ font-size:12px; color:#555; line-height:2; margin-top:8px; }}

  .stButton > button {{
    background: {NAVY} !important; color: {WHITE} !important; border: none !important;
    border-radius: 8px !important; padding: 12px 0 !important; font-size: 14px !important;
    font-weight: 600 !important; width: 100% !important;
    font-family: 'DM Sans', sans-serif !important; transition: background 0.2s !important;
  }}
  .stButton > button:hover {{ background: {TEAL} !important; }}

  .log_table {{ width:100%; border-collapse:collapse; font-size:12px; background:{WHITE}; border-radius:12px; overflow:hidden; box-shadow:0 2px 12px rgba(47,65,86,0.07); }}
  .log_table th {{ background:{NAVY}; color:{WHITE}; padding:11px 16px; text-align:left; font-size:10px; letter-spacing:1.5px; text-transform:uppercase; }}
  .log_table td {{ padding:10px 16px; border-bottom:1px solid {SKY}; color:{NAVY}; }}
  .log_table tr:last-child td {{ border-bottom:none; }}
  .log_table tr:hover td {{ background:{BEIGE}; }}
  .td-device {{ background:{SKY}; color:{NAVY}; padding:2px 8px; border-radius:4px; font-size:10px; font-weight:600; letter-spacing:0.5px; }}
  .td-red {{ color:{RED}; font-weight:600; }}

  .empty_state {{ text-align:center; padding:32px; color:{TEAL}; font-size:13px; background:{WHITE}; border-radius:12px; border:1px dashed {SKY}; }}
  .row_divider {{ height:1px; background:{SKY}; margin:28px 0; }}
  .toast_msg {{ background:{TEAL}; color:{WHITE}; padding:12px 20px; border-radius:8px; font-size:13px; font-weight:500; text-align:center; margin-top:8px; }}
  [data-testid="column"] {{ padding: 0 8px !important; }}

  .camera_box {{
    background: {SKY};
    border-radius: 12px;
    border: 2px solid {SKY};
    aspect-ratio: 4/3;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    position: relative;
    overflow: hidden;
  }}
  .camera_box_waste {{
    background: #f0dada;
    border-color: {RED};
  }}
  .camera_label {{
    font-size: 13px;
    font-weight: 600;
    color: {NAVY};
    letter-spacing: 0.5px;
  }}
  .camera_sub {{
    font-size: 11px;
    color: {TEAL};
  }}
  .camera_badge {{
    position: absolute;
    top: 12px; left: 12px;
    background: rgba(47,65,86,0.8);
    color: {SKY};
    font-size: 10px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    letter-spacing: 1px;
  }}
  .detect_box_green {{
    border: 2px solid #27AE60;
    background: rgba(39,174,96,0.08);
    padding: 8px 20px;
    border-radius: 6px;
    font-size: 11px;
    color: #27AE60;
    font-weight: 600;
  }}
  .detect_box_red {{
    border: 2px solid {RED};
    background: rgba(192,57,43,0.08);
    padding: 8px 20px;
    border-radius: 6px;
    font-size: 11px;
    color: {RED};
    font-weight: 600;
    animation: pulsered 1.2s infinite;
  }}
</style>
""", unsafe_allow_html=True)


# session state: stuff that needs to persist across reruns
if "waste_log"      not in st.session_state: st.session_state.waste_log      = []
if "total_carbon"   not in st.session_state: st.session_state.total_carbon   = 0.0
if "total_cost"     not in st.session_state: st.session_state.total_cost     = 0.0
if "turned_off"     not in st.session_state: st.session_state.turned_off     = False
if "turn_off_msg"   not in st.session_state: st.session_state.turn_off_msg   = False
if "tick"           not in st.session_state: st.session_state.tick           = 0
if "room_state"     not in st.session_state: st.session_state.room_state     = "occupied"
if "state_counter"  not in st.session_state: st.session_state.state_counter  = 0
if "minutes_waste"  not in st.session_state: st.session_state.minutes_waste  = 0.0

# this is used to randomly stimulate a device being left unused
DEVICES = ["laptop", "laptop", "cell phone", "monitor", "laptop"]

# advance the tick each rerun
st.session_state.tick          += 1
st.session_state.state_counter += 1

# simple state machine to simulate a realistic room over time:
#   if the room occupied for a while or device left on in empty room it  clears up and repeat
if st.session_state.room_state == "occupied":
    if st.session_state.state_counter >= random.randint(8, 14):
        st.session_state.room_state    = "empty_waste"
        st.session_state.state_counter = 0
        st.session_state.minutes_waste = 0.0
        st.session_state.turned_off    = False
        st.session_state.turn_off_msg  = False

elif st.session_state.room_state == "empty_waste":
    st.session_state.minutes_waste += 0.5   # each tick is roughly half a minute of waste
    if st.session_state.state_counter >= random.randint(6, 12):
        st.session_state.room_state    = "empty_clean"
        st.session_state.state_counter = 0

elif st.session_state.room_state == "empty_clean":
    if st.session_state.state_counter >= random.randint(2, 5):
        st.session_state.room_state    = "occupied"
        st.session_state.state_counter = 0

# figure out what's happening right now based on state
room_state     = st.session_state.room_state
person_present = (room_state == "occupied")
device         = random.choice(DEVICES) if room_state == "empty_waste" else None
waste_happening = (
    room_state == "empty_waste" and
    not st.session_state.turned_off
)

# nepal ma its about 490 gCO2 per kWh
carbon_intensity = 490
minutes_wasting  = round(st.session_state.minutes_waste, 1)
carbon_now       = round(minutes_wasting * 0.004 * carbon_intensity, 2) if waste_happening else 0
cost_now         = round(minutes_wasting * 0.0002, 3) if waste_happening else 0

# keep running totals and log an entry every 2 ticks while waste is happening
if waste_happening:
    st.session_state.total_carbon = round(st.session_state.total_carbon + 0.08, 2)
    st.session_state.total_cost   = round(st.session_state.total_cost   + 0.001, 3)
    if st.session_state.tick % 2 == 0 and len(st.session_state.waste_log) < 100:
        st.session_state.waste_log.append({
            "time":   datetime.now().strftime("%H:%M:%S"),
            "device": device or "laptop",
            "co2_g":  carbon_now,
            "rs":     cost_now
        })


# header
room_label = "🟢 OCCUPIED" if person_present else "🔴 EMPTY"
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


# three metric cards across the top
st.markdown('<div class="section_title">    Live Metrics</div>', unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(f"""<div class="metric_card">
        <div class="metric_icon">⚡</div>
        <div class="metric_value">{carbon_intensity}</div>
        <div class="metric_label">gCO₂ / kWh — Grid Now</div>
    </div>""", unsafe_allow_html=True)

with m2:
    st.markdown(f"""<div class="metric_card">
        <div class="metric_icon">💨</div>
        <div class="metric_value">{st.session_state.total_carbon:.1f}g</div>
        <div class="metric_label">Total CO₂ Leaked</div>
    </div>""", unsafe_allow_html=True)

with m3:
    st.markdown(f"""<div class="metric_card">
        <div class="metric_icon">💸</div>
        <div class="metric_value">Rs {st.session_state.total_cost:.2f}</div>
        <div class="metric_label">Money Wasted Today</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# main content : fake camera feed on the left, status + controls on the right
col_feed, col_right = st.columns([6, 4], gap="large")

with col_feed:
    st.markdown('<div class="section_title">    📷 Live Sentinel Feed</div>', unsafe_allow_html=True)

    # show different states in the mock feed depending on whats happening
    if person_present:
        st.markdown(f"""
        <div class="camera_box">
          <div class="camera_badge">AI DETECTION ACTIVE</div>
          <div style="font-size:52px">🧑‍💻</div>
          <div class="detect_box_green">PERSON  ✓  DETECTED</div>
          <div class="camera_sub">Room occupied — monitoring</div>
        </div>""", unsafe_allow_html=True)

    elif waste_happening:
        dev_icon = "💻" if device == "laptop" else "📱" if device == "cell phone" else "🖥️"
        st.markdown(f"""
        <div class="camera_box camera_box_waste">
          <div class="camera_badge">AI DETECTION ACTIVE</div>
          <div style="font-size:52px">{dev_icon}</div>
          <div class="detect_box_red">⚠ {(device or "DEVICE").upper()}  —  ROOM EMPTY</div>
          <div class="camera_sub" style="color:{RED}">Vampire power detected!</div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="camera_box">
          <div class="camera_badge">AI DETECTION ACTIVE</div>
          <div class="camera_label">Room Empty</div>
          <div class="camera_sub">No devices detected</div>
        </div>""", unsafe_allow_html=True)


with col_right:
    st.markdown('<div class="section_title">🚨 Status</div>', unsafe_allow_html=True)

    # alert box changes based on whats going on in the room
    if waste_happening:
        st.markdown(f"""
        <div class="alert_box_waste">
            <div class="alert_title_waste">🚨 Waste Detected!</div>
            <div class="alert_detail">
                Room is <b style="color:{RED}">EMPTY</b> but <b>{(device or 'DEVICE').upper()}</b> is ON<br>
                ⏱ Wasting for <b>{minutes_wasting} mins</b><br>
                💨 Carbon: <b>{carbon_now}g CO₂</b><br>
                💸 Cost: <b>Rs {cost_now}</b><br>
                ⚡ Grid: <b>{carbon_intensity} gCO₂/kWh</b>
            </div>
        </div>""", unsafe_allow_html=True)

    elif st.session_state.turned_off:
        st.markdown(f"""
        <div class="alert_box_safe">
            <div class="alert_title_safe">✅ Devices Off</div>
            <div class="alert_detail" style="text-align:center;color:{TEAL}">
                All devices powered down.<br>No waste detected.
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

    # button to simulate turning everything off
    st.markdown('<div class="section_title">🔌 Control</div>', unsafe_allow_html=True)
    if st.button("⚡  Turn Off All Devices", key="turnoff"):
        st.session_state.turned_off   = True
        st.session_state.turn_off_msg = True

    if st.session_state.turn_off_msg:
        st.markdown(
            f'<div class="toast_msg">✅ Signal sent — all devices powered down!</div>',
            unsafe_allow_html=True
        )


# incident log that shows the last 10 waste events
st.markdown('<div class="row_divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section_title">📋 Incident Log</div>', unsafe_allow_html=True)

if st.session_state.waste_log:
    rows = ""
    for e in reversed(st.session_state.waste_log[-10:]):
        rows += f"""<tr>
            <td>{e['time']}</td>
            <td><span class="td-device">{e['device'].upper()}</span></td>
            <td class="td-red">{e['co2_g']}g</td>
            <td class="td-red">Rs {e['rs']}</td>
        </tr>"""
    st.markdown(f"""
    <table class="log_table">
      <thead><tr><th>Time</th><th>Device</th><th>CO₂ Leaked</th><th>Cost</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>""", unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="empty_state">👀 No incidents yet — waiting for first waste event...</div>',
        unsafe_allow_html=True
    )


#  carbon chart ( only shows up once we have enough data points)
st.markdown('<div class="row_divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section_title">📈 Carbon Leaked Over Time(gm)</div>', unsafe_allow_html=True)

if len(st.session_state.waste_log) >= 2:
    df = pd.DataFrame(st.session_state.waste_log)
    df["cumulative"] = df["co2_g"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["cumulative"],
        mode="lines+markers",
        line=dict(color=TEAL, width=2.5),
        marker=dict(color=NAVY, size=6, line=dict(color=WHITE, width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(86,124,141,0.1)",
    ))
    fig.update_layout(
        paper_bgcolor=BEIGE, plot_bgcolor=WHITE,
        font=dict(color=NAVY, family="DM Sans"),
        margin=dict(l=40, r=20, t=20, b=40), height=260,
        xaxis=dict(gridcolor=SKY, title="Time",               tickfont=dict(size=10, color=TEAL), linecolor=SKY),
        yaxis=dict(gridcolor=SKY, title="Cumulative CO₂ (g)", tickfont=dict(size=10, color=TEAL), linecolor=SKY),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.markdown(
        '<div class="empty_state">📈 Chart will appear once the first waste incident is logged...</div>',
        unsafe_allow_html=True
    )


# auto refresh every 3 seconds to keep the simulation ticking
st.markdown("</div>", unsafe_allow_html=True)
time.sleep(3)
st.rerun()
