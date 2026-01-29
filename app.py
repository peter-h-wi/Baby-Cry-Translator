import streamlit as st
import time
import os
import json
import numpy as np
from datetime import datetime
from src.inference import CryPredictor

# --- Config & Setup ---
EVENTS_FILE = "data/events.json"
st.set_page_config(page_title="Baby Interpreter Network", page_icon="📡", layout="wide")

if "predictor" not in st.session_state:
    st.session_state.predictor = CryPredictor(model_type='rf')

# --- Helper Functions ---
def load_events():
    if not os.path.exists(EVENTS_FILE):
        return []
    try:
        with open(EVENTS_FILE, "r") as f:
            return json.load(f)
    except:
        return []

def save_events(events):
    with open(EVENTS_FILE, "w") as f:
        json.dump(events, f, indent=4)

def add_event(prediction, confidence):
    events = load_events()
    new_event = {
        "id": str(int(time.time())),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": prediction,
        "confidence": confidence,
        "confirmed": None # None = Pending, True = Correct, False = Wrong
    }
    # Prepend to list (newest first)
    events.insert(0, new_event)
    save_events(events)
    return new_event

def update_event_feedback(event_id, correct, actual_label=None):
    events = load_events()
    for event in events:
        if event["id"] == event_id:
            event["confirmed"] = correct
            if not correct and actual_label:
                event["correction"] = actual_label
            break
    save_events(events)

# --- UI Modes ---

def render_baby_station():
    st.header("👶 Baby Station (Monitoring)")
    st.markdown("Place this device near the crib.")
    
    # 1. Manual Trigger (for Testing)
    st.subheader("🛠 Manual Trigger (Test)")
    if st.button("🚨 Simulate Cry Event", type="primary"):
        # Fake a prediction
        fake_labels = ['hungry', 'tired', 'belly_pain', 'discomfort']
        pred = np.random.choice(fake_labels)
        conf = np.random.uniform(0.7, 0.99)
        new_event = add_event(pred, conf)
        st.success(f"Event Generated: {pred} ({conf*100:.1f}%)")
        
    # 2. Continuous Loop Simulation
    st.divider()
    st.subheader("👂 Continuous Monitoring")
    is_active = st.toggle("Start Listening", value=False)
    
    status_placeholder = st.empty()
    log_placeholder = st.empty()
    
    if is_active:
        while True:
            # Random noise level
            db = np.random.uniform(30, 60)
            
            # 10% chance of spike
            if np.random.random() < 0.1:
                db = np.random.uniform(70, 90)
            
            status_placeholder.metric("Noise Level", f"{db:.1f} dB")
            
            if db > 80:
                # Trigger Event
                real_pred_label = "hungry" # In sim we just pick one, or use model if we had audio
                # Let's verify with model if we have a file, but here we sim.
                # Let's randomize for demo variety
                real_pred_label = np.random.choice(['hungry', 'tired', 'belly_pain'])
                conf = np.random.uniform(0.6, 0.95)
                
                add_event(real_pred_label, conf)
                log_placeholder.error(f"🚨 Cry Detected! Sent alert: {real_pred_label}")
                time.sleep(2) # Wait a bit so we don't spam
            
            time.sleep(1)
            # Rerun logic needed? Streamlit loops block interaction. 
            # Ideally use st_autorefresh or just rely on manual trigger for robust demo.

@st.fragment(run_every=2)
def render_feed():
    events = load_events()
    
    if not events:
        st.info("No events yet. Start the Baby Station to generate events.")
        return

    # Check for new high-confidence alerts to show a toast
    if events:
        latest = events[0]
        # logic to toast only once could be added here using session state tracking of last_seen_id
        if "last_seen_id" not in st.session_state or st.session_state.last_seen_id != latest["id"]:
             st.toast(f"🚨 New Alert: {latest['prediction'].upper()}")
             st.session_state.last_seen_id = latest["id"]

    for event in events:
        # Card Style
        with st.container(border=True):
            cols = st.columns([1, 3, 2])
            
            # Column 1: Icon & Prediction
            with cols[0]:
                st.write(f"### {event['timestamp'].split(' ')[1]}") # Time only
            
            # Column 2: Details
            with cols[1]:
                st.markdown(f"**Prediction:** {event['prediction'].upper()}")
                st.caption(f"Confidence: {event['confidence']*100:.1f}%")
                if event.get("confirmed") is True:
                     st.success("✅ Confirmed Correct")
                elif event.get("confirmed") is False:
                     st.warning(f"❌ Corrected to: {event.get('correction', 'Unknown')}")
            
            # Column 3: Actions
            with cols[2]:
                if event["confirmed"] is None:
                    # Check if we are in "Correction Mode" for this event
                    edit_key = f"edit_mode_{event['id']}"
                    
                    if st.session_state.get(edit_key):
                        # Correction UI
                        labels = list(id for id in ["hungry", "tired", "belly_pain", "discomfort", "burping"] if id != event['prediction'])
                        actual_label = st.selectbox("Actual reason:", labels, key=f"sel_{event['id']}")
                        
                        if st.button("Save Correction", key=f"save_{event['id']}"):
                            update_event_feedback(event['id'], False, actual_label)
                            del st.session_state[edit_key]
                            st.rerun()
                    else:
                        # Standard UI
                        st.write("Is this correct?")
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("✅", key=f"yes_{event['id']}"):
                                update_event_feedback(event['id'], True)
                                st.rerun()
                        with c2:
                            if st.button("❌", key=f"no_{event['id']}"):
                                st.session_state[edit_key] = True
                                st.rerun()

def render_parent_station():
    st.header("👨‍👩‍👧 Parent Station (Receiver)")
    st.markdown("Notifications will appear here automatically.")
    
    # Render the auto-refreshing feed
    render_feed()

# --- Main App Logic ---
mode = st.sidebar.radio("Select Device Mode", ["Baby Station", "Parent Station"])

if mode == "Baby Station":
    render_baby_station()
else:
    render_parent_station()
