import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, welch

# ----------------- STREAMLIT CONFIG -----------------
st.set_page_config(page_title="NeuroEMG-Insight", layout="wide")

st.title("NeuroEMG-Insight")
st.subheader("Automated EMG Feature Extraction & Interpretation Dashboard")

st.write(
    "Provide raw EMG signal either by uploading a file or pasting the raw values. "
    "The app will filter the signal (if long enough), compute features (RMS, MAV, "
    "MNF, MDF, ZC, WAMP) and generate an interpretation for the chosen muscle "
    "and activity type."
)

# ----------------- SIGNAL PROCESSING HELPERS -----------------
def bandpass_filter(signal, low=20, high=450, fs=1000, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, signal)

def rms(signal):
    return np.sqrt(np.mean(signal ** 2))

def mav(signal):
    return np.mean(np.abs(signal))

def zero_crossing(signal, threshold=0.01):
    return np.sum(((signal[:-1] * signal[1:]) < 0) &
                  (np.abs(signal[:-1] - signal[1:]) > threshold))

def wamp(signal, threshold=0.0005):
    return np.sum(np.abs(np.diff(signal)) >= threshold)

def mnf_mdf(signal, fs=1000):
    freqs, psd = welch(signal, fs)
    mnf = np.sum(freqs * psd) / np.sum(psd)
    cumulative = np.cumsum(psd)
    mdf = freqs[np.where(cumulative >= cumulative[-1] / 2)[0][0]]
    return mnf, mdf

# ----------------- UI: CONTEXT INPUT (MUSCLE + ACTIVITY) -----------------
st.markdown("### Clinical Context")

col1, col2 = st.columns(2)

with col1:
    muscle_name = st.text_input(
        "Muscle name (e.g., Tibialis Anterior, Vastus Lateralis, Biceps Brachii)",
        value="Unknown muscle"
    )

with col2:
    activity_type = st.selectbox(
        "Activity type",
        [
            "Unknown / Not specified",
            "Rest",
            "Isometric contraction",
            "Dynamic movement / exercise",
            "Gait / Walking",
            "Fatigue protocol (repeated contractions)",
        ],
    )

st.markdown("---")

# ----------------- UI: INPUT METHOD -----------------
input_method = st.radio(
    "Choose how to provide EMG signal:",
    ["Upload file (CSV / TXT)", "Paste raw EMG values"],
    horizontal=True,
)

fs = st.number_input("Sampling rate (Hz)", min_value=100, max_value=5000, value=1000)

signal = None

if input_method == "Upload file (CSV / TXT)":
    uploaded_file = st.file_uploader("Upload EMG file", type=["csv", "txt"])
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            col = df.columns[0]
            signal = df[col].values.astype(float)
            st.write("Detected CSV column as EMG signal:", col)
        else:  # txt
            signal = np.loadtxt(uploaded_file).astype(float).flatten()
        st.success(f"Loaded EMG signal with {len(signal)} samples.")

else:  # Paste raw values
    text_data = st.text_area(
        "Paste raw EMG samples (one value per line, or separated by space/comma):",
        height=200,
    )
    if text_data.strip():
        tokens = text_data.replace(",", " ").split()
        try:
            vals = [float(x) for x in tokens]
            signal = np.array(vals, dtype=float)
            st.success(f"Parsed {len(signal)} EMG samples from pasted text.")
        except ValueError:
            st.error("Could not parse numbers. Check that all entries are numeric.")

# ----------------- PROCESSING & INTERPRETATION -----------------
MIN_FREQ_LEN = 50   # minimum samples for filtering + frequency features

if signal is not None:
    n = len(signal)
    st.write(f"### Raw EMG Signal  —  {n} samples")
    st.line_chart(signal)

    # ---- Always compute simple time-domain features ----
    time_features = {
        "RMS": rms(signal),
        "MAV": mav(signal),
    }

    can_do_freq = n >= MIN_FREQ_LEN

    if not can_do_freq:
        st.warning(
            f"Signal length ({n} samples) is too short for safe filtering and "
            f"frequency-domain analysis (need at least {MIN_FREQ_LEN}). "
            "Showing only time-domain features."
        )
        filtered = None
    else:
        filtered = bandpass_filter(signal, fs=fs)
        st.write("### Filtered Signal (20–450 Hz Bandpass)")
        st.line_chart(filtered)

    # ---- Feature table ----
    feature_rows = []

    # Time-domain features (always)
    feature_rows.append(("RMS", time_features["RMS"]))
    feature_rows.append(("MAV", time_features["MAV"]))

    freq_features = {}
    if can_do_freq and filtered is not None:
        mnf_val, mdf_val = mnf_mdf(filtered, fs=fs)
        freq_features["MNF (Hz)"] = mnf_val
        freq_features["MDF (Hz)"] = mdf_val
        freq_features["ZC"] = zero_crossing(filtered)
        freq_features["WAMP"] = wamp(filtered)

        for k, v in freq_features.items():
            feature_rows.append((k, v))

    st.write("### Extracted Features")
    st.table(pd.DataFrame(feature_rows, columns=["Feature", "Value"]))

    # ---- Interpretation Engine ----
    st.write("### Interpretation (Prototype)")

    interpretation = []

    # Context info
    interpretation.append(
        f"Muscle: **{muscle_name}**, Activity: **{activity_type}**."
    )

    # Time-domain logic
    if time_features["RMS"] > 0.9:
        interpretation.append("High RMS → strong muscle contraction intensity.")
    elif time_features["RMS"] < 0.3:
        interpretation.append("Low RMS → weak muscle activation / poor recruitment.")

    if time_features["MAV"] < 0.1:
        interpretation.append("Very low MAV → overall low EMG amplitude.")

    # Frequency-domain logic (only if we have it)
    if can_do_freq and filtered is not None:
        mnf_val = freq_features["MNF (Hz)"]
        mdf_val = freq_features["MDF (Hz)"]

        if mnf_val < 70 and mdf_val < 75:
            interpretation.append(
                "Frequency content shifted to lower band → this pattern is consistent "
                "with developing muscle fatigue or reduced motor unit firing rate."
            )
        elif mnf_val > 100:
            interpretation.append(
                "High mean frequency → strong, fast motor unit firing, typical of "
                "high-intensity or explosive contractions."
            )
    else:
        interpretation.append(
            "Frequency-based features (MNF, MDF, ZC, WAMP) not computed because the "
            f"signal was too short (< {MIN_FREQ_LEN} samples). For full analysis, "
            "record a longer EMG segment."
        )

    # Activity-specific note (simple version)
    if "Fatigue" in activity_type and can_do_freq and filtered is not None:
        interpretation.append(
            "Since this is a fatigue protocol, monitor MNF/MDF decrease over time to "
            "quantify fatigue progression in this muscle."
        )
    elif "Rest" in activity_type:
        interpretation.append(
            "During rest, EMG activity should be minimal. Elevated RMS/MAV at rest "
            "may indicate noise, poor electrode placement, or involuntary activation."
        )

    for line in interpretation:
        st.markdown(f"- {line}")
