import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, welch
import matplotlib.pyplot as plt
from io import StringIO

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

# ----------------- HELPERS -----------------
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
    return mnf, mdf, freqs, psd

def make_example_signal(fs=1000, duration_s=2.0):
    t = np.linspace(0, duration_s, int(fs * duration_s), endpoint=False)
    # simple EMG-like noisy burst
    sig = 0.3 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(len(t))
    return t, sig

# ----------------- SIDEBAR: SAMPLE DATA DOWNLOAD -----------------
with st.sidebar:
    st.header("Sample EMG data")
    t_ex, sig_ex = make_example_signal()
    df_ex = pd.DataFrame({"emg": sig_ex})
    csv_buf = StringIO()
    df_ex.to_csv(csv_buf, index=False)
    st.download_button(
        "Download example EMG CSV",
        data=csv_buf.getvalue(),
        file_name="example_emg.csv",
        mime="text/csv",
    )
    st.caption("Use this if you just want to try the tool quickly.")

# ----------------- CLINICAL CONTEXT -----------------
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

# ----------------- INPUT METHOD -----------------
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

# ----------------- MAIN ANALYSIS -----------------
MIN_FREQ_LEN = 50   # minimum samples for safe freq analysis

if signal is not None:
    n = len(signal)
    st.write(f"Loaded EMG signal with **{n} samples**.")

    # Precompute time-domain features on raw signal
    time_feats = {
        "RMS": rms(signal),
        "MAV": mav(signal),
    }

    can_do_freq = n >= MIN_FREQ_LEN
    filtered = None
    freq_feats = {}
    freqs = None
    psd = None

    if can_do_freq:
        filtered = bandpass_filter(signal, fs=fs)
        mnf_val, mdf_val, freqs, psd = mnf_mdf(filtered, fs=fs)
        freq_feats = {
            "MNF (Hz)": mnf_val,
            "MDF (Hz)": mdf_val,
            "ZC": zero_crossing(filtered),
            "WAMP": wamp(filtered),
        }

    # ----------------- TABS LAYOUT -----------------
    tab_time, tab_freq, tab_features = st.tabs(
        ["Time-domain signals", "Frequency analysis", "Features & interpretation"]
    )

    # === TIME TAB ===
    with tab_time:
        st.subheader("Raw EMG Signal")
        st.line_chart(signal)

        if filtered is not None:
            st.subheader("Filtered EMG (20–450 Hz bandpass)")
            st.line_chart(filtered)
        else:
            st.info(
                f"Signal too short for filtering and frequency analysis "
                f"(need at least {MIN_FREQ_LEN} samples)."
            )

    # === FREQUENCY TAB ===
    with tab_freq:
        if can_do_freq and filtered is not None and freqs is not None:
            st.subheader("Power Spectral Density (Welch)")
            fig_psd, ax_psd = plt.subplots()
            ax_psd.semilogy(freqs, psd)
            ax_psd.set_xlabel("Frequency (Hz)")
            ax_psd.set_ylabel("PSD (power / Hz)")
            ax_psd.set_title("EMG Power Spectrum")
            st.pyplot(fig_psd)

            # Simple FFT magnitude
            st.subheader("FFT Magnitude")
            freqs_fft = np.fft.rfftfreq(len(filtered), d=1.0/fs)
            fft_mag = np.abs(np.fft.rfft(filtered))
            fig_fft, ax_fft = plt.subplots()
            ax_fft.plot(freqs_fft, fft_mag)
            ax_fft.set_xlabel("Frequency (Hz)")
            ax_fft.set_ylabel("Magnitude")
            ax_fft.set_title("FFT of EMG signal")
            st.pyplot(fig_fft)
        else:
            st.warning(
                f"Frequency domain analysis not available because signal length "
                f"is < {MIN_FREQ_LEN} samples."
            )

    # === FEATURES & INTERPRETATION TAB ===
    with tab_features:
        st.subheader("Extracted Features")

        rows = [("RMS", time_feats["RMS"]), ("MAV", time_feats["MAV"])]
        for k, v in freq_feats.items():
            rows.append((k, v))

        st.table(pd.DataFrame(rows, columns=["Feature", "Value"]))

        st.subheader("Interpretation (Prototype)")
        interpretation = []

        # Context line
        interpretation.append(
            f"Muscle: **{muscle_name}**, Activity: **{activity_type}**."
        )

        # Time-domain logic
        if time_feats["RMS"] > 0.9:
            interpretation.append("High RMS → strong muscle contraction intensity.")
        elif time_feats["RMS"] < 0.3:
            interpretation.append("Low RMS → weak muscle activation / poor recruitment.")

        if time_feats["MAV"] < 0.1:
            interpretation.append("Very low MAV → overall low EMG amplitude.")

        # Frequency-domain logic
        if can_do_freq and filtered is not None:
            mnf_val = freq_feats["MNF (Hz)"]
            mdf_val = freq_feats["MDF (Hz)"]
            if mnf_val < 70 and mdf_val < 75:
                interpretation.append(
                    "Frequency content shifted to lower band → pattern consistent "
                    "with developing muscle fatigue or reduced motor unit firing rate."
                )
            elif mnf_val > 100:
                interpretation.append(
                    "High mean frequency → strong, fast motor unit firing, typical "
                    "of high-intensity or explosive contractions."
                )
        else:
            interpretation.append(
                f"Frequency-based features (MNF, MDF, ZC, WAMP) not computed because "
                f"the signal was too short (< {MIN_FREQ_LEN} samples)."
            )

        # Activity-specific hints
        if "Fatigue" in activity_type and can_do_freq and filtered is not None:
            interpretation.append(
                "For fatigue protocols, track MNF/MDF decrease across repetitions "
                "to quantify fatigue progression for this muscle."
            )
        elif "Rest" in activity_type:
            interpretation.append(
                "During rest, EMG should be near baseline. Elevated RMS/MAV at rest "
                "may indicate noise, poor electrode placement, or involuntary activity."
            )

        for line in interpretation:
            st.markdown(f"- {line}")
else:
    st.info("Upload or paste an EMG signal to begin analysis.")
