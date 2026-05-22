"""
app.py — Streamlit GUI for the audiobook censoring tool.

Run with:   streamlit run app.py
"""

import queue
import tempfile
import threading
from pathlib import Path

import streamlit as st

from censor_core import censor_audiobook, CensorResult

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Audiobook Censor",
    page_icon="🔇",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Styling — dark industrial/utilitarian theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  :root {
    --bg:       #0e0e0e;
    --surface:  #1a1a1a;
    --border:   #2e2e2e;
    --accent:   #e8ff3c;
    --danger:   #ff3c3c;
    --text:     #e0e0e0;
    --muted:    #666;
    --mono:     'IBM Plex Mono', monospace;
    --sans:     'IBM Plex Sans', sans-serif;
  }

  html, body, [class*="css"] {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
  }

  /* Main container */
  .main .block-container {
    max-width: 780px;
    padding-top: 2.5rem;
  }

  /* Header */
  .app-header {
    border-left: 4px solid var(--accent);
    padding: 0.5rem 0 0.5rem 1.2rem;
    margin-bottom: 2.5rem;
  }
  .app-header h1 {
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--accent);
    margin: 0;
  }
  .app-header p {
    font-size: 0.85rem;
    color: var(--muted);
    margin: 0.3rem 0 0;
    font-family: var(--mono);
  }

  /* Section labels */
  .section-label {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
  }

  /* Stats panel */
  .stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    margin: 1.5rem 0;
  }
  .stat-box {
    background: var(--surface);
    padding: 1.2rem 1rem;
    text-align: center;
  }
  .stat-value {
    font-family: var(--mono);
    font-size: 2rem;
    font-weight: 600;
    color: var(--accent);
    line-height: 1;
  }
  .stat-label {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.4rem;
  }

  /* Log box */
  .log-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 1rem;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: #aaa;
    max-height: 300px;
    overflow-y: auto;
    white-space: pre-wrap;
  }

  /* Success banner */
  .success-banner {
    background: #1a2200;
    border: 1px solid var(--accent);
    border-left: 4px solid var(--accent);
    padding: 1rem 1.2rem;
    font-family: var(--mono);
    font-size: 0.85rem;
    color: var(--accent);
    margin: 1rem 0;
  }

  /* Error banner */
  .error-banner {
    background: #220000;
    border: 1px solid var(--danger);
    border-left: 4px solid var(--danger);
    padding: 1rem 1.2rem;
    font-family: var(--mono);
    font-size: 0.85rem;
    color: var(--danger);
    margin: 1rem 0;
  }

  /* Override Streamlit widget labels */
  .stSelectbox label, .stFileUploader label, .stSlider label {
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted) !important;
  }

  /* Streamlit button override */
  .stButton > button {
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
    font-weight: 600;
    letter-spacing: 0.05em;
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 1px !important;
    padding: 0.65rem 2rem !important;
    transition: opacity 0.15s;
  }
  .stButton > button:hover { opacity: 0.85; }
  .stButton > button:disabled {
    background: var(--border) !important;
    color: var(--muted) !important;
  }

  /* Download button */
  .stDownloadButton > button {
    font-family: var(--mono) !important;
    font-size: 0.85rem !important;
    background: #1a2200 !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 1px !important;
  }

  /* Progress bar */
  .stProgress > div > div > div > div {
    background: var(--accent) !important;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div class="app-header">
  <h1>🔇 AUDIOBOOK CENSOR</h1>
  <p>Transcribe → detect → silence. One ffmpeg pass, zero re-encodes.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "running" not in st.session_state:
    st.session_state.running = False
if "result" not in st.session_state:
    st.session_state.result = None
if "log_lines" not in st.session_state:
    st.session_state.log_lines = []
if "output_bytes" not in st.session_state:
    st.session_state.output_bytes = None
if "error" not in st.session_state:
    st.session_state.error = None

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------

st.markdown('<div class="section-label">01 — Upload audiobook</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drag & drop your audiobook",
    type=["m4b", "mp3", "mp4", "aac", "ogg", "flac", "m4a"],
    label_visibility="collapsed",
)

st.markdown('<div class="section-label" style="margin-top:1.5rem">02 — Model settings</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    model_size = st.selectbox(
        "Whisper model",
        options=["tiny", "small", "medium", "large"],
        index=1,
        help="Bigger = more accurate, slower. 'small' is recommended.",
    )
with col2:
    beam_size = st.selectbox(
        "Beam size",
        options=[1, 3, 5, 8],
        index=2,
        help="Higher = more accurate, slower. 5 is a good balance.",
    )

bitrate = st.select_slider(
    "Output audio bitrate",
    options=["96k", "128k", "192k", "256k", "320k"],
    value="192k",
)

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

can_run = uploaded is not None and not st.session_state.running
run_clicked = st.button(
    "▶  RUN CENSORING PIPELINE",
    disabled=not can_run,
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

if run_clicked and uploaded:
    st.session_state.running = True
    st.session_state.result = None
    st.session_state.log_lines = []
    st.session_state.output_bytes = None
    st.session_state.error = None

    log_queue: queue.Queue[str] = queue.Queue()

    def progress_cb(msg: str) -> None:
        log_queue.put(msg)

    # Save upload to temp dir
    tmp_dir = tempfile.mkdtemp()
    input_path = Path(tmp_dir) / uploaded.name
    output_path = Path(tmp_dir) / (input_path.stem + "_censored" + input_path.suffix)

    with open(input_path, "wb") as f:
        f.write(uploaded.getbuffer())

    result_holder: list[CensorResult | Exception] = []

    def run_pipeline() -> None:
        try:
            r = censor_audiobook(
                input_path=input_path,
                output_path=output_path,
                model_size=model_size,
                beam_size=int(beam_size),
                audio_bitrate=bitrate,
                progress_cb=progress_cb,
            )
            result_holder.append(r)
            if output_path.exists():
                with open(output_path, "rb") as f:
                    log_queue.put("__OUTPUT_READY__:" + str(output_path))
        except Exception as e:
            result_holder.append(e)
        finally:
            log_queue.put("__DONE__")

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    # Live log display while thread runs
    log_placeholder = st.empty()
    progress_placeholder = st.empty()
    log_lines: list[str] = []

    while True:
        try:
            msg = log_queue.get(timeout=0.3)
        except queue.Empty:
            # Update display even if no new messages
            log_placeholder.markdown(
                f'<div class="log-box">{"<br>".join(log_lines[-40:])}</div>',
                unsafe_allow_html=True,
            )
            continue

        if msg == "__DONE__":
            break
        if msg.startswith("__OUTPUT_READY__:"):
            fpath = Path(msg.split(":", 1)[1])
            if fpath.exists():
                with open(fpath, "rb") as f:
                    st.session_state.output_bytes = f.read()
            continue

        log_lines.append(msg)
        log_placeholder.markdown(
            f'<div class="log-box">{"<br>".join(log_lines[-40:])}</div>',
            unsafe_allow_html=True,
        )

    thread.join()

    # Final log render
    log_placeholder.markdown(
        f'<div class="log-box">{"<br>".join(log_lines)}</div>',
        unsafe_allow_html=True,
    )

    if result_holder:
        r = result_holder[0]
        if isinstance(r, Exception):
            st.session_state.error = str(r)
        else:
            st.session_state.result = r

    st.session_state.running = False
    st.session_state.log_lines = log_lines
    st.rerun()

# ---------------------------------------------------------------------------
# Show persisted log after rerun
# ---------------------------------------------------------------------------

if st.session_state.log_lines and not st.session_state.running:
    st.markdown('<div class="section-label" style="margin-top:1.5rem">Processing log</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="log-box">{"<br>".join(st.session_state.log_lines)}</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if st.session_state.error:
    st.markdown(
        f'<div class="error-banner">❌ Error: {st.session_state.error}</div>',
        unsafe_allow_html=True,
    )

if st.session_state.result:
    result: CensorResult = st.session_state.result
    hours = int(result.duration // 3600)
    minutes = int((result.duration % 3600) // 60)

    st.markdown(f"""
    <div class="stats-grid">
      <div class="stat-box">
        <div class="stat-value">{result.total_censored}</div>
        <div class="stat-label">Words Censored</div>
      </div>
      <div class="stat-box">
        <div class="stat-value">{hours}h {minutes}m</div>
        <div class="stat-label">Audio Duration</div>
      </div>
      <div class="stat-box">
        <div class="stat-value">1x</div>
        <div class="stat-label">ffmpeg Pass</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="success-banner">✅ Censoring complete. Download your file below.</div>', unsafe_allow_html=True)

    if st.session_state.output_bytes and uploaded:
        st.download_button(
            label="📥  DOWNLOAD CENSORED AUDIOBOOK",
            data=st.session_state.output_bytes,
            file_name=f"censored_{uploaded.name}",
            mime="audio/x-m4b" if uploaded.name.endswith(".m4b") else "audio/mpeg",
            use_container_width=True,
        )

    # Detailed mute list (expandable)
    if result.mutes:
        with st.expander(f"View all {len(result.mutes)} censored timestamps"):
            rows = "\n".join(
                f"  {i+1:3d}.  {m.start:7.2f}s → {m.end:7.2f}s   '{m.word}'"
                for i, m in enumerate(result.mutes)
            )
            st.code(rows, language=None)
