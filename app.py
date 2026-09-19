import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import av
import time
import os
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="SIGNIFY", page_icon="🤟", layout="wide")

STATIC_MODEL_PATH = "models/gesture_model.pkl"
DYNAMIC_MODEL_PATH = "models/dynamic_gesture_model.keras"
ENCODER_PATH = "models/dynamic_label_encoder.pkl"
CUSTOM_VOCABULARY_PATH = "Data/custom_vocabulary"
os.makedirs(CUSTOM_VOCABULARY_PATH, exist_ok=True)

st.markdown("""
<style>
/* ===== SIGNIFY visual layer - frontend only ===== */
.stApp {
    background: linear-gradient(135deg, #f8fbfa 0%, #f3f7f6 48%, #eef5f4 100%) !important;
}
.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Typography */
.stApp p, .stApp label, .stApp span, .stApp small,
.stApp div[data-testid="stMarkdownContainer"] p {
    color: #24343a !important;
    -webkit-text-fill-color: #24343a !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: #14282d !important;
    -webkit-text-fill-color: #14282d !important;
    letter-spacing: -0.02em;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e8f5f2 0%, #dceeea 100%) !important;
    border-right: 1px solid #c8dfda;
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.4rem;
}
section[data-testid="stSidebar"] h1 {
    font-size: 1.75rem !important;
    margin-bottom: 0.1rem !important;
}
section[data-testid="stSidebar"] .stRadio label {
    border-radius: 10px;
    padding: 0.45rem 0.55rem;
}

/* Main headings */
.hero-title {
    font-size: 2.65rem;
    font-weight: 800;
    line-height: 1.05;
    margin: 0;
    color: #12282d;
}
.hero-subtitle {
    margin-top: 0.45rem;
    color: #5d7379;
    font-size: 1.02rem;
}
.eyebrow {
    color: #168c82 !important;
    font-size: 0.78rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}

/* Cards */
.signify-card {
    background: rgba(255,255,255,0.92);
    border: 1px solid #d9e7e4;
    border-radius: 18px;
    padding: 1.15rem 1.25rem;
    box-shadow: 0 8px 28px rgba(38, 76, 80, 0.07);
    margin-bottom: 1rem;
}
.signify-card-soft {
    background: #eef8f6;
    border: 1px solid #d4eae5;
    border-radius: 16px;
    padding: 1rem 1.15rem;
    margin-bottom: 1rem;
}
.card-label {
    color: #6b7f84 !important;
    font-size: 0.74rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}
.card-value {
    color: #14282d !important;
    font-size: 1.25rem !important;
    font-weight: 750 !important;
    margin-top: 0.18rem;
}

/* Status pill */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.42rem 0.75rem;
    border-radius: 999px;
    background: #e5f7ef;
    border: 1px solid #c7e9da;
    color: #177653 !important;
    font-size: 0.82rem;
    font-weight: 750;
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #23a56d;
    display: inline-block;
}

/* Mode cards */
.mode-note {
    border-radius: 14px;
    padding: 0.9rem 1rem;
    background: #e9f3fb;
    border: 1px solid #cfe2f2;
    color: #284b61 !important;
    margin-bottom: 1rem;
}

/* Metric styling */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #dce9e6;
    border-radius: 15px;
    padding: 0.85rem 1rem;
    box-shadow: 0 5px 18px rgba(38, 76, 80, 0.05);
}
div[data-testid="stMetricLabel"] {
    color: #6b7f84 !important;
    -webkit-text-fill-color: #6b7f84 !important;
}
div[data-testid="stMetricValue"] {
    color: #16383d !important;
    -webkit-text-fill-color: #16383d !important;
}

/* Buttons */
.stButton > button {
    border-radius: 11px !important;
    border: 1px solid #b9d8d3 !important;
    background: #ffffff !important;
    color: #145d59 !important;
    -webkit-text-fill-color: #145d59 !important;
    font-weight: 700 !important;
    min-height: 2.55rem;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: #168c82 !important;
    background: #eef9f7 !important;
    transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: #168c82 !important;
    border-color: #168c82 !important;
    color: white !important;
    -webkit-text-fill-color: white !important;
}

/* Inputs */
.stTextInput input, .stSelectbox div[data-baseweb="select"],
.stTextArea textarea {
    border-radius: 10px !important;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 13px !important;
}

/* Divider */
hr {
    border-color: #d9e7e4 !important;
}

/* Camera area */
div[data-testid="stVideo"] {
    border-radius: 16px;
    overflow: hidden;
}

/* Vocabulary cards */
.vocab-card {
    background: #fff;
    border: 1px solid #dce9e6;
    border-radius: 16px;
    padding: 1.05rem;
    min-height: 130px;
    box-shadow: 0 6px 20px rgba(38, 76, 80, 0.06);
}
.vocab-icon { font-size: 1.65rem; }
.vocab-name { font-size: 1.25rem; font-weight: 800; color: #17383d !important; }
.vocab-meta { color: #6a7f84 !important; font-size: 0.86rem; margin-top: 0.25rem; }

/* Hide Streamlit decoration */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load existing models
# -----------------------------
static_model = None
try:
    with open(STATIC_MODEL_PATH, "rb") as f:
        static_model = pickle.load(f)
except Exception as e:
    st.error(f"Could not load static model: {e}")

dynamic_model = None
label_encoder = None
try:
    from tensorflow.keras.models import load_model
    dynamic_model = load_model(DYNAMIC_MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
except Exception as e:
    st.warning(f"Could not load dynamic J/Z model: {e}")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def get_saved_custom_static():
    data = []
    if not os.path.isdir(CUSTOM_VOCABULARY_PATH):
        return data
    for sign in sorted(os.listdir(CUSTOM_VOCABULARY_PATH)):
        folder = os.path.join(CUSTOM_VOCABULARY_PATH, sign)
        if not os.path.isdir(folder):
            continue
        for filename in sorted(os.listdir(folder)):
            if filename.startswith("sample_") and filename.endswith(".pkl"):
                try:
                    with open(os.path.join(folder, filename), "rb") as f:
                        x = np.asarray(pickle.load(f), dtype=np.float32).reshape(-1)
                    if x.size == 63:
                        data.append((sign, x))
                except Exception:
                    pass
    return data


def normalize_dynamic_sequence(sequence):
    """Normalize position/scale while preserving the hand's movement path."""
    arr = np.asarray(sequence, dtype=np.float32).reshape(60, 21, 3).copy()

    # Remove starting camera position using the first-frame wrist.
    start_wrist = arr[0, 0].copy()
    arr = arr - start_wrist

    # Scale using a stable hand-size measure across the sequence.
    wrist = arr[:, 0, :]
    middle_mcp = arr[:, 9, :]
    hand_scale = np.linalg.norm(middle_mcp - wrist, axis=1)
    valid = hand_scale[hand_scale > 1e-5]
    scale = float(np.median(valid)) if len(valid) else 1.0
    arr /= max(scale, 1e-5)

    return arr.reshape(60, 63)


def dtw_distance(a, b):
    """DTW distance for two 60-frame normalized sequences."""
    a = np.asarray(a, dtype=np.float32).reshape(60, 63)
    b = np.asarray(b, dtype=np.float32).reshape(60, 63)

    # Use landmark blocks and average Euclidean distance per frame.
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    n, m = cost.shape
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - 12)
        j_end = min(m, i + 12)
        for j in range(j_start, j_end + 1):
            dp[i, j] = cost[i - 1, j - 1] + min(
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1]
            )

    return float(dp[n, m] / (n + m))


def load_custom_dynamic_sequences():
    """Load every saved sequence_*.pkl from custom vocabulary."""
    data = []
    if not os.path.isdir(CUSTOM_VOCABULARY_PATH):
        return data

    for sign in sorted(os.listdir(CUSTOM_VOCABULARY_PATH)):
        folder = os.path.join(CUSTOM_VOCABULARY_PATH, sign)
        if not os.path.isdir(folder):
            continue
        for filename in sorted(os.listdir(folder)):
            if filename.startswith("sequence_") and filename.endswith(".pkl"):
                path = os.path.join(folder, filename)
                try:
                    with open(path, "rb") as f:
                        seq = np.asarray(pickle.load(f), dtype=np.float32)
                    if seq.shape == (60, 63):
                        data.append((sign, normalize_dynamic_sequence(seq)))
                except Exception:
                    pass
    return data


class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self.mode = "STATIC"
        self.static_prediction = "?"
        self.static_confidence = 0.0

        self.dynamic_prediction = "?"
        self.dynamic_confidence = 0.0
        self.dynamic_source = ""
        self.custom_dynamic_distance = None

        self.sequence = deque(maxlen=60)
        self.dynamic_frames = 0
        self.previous_wrist = None
        self.movement_started = False
        self.hand_count = 0

        # Used by Teach a New Sign.
        self.latest_hand_features = None
        self.teaching_active = False
        self.teaching_sequence = deque(maxlen=60)

        # Snapshot of custom vocabulary for this camera session.
        self.custom_dynamic_data = load_custom_dynamic_sequences()
        self.custom_static_data = get_saved_custom_static()

    def set_mode(self, mode):
        mode = mode.upper()
        if mode != self.mode:
            self.mode = mode
            self.reset_dynamic()
            self.static_prediction = "?"
            self.static_confidence = 0.0
            self.dynamic_source = ""

    def reset_dynamic(self):
        self.sequence.clear()
        self.dynamic_frames = 0
        self.dynamic_prediction = "?"
        self.dynamic_confidence = 0.0
        self.dynamic_source = ""
        self.custom_dynamic_distance = None
        self.previous_wrist = None
        self.movement_started = False

    def start_teaching(self):
        self.teaching_sequence.clear()
        self.teaching_active = True

    def stop_teaching(self):
        self.teaching_active = False

    def get_teaching_sequence(self):
        return list(self.teaching_sequence)

    def get_hand_features(self, hand):
        features = []
        for landmark in hand.landmark:
            features.extend([float(landmark.x), float(landmark.y), float(landmark.z)])
        return features

    def get_static_features(self, detected_hands):
        features = []
        for hand in detected_hands[:2]:
            features.extend(self.get_hand_features(hand))
        if len(features) < 126:
            features.extend([0.0] * (126 - len(features)))
        if len(features) > 126:
            features = features[:126]
        return np.asarray(features, dtype=np.float32).reshape(1, 126)

    def predict_static(self, detected_hands):
        if static_model is None:
            return "?", 0.0
        try:
            features = self.get_static_features(detected_hands)
            probabilities = static_model.predict_proba(features)[0]
            best = int(np.argmax(probabilities))
            return str(static_model.classes_[best]), float(probabilities[best]) * 100.0
        except Exception:
            return "?", 0.0

    def predict_existing_dynamic(self):
        if dynamic_model is None or label_encoder is None or len(self.sequence) < 60:
            return "?", 0.0
        try:
            arr = np.asarray(self.sequence, dtype=np.float32).reshape(1, 60, 63)
            probs = dynamic_model.predict(arr, verbose=0)[0]
            best = int(np.argmax(probs))
            conf = float(probs[best]) * 100.0
            pred = str(label_encoder.inverse_transform([best])[0])
            if conf >= 60.0:
                return pred, conf
            return "?", conf
        except Exception:
            return "?", 0.0

    def predict_custom_dynamic(self):
        if len(self.sequence) < 60 or not self.custom_dynamic_data:
            return "?", 0.0, None

        try:
            live = normalize_dynamic_sequence(np.asarray(self.sequence, dtype=np.float32))
            best_sign = None
            best_distance = float("inf")
            for sign, saved in self.custom_dynamic_data:
                distance = dtw_distance(live, saved)
                if distance < best_distance:
                    best_distance = distance
                    best_sign = sign

            # Convert distance to a readable confidence. This is a similarity
            # score, not a neural-network probability.
            confidence = max(0.0, min(100.0, 100.0 * np.exp(-1.6 * best_distance)))

            # This threshold is intentionally conservative for a one-sequence
            # custom vocabulary. A matching gesture should normally score well.
            if best_sign is not None and best_distance <= 0.95:
                return best_sign, float(confidence), float(best_distance)

            return "?", float(confidence), float(best_distance)
        except Exception:
            return "?", 0.0, None

    def predict_dynamic_combined(self):
        # Keep the existing J/Z GRU exactly as the primary recognizer.
        existing_pred, existing_conf = self.predict_existing_dynamic()

        # If J/Z is confidently recognized, do not disturb that working path.
        if existing_pred in ("J", "Z") and existing_conf >= 60.0:
            self.dynamic_source = "Built-in J/Z GRU"
            self.custom_dynamic_distance = None
            return existing_pred, existing_conf

        # Otherwise try the user's taught dynamic signs.
        custom_pred, custom_conf, distance = self.predict_custom_dynamic()
        self.custom_dynamic_distance = distance
        if custom_pred != "?":
            self.dynamic_source = "My Vocabulary"
            return custom_pred, custom_conf

        self.dynamic_source = ""
        return existing_pred, existing_conf

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            self.hand_count = 0
            self.latest_hand_features = None
            self.static_prediction = "?"
            self.static_confidence = 0.0
            self.reset_dynamic()
            # Do not destroy a teaching sequence already captured.
            return av.VideoFrame.from_ndarray(image, format="bgr24")

        detected_hands = results.multi_hand_landmarks[:2]
        self.hand_count = len(detected_hands)

        self.latest_hand_features = np.asarray(
            self.get_hand_features(detected_hands[0]), dtype=np.float32
        )

        # Separate teaching path. It does not change J/Z recognition.
        if self.teaching_active and len(self.teaching_sequence) < 60:
            self.teaching_sequence.append(self.latest_hand_features.copy())
            if len(self.teaching_sequence) >= 60:
                self.teaching_active = False

        for hand in detected_hands:
            mp_draw.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

        if self.mode == "STATIC":
            self.static_prediction, self.static_confidence = self.predict_static(detected_hands)
            self.reset_dynamic()
        else:
            first_hand = detected_hands[0]
            dynamic_features = self.get_hand_features(first_hand)
            wrist = first_hand.landmark[0]
            current_wrist = np.array([wrist.x, wrist.y], dtype=np.float32)
            movement = 0.0
            if self.previous_wrist is not None:
                movement = float(np.linalg.norm(current_wrist - self.previous_wrist))
            self.previous_wrist = current_wrist

            # KEEP EXISTING J/Z SETTING.
            MOVEMENT_THRESHOLD = 0.015
            if movement > MOVEMENT_THRESHOLD:
                self.movement_started = True

            if self.movement_started:
                self.sequence.append(dynamic_features)
                self.dynamic_frames = len(self.sequence)

            if self.dynamic_frames >= 60:
                self.dynamic_prediction, self.dynamic_confidence = self.predict_dynamic_combined()

        return av.VideoFrame.from_ndarray(image, format="bgr24")


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown('<div class="eyebrow">AI • COMPUTER VISION</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title" style="font-size:1.75rem;">🤟 SIGNIFY</div>', unsafe_allow_html=True)
    st.caption("Real-Time Sign Language Translation")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Live Translation", "Teach a New Sign", "My Vocabulary", "History", "Settings"],
    )
    st.divider()
    st.markdown('<div class="status-pill"><span class="status-dot"></span> System Online</div>', unsafe_allow_html=True)
    st.caption("MediaPipe • Random Forest • GRU + DTW")


# -----------------------------
# Live Translation
# -----------------------------
if page == "Live Translation":
    top_left, top_right = st.columns([4, 1])
    with top_left:
        st.markdown('<div class="eyebrow">SIGNIFY / LIVE</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">Live Translation</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-subtitle">Perform an ASL gesture and see the recognition result in real time.</div>', unsafe_allow_html=True)
    with top_right:
        st.markdown('<div style="text-align:right; padding-top:0.4rem;"><span class="status-pill"><span class="status-dot"></span> LIVE</span></div>', unsafe_allow_html=True)

    st.write("")
    selected_mode = st.radio("Recognition Mode", ["Static", "Dynamic"], horizontal=True)

    if selected_mode == "Static":
        st.markdown('<div class="mode-note">✋ <b>Static Mode</b> · Hold a stationary ASL alphabet sign. A–Y are recognized here.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="mode-note">🔄 <b>Dynamic Mode</b> · J/Z use the existing GRU. Custom movement signs from My Vocabulary are checked after 60 frames.</div>', unsafe_allow_html=True)

    camera_column, recognition_column = st.columns([1.65, 1], gap="large")

    with camera_column:
        st.markdown('<div class="signify-card">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">CAMERA INPUT</div>', unsafe_allow_html=True)
        st.subheader("📷 Live Camera")
        ctx = webrtc_streamer(
            key="signify-live-camera",
            video_processor_factory=SignLanguageProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with recognition_column:
        st.markdown('<div class="eyebrow">AI RECOGNITION</div>', unsafe_allow_html=True)
        status_placeholder = st.empty()
        result_placeholder = st.empty()
        confidence_placeholder = st.empty()
        hands_placeholder = st.empty()
        progress_placeholder = st.empty()
        message_placeholder = st.empty()
        source_placeholder = st.empty()

        if ctx is not None and ctx.video_processor is not None:
            processor = ctx.video_processor
            processor.set_mode(selected_mode)
        else:
            processor = None

        if processor is not None and ctx.state.playing:
            while ctx.state.playing:
                processor.set_mode(selected_mode)
                hand_count = processor.hand_count
                static_prediction = processor.static_prediction
                static_confidence = processor.static_confidence
                dynamic_prediction = processor.dynamic_prediction
                dynamic_confidence = processor.dynamic_confidence
                dynamic_frames = processor.dynamic_frames

                if selected_mode == "Static":
                    status_placeholder.markdown('<div class="signify-card"> <div class="card-label">Recognition</div> <div class="card-value">Static Alphabet</div>', unsafe_allow_html=True)
                    result_placeholder.metric("Detected Sign", static_prediction)
                    confidence_placeholder.metric("Confidence", f"{static_confidence:.1f}%")
                    hands_placeholder.metric("Hands Detected", hand_count)
                    progress_placeholder.empty()
                    source_placeholder.empty()
                    if hand_count == 0:
                        message_placeholder.warning("Show your hand clearly in front of the camera.")
                    else:
                        message_placeholder.success(f"Recognized ASL Letter: {static_prediction}")
                    status_placeholder.markdown('</div>', unsafe_allow_html=True)
                else:
                    status_placeholder.markdown('<div class="signify-card"> <div class="card-label">Recognition</div> <div class="card-value">Dynamic Gesture</div>', unsafe_allow_html=True)
                    progress_placeholder.write(f"Frames captured: {dynamic_frames}/60")
                    progress_placeholder.progress(max(0.0, min(1.0, dynamic_frames / 60.0)))
                    result_placeholder.metric("Prediction", dynamic_prediction)
                    confidence_placeholder.metric("Confidence", f"{dynamic_confidence:.1f}%")
                    hands_placeholder.metric("Hands Detected", hand_count)
                    if processor.dynamic_source:
                        source_placeholder.caption(f"Source: {processor.dynamic_source}")
                    else:
                        source_placeholder.empty()

                    if hand_count == 0:
                        message_placeholder.warning("Show your hand and perform the gesture.")
                    elif dynamic_frames > 0 and dynamic_frames < 60:
                        message_placeholder.info("🔄 Keep performing the movement...")
                    elif dynamic_prediction != "?":
                        if processor.dynamic_source == "My Vocabulary":
                            distance_text = ""
                            if processor.custom_dynamic_distance is not None:
                                distance_text = f"  |  Match distance: {processor.custom_dynamic_distance:.3f}"
                            message_placeholder.success(f"🤟 Custom sign recognized: {dynamic_prediction}{distance_text}")
                        else:
                            message_placeholder.success(f"Recognized Dynamic ASL Gesture: {dynamic_prediction}")
                    else:
                        message_placeholder.info("No matching dynamic sign found. Try the gesture again.")
                    status_placeholder.markdown('</div>', unsafe_allow_html=True)
                time.sleep(0.1)

    st.write("")
    st.markdown('<div class="eyebrow">SYSTEM COMPONENTS</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Static Model", "Random Forest")
    with c2:
        st.metric("Static Features", "126")
    with c3:
        st.metric("Dynamic Model", "GRU + Custom DTW")


# -----------------------------
# Teach a New Sign
# -----------------------------
elif page == "Teach a New Sign":
    st.markdown('<div class="eyebrow">SIGNIFY / PERSONALIZATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Teach a New Sign</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Create a custom sign that can be stored in your personal vocabulary.</div>', unsafe_allow_html=True)
    st.write("")

    if "custom_samples" not in st.session_state:
        st.session_state.custom_samples = []
    if "custom_sequence" not in st.session_state:
        st.session_state.custom_sequence = []
    if "custom_sign_name" not in st.session_state:
        st.session_state.custom_sign_name = ""

    st.markdown('<div class="signify-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">STEP 01 · SIGN IDENTITY</div>', unsafe_allow_html=True)
    sign_name_input = st.text_input(
        "Enter the name of the sign",
        value=st.session_state.custom_sign_name,
        placeholder="Example: HELLO",
    )
    sign_name = sign_name_input.strip().upper().replace(" ", "_")
    if sign_name:
        st.session_state.custom_sign_name = sign_name
        st.success(f"Sign name: {sign_name}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="signify-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">STEP 02 · GESTURE TYPE</div>', unsafe_allow_html=True)
    sign_type = st.radio(
        "Is this a stationary or movement-based sign?",
        ["Static", "Dynamic"],
        horizontal=True,
    )
    if sign_type == "Static":
        st.markdown('<div class="mode-note">✋ <b>Static Sign</b> · Capture 20 individual stationary hand samples.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="mode-note">🔄 <b>Dynamic Sign</b> · Capture one complete movement as 60 consecutive frames. Example: HELLO.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    camera_column, control_column = st.columns([1.65, 1], gap="large")

    with camera_column:
        st.markdown('<div class="signify-card">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">CAMERA INPUT</div>', unsafe_allow_html=True)
        st.subheader("📷 Teaching Camera")
        teach_ctx = webrtc_streamer(
            key="teach-custom-sign-camera",
            video_processor_factory=SignLanguageProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with control_column:
        st.markdown('<div class="signify-card">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">CAPTURE</div>', unsafe_allow_html=True)
        teaching_processor = teach_ctx.video_processor if teach_ctx is not None else None

        if sign_type == "Static":
            count = len(st.session_state.custom_samples)
            st.metric("Static Samples", f"{count}/20")
            st.progress(min(count / 20.0, 1.0))

            if st.button("📸 Capture Sample", use_container_width=True):
                if not sign_name:
                    st.warning("Enter a sign name first.")
                elif teaching_processor is None:
                    st.warning("Start the camera first.")
                elif teaching_processor.hand_count == 0 or teaching_processor.latest_hand_features is None:
                    st.warning("🖐️ No hand data yet. Show your hand clearly.")
                elif count >= 20:
                    st.warning("You already captured 20 samples.")
                else:
                    st.session_state.custom_samples.append(teaching_processor.latest_hand_features.copy())
                    st.success(f"✅ Sample {count + 1}/20 captured!")

            if st.button("🗑️ Clear Static Samples", use_container_width=True):
                st.session_state.custom_samples = []
                st.success("Static samples cleared.")
        else:
            count = len(st.session_state.custom_sequence)
            if teaching_processor is not None and teaching_processor.teaching_active:
                count = len(teaching_processor.teaching_sequence)
            st.metric("Dynamic Frames", f"{count}/60")
            st.progress(min(count / 60.0, 1.0))

            if st.button(
                "🔄 Start Dynamic Capture",
                use_container_width=True,
                disabled=(teaching_processor is None or (teaching_processor is not None and teaching_processor.teaching_active)),
            ):
                if not sign_name:
                    st.warning("Enter a sign name first.")
                elif teaching_processor is None:
                    st.warning("Start the camera first.")
                elif teaching_processor.hand_count == 0:
                    st.warning("🖐️ Show your hand clearly before starting.")
                else:
                    st.session_state.custom_sequence = []
                    teaching_processor.start_teaching()
                    st.rerun()

            if teaching_processor is not None and teaching_processor.teaching_active:
                st.info(f"🔴 Capturing automatically: {len(teaching_processor.teaching_sequence)}/60 frames")

            if teaching_processor is not None and not teaching_processor.teaching_active:
                seq = teaching_processor.get_teaching_sequence()
                if len(seq) >= 60 and len(st.session_state.custom_sequence) != 60:
                    st.session_state.custom_sequence = seq[:60]
                    st.success("✅ Complete 60-frame gesture captured!")

            if st.button("🗑️ Clear Dynamic Capture", use_container_width=True):
                if teaching_processor is not None:
                    teaching_processor.stop_teaching()
                    teaching_processor.teaching_sequence.clear()
                st.session_state.custom_sequence = []
                st.success("Dynamic capture cleared.")
        st.markdown('</div>', unsafe_allow_html=True)

    if sign_type == "Dynamic" and teaching_processor is not None and teach_ctx.state.playing and teaching_processor.teaching_active:
        progress_slot = st.empty()
        while teach_ctx.state.playing and teaching_processor.teaching_active:
            n = len(teaching_processor.teaching_sequence)
            progress_slot.progress(min(n / 60.0, 1.0), text=f"Capturing dynamic sign: {n}/60 frames")
            time.sleep(0.08)
        final_seq = teaching_processor.get_teaching_sequence()
        if len(final_seq) >= 60:
            st.session_state.custom_sequence = final_seq[:60]
            progress_slot.progress(1.0, text="Dynamic capture complete: 60/60 frames")
            st.success("🎉 Dynamic gesture captured. You can save it now.")

    st.markdown('<div class="signify-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">STEP 03 · SAVE TO VOCABULARY</div>', unsafe_allow_html=True)

    if sign_type == "Static":
        count = len(st.session_state.custom_samples)
        if count < 20:
            st.warning(f"Capture {20 - count} more sample(s) before saving.")
        else:
            st.success("🎉 20 static samples are ready to save!")

        if st.button("💾 Save Static Sign", use_container_width=True):
            if not sign_name:
                st.warning("Please enter a sign name.")
            elif count < 20:
                st.warning("Please capture all 20 samples first.")
            else:
                folder = os.path.join(CUSTOM_VOCABULARY_PATH, sign_name)
                os.makedirs(folder, exist_ok=True)
                existing = []
                for fn in os.listdir(folder):
                    if fn.startswith("sample_") and fn.endswith(".pkl"):
                        try:
                            existing.append(int(fn[7:-4]))
                        except ValueError:
                            pass
                start = max(existing) + 1 if existing else 0
                for i, sample in enumerate(st.session_state.custom_samples):
                    with open(os.path.join(folder, f"sample_{start + i}.pkl"), "wb") as f:
                        pickle.dump(np.asarray(sample, dtype=np.float32).reshape(63), f)
                st.session_state.custom_samples = []
                st.success(f"🎉 Static sign '{sign_name}' saved successfully!")
                st.info(f"Saved inside: Data/custom_vocabulary/{sign_name}")
                st.balloons()
    else:
        count = len(st.session_state.custom_sequence)
        if count < 60:
            st.warning("Capture a complete 60-frame gesture before saving.")
        else:
            st.success("🎉 60-frame dynamic sequence is ready to save!")

        if st.button("💾 Save Dynamic Sign", use_container_width=True):
            if not sign_name:
                st.warning("Please enter a sign name.")
            elif count < 60:
                st.warning("Please capture 60 frames first.")
            else:
                folder = os.path.join(CUSTOM_VOCABULARY_PATH, sign_name)
                os.makedirs(folder, exist_ok=True)
                existing = []
                for fn in os.listdir(folder):
                    if fn.startswith("sequence_") and fn.endswith(".pkl"):
                        try:
                            existing.append(int(fn[9:-4]))
                        except ValueError:
                            pass
                idx = max(existing) + 1 if existing else 0
                seq = np.asarray(st.session_state.custom_sequence, dtype=np.float32)
                if seq.shape != (60, 63):
                    st.error(f"Invalid sequence shape {seq.shape}; expected (60, 63).")
                else:
                    with open(os.path.join(folder, f"sequence_{idx}.pkl"), "wb") as f:
                        pickle.dump(seq, f)
                    st.session_state.custom_sequence = []
                    st.success(f"🎉 Dynamic sign '{sign_name}' saved successfully!")
                    st.info(f"Saved inside: Data/custom_vocabulary/{sign_name}")
                    st.warning("Restart the Live Translation camera once so it reloads the new vocabulary.")
                    st.balloons()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="signify-card-soft">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">CURRENT CAPTURE</div>', unsafe_allow_html=True)
    st.write(f"**Sign:** {sign_name if sign_name else 'Not entered'}")
    st.write(f"**Type:** {sign_type}")
    if sign_type == "Static":
        st.write(f"**Samples:** {len(st.session_state.custom_samples)}/20")
    else:
        st.write(f"**Frames:** {len(st.session_state.custom_sequence)}/60")
    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# My Vocabulary
# -----------------------------
elif page == "My Vocabulary":
    st.markdown('<div class="eyebrow">SIGNIFY / PERSONAL VOCABULARY</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">My Vocabulary</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Your custom signs are stored in Data/custom_vocabulary.</div>', unsafe_allow_html=True)
    st.write("")

    folders = []
    for item in os.listdir(CUSTOM_VOCABULARY_PATH):
        path = os.path.join(CUSTOM_VOCABULARY_PATH, item)
        if os.path.isdir(path):
            folders.append(item)

    if not folders:
        st.info("No custom signs have been added yet.")
    else:
        st.success(f"{len(folders)} custom sign(s) available.")
        cols = st.columns(3, gap="medium")
        for idx, sign in enumerate(sorted(folders)):
            path = os.path.join(CUSTOM_VOCABULARY_PATH, sign)
            static_count = len([f for f in os.listdir(path) if f.startswith("sample_") and f.endswith(".pkl")])
            dynamic_count = len([f for f in os.listdir(path) if f.startswith("sequence_") and f.endswith(".pkl")])
            with cols[idx % 3]:
                st.markdown('<div class="vocab-card">', unsafe_allow_html=True)
                st.markdown('<div class="vocab-icon">🤟</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="vocab-name">{sign}</div>', unsafe_allow_html=True)
                if static_count:
                    st.markdown(f'<div class="vocab-meta">Static · {static_count} samples</div>', unsafe_allow_html=True)
                if dynamic_count:
                    st.markdown(f'<div class="vocab-meta">Dynamic · {dynamic_count} sequence(s)</div>', unsafe_allow_html=True)
                if not static_count and not dynamic_count:
                    st.markdown('<div class="vocab-meta">No valid saved samples yet.</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# History
# -----------------------------
elif page == "History":
    st.markdown('<div class="eyebrow">SIGNIFY / ACTIVITY</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Recognition History</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Previously recognized signs will appear here.</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="signify-card-soft">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">HISTORY</div>', unsafe_allow_html=True)
    st.info("Recognition history is currently empty.")
    st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------
# Settings
# -----------------------------
elif page == "Settings":
    st.markdown('<div class="eyebrow">SIGNIFY / PREFERENCES</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Adjust the visual and camera preferences for the application.</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="signify-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">CAMERA DISPLAY</div>', unsafe_allow_html=True)
    st.checkbox("Mirror camera", value=True)
    st.checkbox("Show hand landmarks", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

