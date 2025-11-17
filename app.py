# app.py
import os
import re
import tempfile
import subprocess
import json
from typing import List, Dict, Optional, Tuple

import streamlit as st
import yt_dlp
import urllib.request
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Optional: whisper (offline transcription)
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# ---------------------------
# Config & constants
# ---------------------------
st.set_page_config(page_title="Video Relevance Analyzer", page_icon="🎥", layout="wide")
PROMO_KEYWORDS = [
    'subscribe', 'like', 'comment', 'share', 'notification', 'bell', 'sponsor',
    'advertisement', 'promo code', 'discount', 'affiliate', 'check out', 'link in description',
    'merch', 'patreon', 'support', 'follow me', 'follow us', 'instagram', 'twitter',
    'facebook', 'tiktok', 'buy now', 'sign up', 'download', 'visit', 'coupon'
]

# ---------------------------
# Session state initialization
# ---------------------------
if 'video_info' not in st.session_state:
    st.session_state.video_info = None
if 'transcript_segments' not in st.session_state:
    st.session_state.transcript_segments = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'last_params' not in st.session_state:
    st.session_state.last_params = {}
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'promo_clf' not in st.session_state:
    st.session_state.promo_clf = None
if 'openai_key' not in st.session_state:
    st.session_state.openai_key = None

# ---------------------------
# Utility functions
# ---------------------------
def extract_video_id(url: str) -> Optional[str]:
    if not url or not isinstance(url, str):
        return None
    patterns = [
        r"v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"embed/([A-Za-z0-9_-]{11})",
        r"shorts/([A-Za-z0-9_-]{11})"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

def parse_vtt_or_srt(content: str) -> List[Dict]:
    """
    Parse simple VTT or SRT content into list of segments with start,end,text.
    This is a best-effort parser for common VTT/SRT formats.
    """
    lines = content.splitlines()
    segments = []
    cur_text = []
    cur_start = None
    cur_end = None

    time_re = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}:\d{2},\d{3})')

    def to_seconds(t: str) -> float:
        t = t.replace(',', '.')
        parts = t.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return 0.0

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # skip sequence numbers
        if re.match(r'^\d+$', line):
            i += 1
            continue

        m = time_re.search(line)
        if m:
            cur_start = to_seconds(m.group(1))
            cur_end = to_seconds(m.group(2))
            # gather following lines until blank
            cur_text = []
            i += 1
            while i < len(lines) and lines[i].strip():
                cur_text.append(re.sub(r'<[^>]+>', '', lines[i].strip()))
                i += 1
            segments.append({
                'start': cur_start,
                'end': cur_end,
                'text': ' '.join(cur_text)
            })
            continue
        else:
            i += 1

    return segments

def join_segments_to_string(segments: List[Dict]) -> str:
    return " ".join([s.get('text', '') for s in segments]).strip()

def estimate_segment_times_by_words(transcript_text: str, duration: float, seg_word_length: int = 200, overlap: int = 50) -> List[Dict]:
    """
    If we only have a plain transcript string and the video duration,
    estimate timestamps by distributing words linearly over duration.
    Returns segments with 'start', 'end', 'text'.
    """
    words = transcript_text.split()
    if len(words) == 0 or duration <= 0:
        # fallback single segment (no times)
        return [{'start': 0.0, 'end': duration if duration>0 else 0.0, 'text': transcript_text}]
    segments = []
    i = 0
    step = seg_word_length - overlap if seg_word_length > overlap else seg_word_length
    total_words = len(words)
    while i < total_words:
        seg_words = words[i:i+seg_word_length]
        start_idx = i
        end_idx = i + len(seg_words) - 1
        start_time = (start_idx / total_words) * duration
        end_time = (end_idx / total_words) * duration
        segments.append({
            'start': float(start_time),
            'end': float(end_time),
            'text': ' '.join(seg_words)
        })
        i += step
    return segments

def get_first_suburl_from_info(info: dict) -> Optional[str]:
    for key in ('subtitles', 'automatic_captions'):
        block = info.get(key, {}) or {}
        if 'en' in block:
            candidates = block.get('en', [])
        else:
            candidates = []
            for lang, formats in block.items():
                if formats:
                    candidates = formats
                    break
        if candidates:
            chosen = None
            for fmt in candidates:
                if fmt.get('ext') == 'vtt':
                    chosen = fmt
                    break
            if not chosen:
                chosen = candidates[0]
            if chosen and chosen.get('url'):
                return chosen.get('url')
    return None

# ---------------------------
# Extraction & transcription
# ---------------------------
def extract_metadata_with_ytdlp(url_or_path: str) -> Optional[dict]:
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(url_or_path, download=False)
            return info
    except Exception as e:
        return None

def download_audio_from_video_file(uploaded_file, dest_path: str) -> str:
    """
    Save uploaded file to temp and extract audio using ffmpeg to dest_path (mp3/wav).
    uploaded_file: Streamlit UploadedFile
    """
    temp_video = os.path.join(tempfile.gettempdir(), f"uploaded_video_{os.path.basename(uploaded_file.name)}")
    with open(temp_video, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # extract audio using ffmpeg
    audio_path = dest_path
    cmd = [
        "ffmpeg", "-y", "-i", temp_video,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

def download_audio_from_url(url: str, dest_path: str) -> Optional[str]:
    """
    Use yt-dlp to download the best audio into dest_path.
    """
    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": dest_path,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192"
            }]
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return dest_path
    except Exception as e:
        return None

def transcribe_with_whisper_local(audio_path: str, model_size: str = "small") -> List[Dict]:
    """
    Uses openai/whisper local package to transcribe and return segments with timestamps.
    Requires 'whisper' package.
    """
    if not WHISPER_AVAILABLE:
        raise RuntimeError("Whisper package not available. Please install openai-whisper.")
    model = whisper.load_model(model_size)
    # transcribe with timestamps
    result = model.transcribe(audio_path, verbose=False)
    # result['segments'] is a list with dicts that include 'start','end','text'
    segs = []
    for seg in result.get('segments', []):
        segs.append({'start': float(seg['start']), 'end': float(seg['end']), 'text': seg['text'].strip()})
    return segs

def extract_video_info(url_or_file: Optional[st.runtime.uploaded_file_manager.UploadedFile], youtube_url: Optional[str]) -> dict:
    """
    Main function: get title/description/duration and a timestamped transcript (if possible).
    Returns dictionary:
    {
        "title": str,
        "description": str,
        "duration": float,
        "transcript_segments": [{'start', 'end', 'text'}, ...] or None,
        "raw_transcript": str or None
    }
    """
    info = {
        "title": None,
        "description": "",
        "duration": 0.0,
        "transcript_segments": None,
        "raw_transcript": None
    }

    # If uploaded file -> try to extract audio and run Whisper (offline)
    if url_or_file and hasattr(url_or_file, "getbuffer"):
        st.info("Processing uploaded video — extracting audio and running Whisper (offline). This may take time.")
        tmp_audio = os.path.join(tempfile.gettempdir(), f"audio_{os.path.basename(url_or_file.name)}.wav")
        try:
            audio_path = download_audio_from_video_file(url_or_file, tmp_audio)
        except Exception as e:
            st.error(f"Audio extraction failed: {e}")
            return info
        if WHISPER_AVAILABLE:
            try:
                segments = transcribe_with_whisper_local(audio_path, model_size="small")
                combined = join_segments_to_string(segments)
                info.update({
                    "title": getattr(url_or_file, "name", "Uploaded Video"),
                    "description": "",
                    "duration": max([s['end'] for s in segments]) if segments else 0.0,
                    "transcript_segments": segments,
                    "raw_transcript": combined
                })
                return info
            except Exception as e:
                st.warning(f"Whisper transcription failed: {e}. Falling back to no transcript.")
                return info
        else:
            st.warning("Whisper is not available in this environment. Install openai-whisper to enable offline transcription.")
            return info

    # If youtube_url provided: use ytdlp metadata & try YouTubeTranscriptApi -> vtt fallback
    if youtube_url:
        video_id = extract_video_id(youtube_url)
        # metadata via yt-dlp
        info_meta = extract_metadata_with_ytdlp(youtube_url)
        if info_meta:
            title = info_meta.get("title", "Unknown")
            description = info_meta.get("description", "") or ""
            duration = float(info_meta.get("duration", 0) or 0.0)
            info.update({"title": title, "description": description, "duration": duration})
        else:
            info.update({"title": "YouTube Video", "description": "", "duration": 0.0})

        # Try YouTubeTranscriptApi
        if video_id:
            try:
                raw_list = YouTubeTranscriptApi.get_transcript(video_id)
                segs = []
                for t in raw_list:
                    start = float(t.get("start", 0.0))
                    dur = float(t.get("duration", 0.0))
                    text = t.get("text", "").strip()
                    segs.append({'start': start, 'end': start + dur, 'text': text})
                info.update({
                    "transcript_segments": segs,
                    "raw_transcript": join_segments_to_string(segs)
                })
                return info
            except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript):
                # try english
                try:
                    raw_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                    segs = []
                    for t in raw_list:
                        start = float(t.get("start", 0.0))
                        dur = float(t.get("duration", 0.0))
                        text = t.get("text", "").strip()
                        segs.append({'start': start, 'end': start + dur, 'text': text})
                    info.update({
                        "transcript_segments": segs,
                        "raw_transcript": join_segments_to_string(segs)
                    })
                    return info
                except Exception:
                    pass
            except Exception:
                pass

        # Fallback: use subtitles URL from yt-dlp info
        if info_meta:
            sub_url = get_first_suburl_from_info(info_meta)
            if sub_url:
                try:
                    with urllib.request.urlopen(sub_url) as resp:
                        raw = resp.read().decode('utf-8', errors='ignore')
                        segs = parse_vtt_or_srt(raw)
                        if segs:
                            info.update({
                                "transcript_segments": segs,
                                "raw_transcript": join_segments_to_string(segs)
                            })
                            return info
                except Exception:
                    pass

        # No transcript found — return metadata only
        return info

    return info

# ---------------------------
# Embedding & scoring
# ---------------------------
@st.cache_resource
def load_embedding_model(name="all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

def compute_segment_embeddings(model, segments: List[Dict]) -> np.ndarray:
    texts = [s['text'] for s in segments]
    if len(texts) == 0:
        return np.zeros((0, model.get_sentence_embedding_dimension()))
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embs

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: (m, d), b: (n, d) -> return (m, n)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0] if b.size else 0))
    # normalized
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-8)
    b_n = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-8)
    return np.dot(a_n, b_n.T)

def per_segment_score(similarity: float, is_promo: bool, is_offtopic: bool) -> float:
    sim = float(similarity) if not np.isnan(similarity) else 0.0
    sim = max(0.0, sim)
    score = sim * 100.0
    if is_promo:
        score *= 0.40
    if is_offtopic:
        score *= 0.2
    return round(max(0.0, min(100.0, score)), 2)

def compute_overall_score(similarities: List[float], promo_flags: List[bool], threshold: float = 0.45) -> Tuple[float, dict]:
    sims = np.array(similarities, dtype=float) if len(similarities) else np.array([0.0])
    sims = np.nan_to_num(sims)
    sims = np.clip(sims, 0.0, 1.0)
    avg_sim = float(sims.mean())
    coverage = float((sims >= threshold).sum() / len(sims))
    promo_prop = float(sum(1 for p in promo_flags if p) / max(1, len(promo_flags)))
    raw = 0.6 * avg_sim + 0.3 * coverage - 0.2 * promo_prop
    raw = max(0.0, raw)
    score = round(min(100.0, raw * 100.0), 2)
    return score, {'avg_similarity': avg_sim, 'coverage': coverage, 'promo_proportion': promo_prop, 'raw': raw}

# ---------------------------
# Promo classifier (toy/demo)
# ---------------------------
def build_promo_classifier():
    """
    Build a small demo classifier that distinguishes promotional text vs non-promotional.
    We create a tiny synthetic dataset and train a TF-IDF + LogisticRegression pipeline.
    This is for demo only and should be replaced by a real dataset for production.
    """
    positive = [
        "Subscribe to my channel and hit the bell", "Use promo code SAVE20 to get discount",
        "Check the link in the description to buy now", "Sponsored by CompanyX", "Get 50% off now",
        "Visit our website to sign up", "Follow me on Instagram for updates", "Support us on Patreon"
    ]
    negative = [
        "In this lecture we discuss machine learning concepts", "Today we cover linear regression",
        "This tutorial explains how neural networks work", "Let's derive the formula for gradient descent",
        "We will analyze experimental results and metrics"
    ]
    X = positive + negative
    y = [1]*len(positive) + [0]*len(negative)
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=1000)),
        ('clf', LogisticRegression())
    ])
    pipe.fit(X, y)
    return pipe

def detect_promo_with_classifier(text: str, clf_pipeline) -> Tuple[bool, float]:
    if not text or not text.strip():
        return False, 0.0
    # keyword heuristic score
    text_lower = text.lower()
    kw_count = sum(1 for kw in PROMO_KEYWORDS if kw in text_lower)
    kw_score = min(1.0, kw_count / 3.0)
    # classifier probability
    try:
        prob = clf_pipeline.predict_proba([text])[0][1]
    except Exception:
        prob = 0.0
    combined = 0.6 * prob + 0.4 * kw_score
    is_promo = combined >= 0.4
    return bool(is_promo), float(round(combined, 3))

# ---------------------------
# Explanation (LLM) generator
# ---------------------------
def generate_explanation_with_openai(overall_score: float, comps: dict, flagged_examples: List[Dict], openai_api_key: Optional[str], title: str) -> str:
    """
    If the user provides OpenAI API key, call the API to generate a polished explanation.
    Otherwise, return a deterministic template.
    """
    if openai_api_key:
        try:
            import openai
            openai.api_key = openai_api_key
            prompt = f"""
You are a concise assistant. Given a video titled: "{title}".
Overall relevance score: {overall_score:.1f}%.
Components: avg_similarity={comps['avg_similarity']:.3f}, coverage={comps['coverage']:.3f}, promo_prop={comps['promo_proportion']:.3f}.
Provide a 2-3 sentence human-friendly justification and list up to 4 short timestamp examples (time range and one-line reason).
Examples (json array) should look like: [{"start":0.0,"end":10.2,"note":"promotional - mentions subscribe"}]
Flag promotional content and filler briefly.
"""
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini" if False else "gpt-4o" if False else "gpt-4", # user may replace model name; this is sample
                messages=[{"role":"user","content":prompt}],
                max_tokens=300,
                temperature=0.2
            )
            txt = response['choices'][0]['message']['content'].strip()
            return txt
        except Exception as e:
            # fallback template if OpenAI fails / not configured
            pass

    # Fallback deterministic template
    lines = []
    lines.append(f"Overall relevance: {overall_score:.1f}%.")
    lines.append(f"Average similarity to title/description: {comps['avg_similarity']:.3f}. Coverage ≥ threshold: {comps['coverage']:.2%}. Promotional proportion: {comps['promo_proportion']:.2%}.")
    if flagged_examples:
        lines.append("Examples:")
        for ex in flagged_examples[:4]:
            start = ex.get('start', 0.0)
            end = ex.get('end', start + 1.0)
            note = ex.get('note', '')
            lines.append(f"- {start:.1f}s–{end:.1f}s: {note}")
    else:
        lines.append("No strongly promotional or off-topic segments were detected by the heuristics/classifier.")
    return " ".join(lines)

# ---------------------------
# App UI & main logic
# ---------------------------
def analyze_and_score(segments: List[Dict], title: str, description: str, duration: float, similarity_threshold: float, seg_by_words: int):
    """
    Compute embeddings, similarities, per-segment scores, detect promo, and aggregate overall score.
    Returns (results_list, overall_score, comps)
    results_list: [{'segment_id','start','end','text','similarity','relevance_score','is_promo','promo_prob'}...]
    """
    model = st.session_state.embedding_model or load_embedding_model()
    st.session_state.embedding_model = model

    # Build context embedding from title + description
    context_text = title.strip() + ". " + (description.strip()[:1000] if description else "")
    context_emb = model.encode([context_text], show_progress_bar=False, convert_to_numpy=True)
    # compute segment embeddings
    segment_embs = compute_segment_embeddings(model, segments)
    if segment_embs.size == 0:
        st.error("No segment embeddings were produced.")
        return [], 0.0, {}

    sims = cosine_similarity_matrix(context_emb, segment_embs)[0]  # shape (n_segments,)
    # Ensure promo classifier exists
    if st.session_state.promo_clf is None:
        st.session_state.promo_clf = build_promo_classifier()
    clf = st.session_state.promo_clf

    results = []
    promo_flags = []
    for i, (seg, sim) in enumerate(zip(segments, sims), start=1):
        is_promo, promo_prob = detect_promo_with_classifier(seg.get('text',''), clf)
        # rudimentary off-topic detection: very short segments with low sim
        is_offtopic = (len(seg.get('text','').split()) < 5 and sim < 0.2)
        seg_score = per_segment_score(sim, is_promo, is_offtopic)
        results.append({
            'segment_id': i,
            'start': seg.get('start', 0.0),
            'end': seg.get('end', seg.get('start', 0.0)),
            'text': seg.get('text', ''),
            'similarity': float(sim),
            'relevance_score': seg_score,
            'is_promo': bool(is_promo),
            'promo_prob': float(promo_prob),
            'is_offtopic': bool(is_offtopic)
        })
        promo_flags.append(bool(is_promo))

    overall_score, comps = compute_overall_score([r['similarity'] for r in results], promo_flags, threshold=similarity_threshold)
    return results, overall_score, comps

def main():
    st.title("🎥 Video Relevance Analyzer")
    st.markdown("Upload a video file (runs Whisper offline) **or** paste a YouTube URL. The app extracts/fetches transcript, computes semantic relevance to title/description, flags promotional segments, and shows a timeline heatmap.")

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.header("Input Options")
        upload_mode = st.radio("Choose input method:", ["Paste YouTube URL", "Upload video file"])
        youtube_url, uploaded_file = None, None
        if upload_mode == "Paste YouTube URL":
            youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        else:
            uploaded_file = st.file_uploader("Upload video file (mp4, mov, mkv, ...)", type=["mp4","mov","mkv","webm","avi"])

        st.markdown("---")
        st.subheader("Segmentation & thresholds")
        seg_length = st.slider("Segment length (words)", 100, 500, 200, step=50)
        overlap = st.slider("Overlap (words)", 0, 150, 50, step=10)
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.45, step=0.01)
        last_params = st.session_state.get('last_params', {})
        if last_params.get('seg_length') != seg_length or last_params.get('overlap') != overlap or last_params.get('similarity_threshold') != similarity_threshold:
            st.session_state.transcript_segments = None
            st.session_state.results = None
        st.session_state.last_params = {'seg_length': seg_length, 'overlap': overlap, 'similarity_threshold': similarity_threshold}

        st.markdown("---")
        st.subheader("LLM / API (optional)")
        st.text("Provide OpenAI API key for polished explanation (optional).")
        api_key_input = st.text_input("OpenAI API Key", type="password")
        if api_key_input:
            st.session_state.openai_key = api_key_input

        st.markdown("---")
        st.markdown("**Notes:** \n- Whisper requires `ffmpeg` and may be slow on CPU.\n- Edit transcripts after fetching if needed.")

        if st.button("Fetch / Transcribe & Analyze"):
            with st.spinner("Processing video..."):
                info = extract_video_info(uploaded_file, youtube_url)
                if not info:
                    st.error("Failed to extract video info.")
                    return
                st.session_state.video_info = {
                    'title': info.get('title',''),
                    'description': info.get('description',''),
                    'duration': float(info.get('duration', 0.0))
                }
                transcript_segments = info.get('transcript_segments')
                raw_transcript = info.get('raw_transcript')
                if transcript_segments is None and raw_transcript:
                    transcript_segments = estimate_segment_times_by_words(raw_transcript, info.get('duration',0.0), seg_word_length=seg_length, overlap=overlap)
                st.session_state.transcript_segments = transcript_segments
                st.session_state.results = None
                st.success("Metadata and transcript processed.")

    # ---------------- Main Area ----------------
    if st.session_state.video_info:
        info = st.session_state.video_info
        dur = info.get('duration',0.0)
        col1, col2 = st.columns([3,1])
        with col1:
            st.subheader("Video metadata")
            st.write(f"**Title:** {info.get('title','')}")
            st.write(f"**Duration:** {dur:.1f} s")
            if info.get('description'):
                with st.expander("Description (preview)"):
                    st.write(info.get('description')[:1000])
        with col2:
            if st.button("Clear"):
                st.session_state.video_info = None
                st.session_state.transcript_segments = None
                st.session_state.results = None
                st.experimental_rerun()

        st.markdown("---")
        st.subheader("Transcript")
        segs = st.session_state.transcript_segments
        if segs:
            raw = join_segments_to_string(segs)
            edited = st.text_area("Editable transcript:", value=raw, height=240)
            if edited != raw:
                st.session_state.transcript_segments = estimate_segment_times_by_words(edited, dur, seg_word_length=seg_length, overlap=overlap)
                segs = st.session_state.transcript_segments
        else:
            st.info("No transcript available. Use sidebar to fetch/transcribe.")

        if st.button("Analyze"):
            if not segs:
                st.error("No transcript available to analyze.")
            else:
                with st.spinner("Analyzing..."):
                    results, overall_score, comps = analyze_and_score(segs, info.get('title',''), info.get('description',''), dur, similarity_threshold, seg_length)
                    st.session_state.results = {'rows': results, 'overall_score': overall_score, 'components': comps}
                    st.success("Analysis complete.")

        # ---------------- Results ----------------
        if st.session_state.results:
            res = st.session_state.results
            rows = res['rows']
            overall = res['overall_score']
            comps = res['components']

            # Convert rows to DataFrame early
            df = pd.DataFrame(rows)

            st.markdown("---")
            st.header(f"Overall Video Relevance: **{overall:.1f}%**")
            st.write(f"Components: avg_sim={comps['avg_similarity']:.3f}, coverage={comps['coverage']:.3f}, promo_prop={comps['promo_proportion']:.3f}")

            # ---------------- Heatmap ----------------
            # Prepare dataframe
            df_heat = pd.DataFrame({
                "Start": [r['start'] for r in rows],
                "End": [r['end'] for r in rows],
                "Relevance": [r['relevance_score'] for r in rows],
                "Flag": ["Promotional / Irrelevant" if r['is_promo'] or r['is_offtopic'] else "Relevant" for r in rows]
            })

            # Map relevance to color
            df_heat['Color'] = df_heat['Relevance']

            # Create timeline bars
            fig = px.scatter(
                df_heat,
                x='Start',
                y=[1]*len(df_heat),  # dummy y for horizontal alignment
                size=[r['end']-r['start'] for r in rows],  # width = duration
                color='Color',
                color_continuous_scale='RdYlGn',
                hover_data=['Start', 'End', 'Relevance', 'Flag'],
                labels={'x':'Time (s)'}
            )

            fig.update_yaxes(visible=False)
            fig.update_layout(
                height=200,
                showlegend=False,
                xaxis_title='Time (s)',
            )

            st.plotly_chart(fig, use_container_width=True)


            # ---------------- Segment-level table ----------------
            st.markdown("---")
            st.subheader("Segment-level Details")
            display_df = pd.DataFrame({
                'Segment ID': df['segment_id'],
                'Start (s)': df['start'].map(lambda x: round(x,1)),
                'End (s)': df['end'].map(lambda x: round(x,1)),
                'Relevance Score': df['relevance_score'].map(lambda x: round(x,2)),
                'Flag': ['Promotional / Irrelevant' if r else 'Relevant' for r in (df['is_promo'] | df['is_offtopic'])]
            })
            st.dataframe(display_df, use_container_width=True)

            # ---------------- Export ----------------
            st.markdown("---")
            st.subheader("Export Results")
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", data=csv, file_name="video_relevance_results.csv", mime="text/csv")
    else:
        st.markdown("""
        ### Welcome
        Use the sidebar to paste a YouTube URL or upload a video file. For uploaded videos, Whisper (offline) will be used if available. Edit transcripts if needed, then press Analyze.
        """)

if __name__ == "__main__":
    main()


