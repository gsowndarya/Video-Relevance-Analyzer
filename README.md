# Video-Relevance-Analyzer

A Streamlit application that evaluates how closely a video aligns with its own title and description. It pulls transcripts (YouTube API, captions, or Whisper offline), segments them, embeds them, computes semantic similarity, detects promotional content, and visualizes relevance over time using a clean timeline heatmap.

🚀 Features
1. Multiple Input Methods
Paste a YouTube URL and automatically fetch metadata + transcripts.
Upload a local video file and auto‑transcribe using Whisper (if installed).

2. Transcript Handling
Uses YouTubeTranscriptApi when available.
Falls back to VTT/SRT captions from yt-dlp if needed.
For plain text transcripts with no timestamps, generates word‑based timestamp estimation.
Lets users edit the transcript manually before analysis.

3. Semantic Relevance Scoring
Embeds segments using SentenceTransformer (MiniLM‑L6‑v2).
Computes cosine similarity between each segment and the combined title + description.

Generates:
Per‑segment relevance score
Coverage proportion above a similarity threshold
Average similarity across the timeline
Overall composite video score

4. Promotional / Off-topic Detection
Keyword heuristics plus a lightweight Logistic Regression classifier trained on synthetic promo vs. non‑promo text.
Flags segments as Promotional / Irrelevant.

5. Visualization Dashboard

A clean, simple heatmap-like timeline scatterplot, where:
X‑axis = segment start time
Point size = segment duration
Color = relevance (0–100)
Segment-level table with flags + scores.

6. Optional LLM Explanation
If users enter an OpenAI API key, the app generates a concise, polished explanation summarizing:
The overall score
Why it’s high or low
Flagged examples

7. Exporting Results
Users can download the full segment-level data as CSV.
