# Video Relevance Analyzer 🎥

A Streamlit dashboard that analyzes YouTube videos by extracting transcripts, measuring semantic similarity, detecting promotional content, and computing relevance scores.

## Features

- Extract metadata and transcripts from YouTube URLs

- Upload local video files and transcribe them using Whisper (if installed)

- Segment transcripts with adjustable word-length and overlap

- Compute semantic similarity between transcript segments and the video title/description

- Detect promotional or irrelevant segments using a lightweight classifier and keyword heuristics

- Generate a timeline heatmap of segment relevance

- Provide detailed segment-level scores and flags

- Export results to CSV

- Optional OpenAI API key input for improved explanations

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage

1. **Enter Video URL**: Paste a YouTube video URL in the sidebar
2. **Adjust Settings** (optional):
   - Segment Length: Number of words per segment (default: 200)
   - Overlap: Number of overlapping words between segments (default: 50)
3. **Click "Analyze Video"**: The app will extract the transcript and analyze it
4. **View Results**:
   - Summary statistics (total segments, average relevance, etc.)
   - Interactive heatmap showing relevance scores
   - Detailed segment-by-segment analysis with explanations
5. **Export**: Download results as CSV for further analysis

## Requirements

- Python 3.8+

- FFmpeg installed (for Whisper + audio extraction)

- Internet connection (for YouTube + embedding model download)

- Whisper offline transcription is optional.

## How It Works

1. **Input Methods**: Paste a YouTube URL or Upload a local video file
1. **Transcript Extraction**: YouTube transcript via API, YouTube subtitle files (.vtt/.srt), Whisper offline transcription (for uploaded files)
2. **Segmentation**: Splits the transcript into overlapping segments
3. **Similarity Calculation**: Computes cosine similarity between title/description and each segment
4. **Promo Detection**: Checks for promotional keywords (subscribe, like, sponsor, etc.)
5. **Scoring**: Calculates relevance score (0-100) based on similarity and promo detection
6. **Visualization**: Creates interactive heatmaps using Plotly

## Notes

- The video must have captions/subtitles enabled for transcript extraction
- First run will download the sentence transformer model (~80MB)
- Processing time depends on video length and transcript size

## Troubleshooting

- **No transcript found**: Ensure the video has captions enabled
- **Slow processing**: Large videos may take time; consider adjusting segment length
- **Model download**: First run requires internet to download the transformer model


