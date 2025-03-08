import ffmpeg
import whisper
import streamlit as st
from transformers import pipeline
import langid
import os
from deep_translator import GoogleTranslator

# Load models
model = whisper.load_model("large")  # Large model for better multilingual support
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)  # CPU (-1) or GPU (0)

def detect_languages(text):
    """Detects dominant language(s) in a given text."""
    if not text.strip():
        return "unknown", "Detected Language: UNKNOWN"
    
    lang, confidence = langid.classify(text)
    return lang, f"Detected Language: {lang.upper()} (Confidence: {confidence:.2f})"

def summarize_text(text, source_lang):
    """Summarizes a long text and translates to English if needed."""
    if not text.strip():
        return "Summary not available"

    try:
        # BART model has a max token limit of 1024
        max_input_tokens = 1024
        chunks = [text[i:i+max_input_tokens] for i in range(0, len(text), max_input_tokens)]

        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
            summaries.append(summary[0]["summary_text"])

        full_summary = " ".join(summaries)

        # Translate summary to English if needed
        translated_summary = GoogleTranslator(source=source_lang, target="en").translate(full_summary) if source_lang != "en" else full_summary
        
        return translated_summary
    except Exception as e:
        st.error(f"Summarization failed: {e}")
        return "Summary not available"

def convert_to_wav(input_audio):
    """Converts an audio file to WAV format safely."""
    output_audio = "converted_audio.wav"
    try:
        ffmpeg.input(input_audio).output(output_audio, format='wav').run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        return output_audio
    except Exception as e:
        st.error(f"Audio conversion failed: {e}")
        return None

# Streamlit UI
st.title("🎙️ AI Multilingual Transcriber")
st.write("Upload an audio file")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
if uploaded_file is not None:
    file_path = f"temp_audio.{uploaded_file.name.split('.')[-1]}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("Audio file uploaded successfully")

    # Convert to WAV
    wav_file = convert_to_wav(file_path)
    if wav_file is None:
        st.error("Failed to process audio file.")
    else:
        # Transcribe
        try:
            result = model.transcribe(wav_file)
            transcription = result["text"]
            detected_language, language_info = detect_languages(transcription)
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            transcription = ""
            detected_language = "unknown"

        if transcription:
            # Generate summary
            summary = summarize_text(transcription, detected_language)

            # Save transcription & summary to files
            with open("transcription.txt", "w", encoding="utf-8-sig") as f:
                f.write(transcription)
            
            with open("summary.txt", "w", encoding="utf-8-sig") as f:
                f.write(summary)

            # Display Results
            st.subheader("📝 Transcription")
            st.write(transcription)

            st.subheader("🌎 Detected Language")
            st.write(language_info)

            st.subheader("🔑 Key Takeaways (English)")
            st.write(summary)

        # Cleanup temporary files
        os.remove(file_path)
        os.remove(wav_file)
