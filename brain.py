import os
import base64
from groq import Groq
import gradio as gr
from gtts import gTTS
import speech_recognition as sr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def encode_image(image_path):
    if image_path is None:
        return ""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_medical_case(user_input, image_path=None):
    SYSTEM_PROMPT = "You are a certified medical assistant. Analyze symptoms and medical images with clinical accuracy."
    content = user_input
    if image_path:
        # Append a truncated base64 string to the prompt for demonstration
        base64_img = encode_image(image_path)
        content += f"\n\n[Attached medical image (base64, first 100 chars)]: {base64_img[:100]}..."

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def speech_to_text(audio_file):
    if audio_file is None:
        return ""
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        return r.recognize_google(audio)
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def text_to_speech(text, filename="response.mp3"):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        return filename
    except Exception as e:
        return None

def process_input(text, image, audio):
    # If audio is provided, transcribe it; otherwise use text
    if audio:
        text = speech_to_text(audio)
        if text.startswith("Error"):
            return text, None
    if not text:
        return "Please provide a description or audio.", None
    diagnosis = analyze_medical_case(text, image)
    audio_path = text_to_speech(diagnosis)
    return diagnosis, audio_path

def create_chat_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# 🩺 Medical AI Assistant with Vision and Voice")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath", label="Upload Medical Image")
                audio_input = gr.Audio(
                    label="Voice Input",
                    type="filepath",
                    sources=["microphone", "upload"],
                    interactive=True
                )
                text_input = gr.Textbox(label="Text Input")
                submit_btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                text_output = gr.Textbox(label="Diagnosis", interactive=False)
                audio_output = gr.Audio(
                    autoplay=True,
                    label="Voice Response",
                    visible=True
                )
        submit_btn.click(
            fn=process_input,
            inputs=[text_input, image_input, audio_input],
            outputs=[text_output, audio_output]
        )
    return interface

if __name__ == "__main__":
    interface = create_chat_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)
