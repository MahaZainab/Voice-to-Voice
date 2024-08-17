# -*- coding: utf-8 -*-



import os
import gradio as gr
import whisper
from gtts import gTTS
from groq import Groq

# Set up Groq API client
client = Groq(
    api_key="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",
)

# Load Whisper model
model = whisper.load_model("base")

def chatbot(audio):
    # Transcribe the audio input using Whisper
    transcription = model.transcribe(audio)
    user_input = transcription["text"]

    # Generate a response using Llama 8B via Groq API
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": user_input,
            }
        ],
        model="llama3-8b-8192",
    )
    response_text = chat_completion.choices[0].message.content

    # Convert the response text to speech using gTTS
    tts = gTTS(text=response_text, lang='en')
    tts.save("response.mp3")

    return response_text, "response.mp3"

# Set up Gradio interface
iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Audio(type="filepath"),  # Corrected input parameters
    outputs=[gr.Textbox(), gr.Audio()],
    live=True
)

iface.launch()


