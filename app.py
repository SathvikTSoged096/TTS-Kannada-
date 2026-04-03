
import gradio as gr
import torch
import soundfile as sf
import tempfile
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import os
from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))

MODEL_NAME = "ai4bharat/indic-parler-tts"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 Using device:", device)
print("⏳ Loading Kannada TTS model...")

model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_NAME
).to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)

print("✅ Model loaded successfully")


# =========================================================
# TTS FUNCTION
# =========================================================
def generate_kannada_tts(prompt_text):
    prompt_text = str(prompt_text).strip()

    if not prompt_text:
        return None

    description = (
        "A calm Kannada male speaker with natural pronunciation, "
        "clear studio quality audio, smooth narration, "
        "and no background noise."
    )

    description_inputs = description_tokenizer(
        description,
        return_tensors="pt"
    ).to(device)

    prompt_inputs = tokenizer(
        prompt_text,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generation = model.generate(
            input_ids=description_inputs.input_ids,
            prompt_input_ids=prompt_inputs.input_ids
        )

    audio = generation.cpu().numpy().squeeze()

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_wav.name, audio, model.config.sampling_rate)

    return temp_wav.name



demo = gr.Interface(
    fn=generate_kannada_tts,
    inputs=gr.Textbox(
        label="Enter Kannada Text",
        placeholder="ನಮಸ್ಕಾರ, ನನ್ನ ಹೆಸರು ಅಥ್ಮಿಕ"
    ),
    outputs=gr.Audio(label="Generated Kannada Speech"),
    title="Kannada Text To Speech using AI4Bharat",
    description="Deep Learning based Kannada TTS model for project presentation"
)

demo.launch()
