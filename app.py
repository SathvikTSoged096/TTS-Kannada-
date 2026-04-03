# =========================================================
# 1) INSTALL DEPENDENCIES
# =========================================================
!pip -q install git+https://github.com/huggingface/parler-tts.git
!pip -q install soundfile transformers accelerate sentencepiece huggingface_hub

# =========================================================
# 2) IMPORTS
# =========================================================
import os
import torch
import soundfile as sf
from IPython.display import Audio, display
from google.colab import files
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from huggingface_hub import notebook_login, hf_hub_download

# =========================================================
# 3) HUGGING FACE LOGIN
# =========================================================
print("🔐 Please login with your Hugging Face READ token")
notebook_login()

# =========================================================
# 4) VERIFY MODEL ACCESS
# =========================================================
MODEL_NAME = "ai4bharat/indic-parler-tts"

try:
    hf_hub_download(
        repo_id=MODEL_NAME,
        filename="config.json"
    )
    print("✅ Model access verified")
except Exception as e:
    print("❌ ACCESS ERROR")
    print("Open this page:")
    print("https://huggingface.co/ai4bharat/indic-parler-tts")
    print("Then click 👉 Agree and access repository")
    raise e

# =========================================================
# 5) DEVICE
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Using device:", device)

# =========================================================
# 6) LOAD MODEL
# =========================================================
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
# 7) SAFE GENERATION FUNCTION
# =========================================================
def generate_kannada_tts(prompt_text, output_file="/content/kannada_output.wav"):
    prompt_text = str(prompt_text).strip()

    if not prompt_text:
        raise ValueError("❌ Kannada input cannot be empty")

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

    sf.write(
        output_file,
        audio,
        model.config.sampling_rate
    )

    print(f"✅ Audio saved → {output_file}")
    display(Audio(output_file))

    return output_file

# =========================================================
# 8) USER INPUT OUTSIDE FUNCTION
# =========================================================
user_text = input("Enter Kannada text: ")

# Example:
# ನಮಸ್ಕಾರ, ನನ್ನ ಹೆಸರು ಅಥ್ಮಿಕ

try:
    output_path = generate_kannada_tts(user_text)
except Exception as e:
    print("❌ Error:", e)

# =========================================================
# 9) DOWNLOAD
# =========================================================
if os.path.exists("/content/kannada_output.wav"):
    files.download("/content/kannada_output.wav")