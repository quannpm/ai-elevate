# Step 1: Setup packages
# pip install transformers torch IPython soundfile
# pip install --upgrade transformers  # Ensure you have the latest version
# pip install TTS (optional, only support python 3.11)

# Step 2: Import các thư viện cần thiết
from transformers import VitsModel, AutoTokenizer
import torch
from IPython.display import Audio
import soundfile as sf
# from TTS.api import TTS

# Step 3: Tải mô hình TTS và tokenizer từ Hugging Face
model = VitsModel.from_pretrained("facebook/mms-tts-vie")  # Mô hình TTS tiếng Việt
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")

# Step 4: Chuẩn bị văn bản đầu vào
text = "Xin chào Quân, đây là một ví dụ chuyển văn bản thành giọng nói bằng mô hình Hugging Face."

# Step 5: Tokenize văn bản đầu vào
inputs = tokenizer(text, return_tensors="pt")

# Step 6: Sinh waveform từ mô hình (inference)
with torch.no_grad():
    output = model(**inputs).waveform

# Step 7: Phát audio trong Jupyter Notebook
Audio(output.numpy(), rate=model.config.sampling_rate)

# Optional: Lưu audio ra file WAV
# Đảm bảo dữ liệu là float32 và 1 chiều
audio_data = output.squeeze().cpu().numpy().astype("float32")
sf.write('output.wav', audio_data, model.config.sampling_rate)

# Sử dụng TTS để lưu văn bản thành file âm thanh
# tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_model")
# tts.tts_to_file(text="Xin chào!", speaker="female", file_path="output.wav")