# RVC pipeline



# run_inference.py
from lib.infer import infer_audio

# Define the parameters for inference
model_name = "your_model_name"  # Replace with the name of your model
audio_path = "path/to/your/audio.wav"  # Replace with the path to your input audio file

# Optional parameters (these are just examples; adjust as needed)
params = {
    "f0_change": 2,
    "f0_method": "rmvpe+",
    "min_pitch": "50",
    "max_pitch": "1100",
    "crepe_hop_length": 128,
    "index_rate": 0.75,
    "filter_radius": 3,
    "rms_mix_rate": 0.25,
    "protect": 0.33,
    "split_infer": True,
    "min_silence": 500,
    "silence_threshold": -50,
    "seek_step": 1,
    "keep_silence": 100,
    "do_formant": False,
    "quefrency": 0,
    "timbre": 1,
    "f0_autotune": False,
    "audio_format": "wav",
    "resample_sr": 0,
}

# Run the inference
output_path = infer_audio(model_name, audio_path, **params)

if output_path:
    print(f"Inference completed successfully. Output saved to: {output_path}")
else:
    print("Inference failed.")
