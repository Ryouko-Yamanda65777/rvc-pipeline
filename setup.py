# setup.py
from setuptools import setup, find_packages

setup(
    name="rvc-pipeline",
    version="0.1.0",
    description="A voice conversion pipeline",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ryouko-Yamanda65777",
    url="https://github.com/Ryouko-Yamanda65777/rvc-pipeline",
    packages=find_packages(),
    install_requires=["torch", "av", "ffmpeg-python>=0.2.0", "faiss_cpu==1.7.3", "praat-parselmouth==0.4.2", "pyworld==0.3.4", "resampy==0.4.2", "pydub==0.25.1", "einops", "local_attention", "torchcrepe==0.0.20", "torchfcpe", "yt_dlp" "audio-separator[gpu]",  # add other dependencies here
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
