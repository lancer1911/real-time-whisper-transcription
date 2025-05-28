# Real-Time Speech Transcription

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A real-time speech transcription system built with OpenAI Whisper supporting continuous audio recording and multiple languages.

## Features

- Real-time transcription with continuous audio recording
- Multi-language support with auto-detection
- Segmented processing with smart overlap handling
- Interactive setup or command-line configuration
- Multiple recording methods: sounddevice, pyaudio, speech_recognition
- Pause/Resume functionality with Ctrl+C menu
- Audio segment saving for debugging and quality analysis
- Energy threshold configuration for speech detection

## Installation

### Prerequisites
- Python 3.8 or higher
- Conda (recommended) or pip

### Create Environment

**Using Conda (Recommended):**
```bash
# Create and activate conda environment
conda create -n whisper-transcription python=3.9
conda activate whisper-transcription

# Clone the repository
git clone https://github.com/lancer1911/real-time-whisper-transcription.git
cd real-time-whisper-transcription

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional packages
pip install sounddevice faster-whisper
```

**Using pip:**
```bash
# Clone the repository
git clone https://github.com/lancer1911/real-time-whisper-transcription.git
cd real-time-whisper-transcription

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional packages
pip install sounddevice faster-whisper
```

**Note:** The requirements.txt includes PyTorch with CUDA 11.6 support. For CPU-only installation, modify the torch installation:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### System Dependencies

**macOS:**
```bash
# Install portaudio for pyaudio
brew install portaudio
```

**Ubuntu/Debian:**
```bash
# Install portaudio and ffmpeg
sudo apt update
sudo apt install portaudio19-dev ffmpeg
```

**Windows:**
- Download and install FFmpeg from https://ffmpeg.org/download.html
- Add FFmpeg to PATH environment variable

## Quick Start

### Activate Environment
```bash
# If using conda
conda activate whisper-transcription
```

### Interactive Mode
```bash
python transcriber.py
```

### Command Line Mode
```bash
# Basic usage
python transcriber.py --language en --model small --save_audio

# Chinese transcription
python transcriber.py --language zh --model small --save_audio
```

## Audio Diagnostics

Test your microphone setup and find optimal settings:

```bash
python audio_diagnostic.py
```

The diagnostic tool will:
- List all available microphone devices
- Test different energy thresholds (4000, 1000, 300, 100, 50)
- Analyze audio volume levels
- Provide recommended energy threshold values

Sample output:
```
1. Available microphone devices:
   0: Built-in Microphone
   1: USB Audio Device

2. Testing default microphone...
   Testing energy threshold: 300
   Volume - Max: 0.156, Average: 0.023
   Threshold 300 working properly!

Recommended energy threshold: 300
```

Use the recommended settings:
```bash
python transcriber.py --energy_threshold 300 --microphone_index 0 --save_audio
```

## Configuration

### Basic Parameters
- `--language`: Target language (`en`, `zh`, `ja`, `ko`, `es`, `fr`, `de`, `auto`)
- `--model`: Whisper model size (`tiny`, `base`, `small`, `medium`, `large`)
- `--microphone_index`: Microphone device index
- `--energy_threshold`: Speech detection threshold (default: 300)

### Audio Settings
- `--recording_method`: Recording method (`sounddevice`, `pyaudio`, `speech_recognition`)
- `--segment_duration`: Segment length in seconds (default: 8.0)
- `--overlap_duration`: Overlap length in seconds (default: 2.0)

### Output Settings
- `--output_file`: Transcription output file (default: `transcription.txt`)
- `--save_audio`: Save audio segments (recommended for debugging and analysis)
- `--debug`: Enable debug mode

## Usage Examples

```bash
# List available microphones
python transcriber.py --list_mics

# Chinese transcription with audio saving
python transcriber.py --language zh --save_audio

# Custom energy threshold for sensitive microphone
python transcriber.py --energy_threshold 100 --save_audio

# Debug mode with custom settings
python transcriber.py --debug --segment_duration 10.0 --save_audio
```

## Interactive Controls

During transcription:
- **Ctrl+C**: Pause and access options menu
  - Option 1: Resume recording
  - Option 2: Stop and exit

## Troubleshooting

### Audio Issues
1. **Run diagnostics first**: `python audio_diagnostic.py`
2. **Low audio detection**: Use lower energy threshold (50-100)
3. **No audio detected**: Check microphone permissions
4. **Try different recording method**: `--recording_method pyaudio`
5. **Enable audio saving**: Use `--save_audio` to analyze recorded segments

### Model Issues
- **Memory errors**: Use smaller model (`--model tiny`)
- **Slow performance**: Use base or small model

### Permission Issues
- **macOS**: Grant microphone permission in System Preferences
- **Linux**: Add user to audio group: `sudo usermod -a -G audio $USER`

## Known Issues / Experimental Features

The following features are implemented but not fully tested:

### GPU Acceleration
- `--device cuda` (NVIDIA GPUs with CUDA 11.6+)
- `--device mps` (Apple Silicon)
- PyTorch with CUDA 11.6 is included in requirements.txt
- May require additional CUDA runtime setup

### Faster-Whisper
- `--use_faster_whisper` flag available
- Requires: `pip install faster-whisper`
- Performance improvements not verified

### LLM Post-processing (Ollama)
- `--enable_llm` flag available
- Requires Ollama installation and setup
- Text correction functionality not tested

## Requirements

### Core Dependencies (in requirements.txt)
- Python 3.8+
- PyTorch (with CUDA 11.6 support)
- OpenAI Whisper
- NumPy
- PyAudio
- SpeechRecognition

### Optional Dependencies
- sounddevice (recommended for better audio recording)
- faster-whisper (experimental performance improvements)

### System Requirements
- 4GB RAM minimum
- Microphone access
- Internet connection for initial model download
- CUDA-capable GPU (optional, for acceleration)

## License

MIT License - see LICENSE file for details.