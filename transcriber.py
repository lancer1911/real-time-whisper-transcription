#!/usr/bin/env python3

import argparse
import asyncio
import json
import numpy as np
import speech_recognition as sr
import whisper
import torch
import sys
from collections import deque
from datetime import datetime, timedelta
from queue import Queue, Empty
import threading
import time
import wave
import os

try:
    import faster_whisper
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

class LLMPostProcessor:
    """Use LLM for transcription post-processing and error correction"""

    def __init__(self, model_type="local"):
        self.model_type = model_type
        self.conversation_context = deque(maxlen=5)

        if model_type == "local":
            try:
                import ollama
                self.client = ollama.Client()
                self.model_name = "llama3.2:3b" # Corrected model name if it was llama3.2
            except ImportError:
                print("Ollama not available, using simple corrections")
                self.client = None
            except Exception as e:
                print(f"Error initializing Ollama client: {e}. Using simple corrections.")
                self.client = None


    def correct_text(self, raw_text, context=""):
        """Correct and format text"""
        if not raw_text.strip():
            return raw_text

        # Simple heuristic corrections
        corrected = self._simple_corrections(raw_text)

        # If LLM is available, perform deep correction
        if self.client:
            corrected = self._llm_correction(corrected, context)

        self.conversation_context.append(corrected)
        return corrected

    def _simple_corrections(self, text):
        """Simple text correction rules"""
        corrections = {
            " i ": " I ",
            " im ": " I'm ",
            " its ": " it's ",
            " youre ": " you're ",
            " dont ": " don't ",
            " wont ": " won't ",
            " cant ": " can't ",
        }

        result = text
        for wrong, correct in corrections.items():
            result = result.replace(wrong, correct)

        if result and result[-1] not in '.!?':
            result += '.'

        return result.strip()

    def _llm_correction(self, text, context):
        """Use LLM for advanced correction"""
        if not self.client:
            return text

        try:
            prompt = f"""Please correct the following speech transcription text errors, maintain the original meaning, and add appropriate punctuation. Only return the corrected text without explanation.

Context: {context}
Original: {text}
Corrected:"""

            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.1, 'num_predict': 100}
            )
            return response['response'].strip()
        except Exception as e:
            print(f"LLM correction failed: {e}")
            return text

class ContinuousAudioRecorder:
    """连续音频录制器 - 解决录音不连续问题"""

    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024, method="sounddevice",
                 microphone_index=None, energy_threshold=300):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.method = method
        self.microphone_index = microphone_index
        self.energy_threshold = energy_threshold
        self.running = False
        self.audio_queue = Queue()
        self.total_frames_recorded = 0
        self.device_name = "Unknown Device"  # 添加设备名称属性

        print(f"🎙️  Initializing audio recorder - Method: {method}")

        # 根据可用性选择录音方法
        if method == "sounddevice" and SOUNDDEVICE_AVAILABLE:
            self._init_sounddevice()
        elif method == "pyaudio" and PYAUDIO_AVAILABLE:
            self._init_pyaudio()
        elif method == "speech_recognition":
            self._init_speech_recognition()
        else:
            # 自动回退到可用的方法
            if SOUNDDEVICE_AVAILABLE:
                print("🔄 Falling back to sounddevice")
                self.method = "sounddevice"
                self._init_sounddevice()
            elif PYAUDIO_AVAILABLE:
                print("🔄 Falling back to pyaudio")
                self.method = "pyaudio"
                self._init_pyaudio()
            elif sr.Microphone is not None: # Check if sr was imported successfully
                print("🔄 Falling back to speech_recognition")
                self.method = "speech_recognition"
                self._init_speech_recognition()
            else:
                raise RuntimeError("No audio recording libraries (sounddevice, pyaudio, SpeechRecognition) are available.")


    def _init_sounddevice(self):
        """Initialize sounddevice (推荐方法)"""
        print("✅ Using sounddevice for continuous recording")

        # 列出可用设备
        devices = sd.query_devices()

        if self.microphone_index is not None:
            if 0 <= self.microphone_index < len(devices):
                device_info = devices[self.microphone_index]
                self.device_name = device_info['name']  # 保存设备名称
                print(f"   Using device {self.microphone_index}: {device_info['name']}")
                self.device_index = self.microphone_index
            else:
                print(f"❌ Device index {self.microphone_index} out of range, using default")
                self.device_index = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
                if self.device_index != -1 and 0 <= self.device_index < len(devices):
                    self.device_name = devices[self.device_index]['name']
                else:
                    self.device_name = "Default (Unknown)"
        else:
            default_input_device_index = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
            if default_input_device_index == -1 and len(devices) > 0 : # No default, pick first available input if any
                 for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        default_input_device_index = i
                        self.device_name = device['name']
                        print(f"   No default input device found. Using first available: {device['name']}")
                        break
            if default_input_device_index != -1:
                 if 0 <= default_input_device_index < len(devices):
                     self.device_name = devices[default_input_device_index]['name']
                     print(f"   Using default input device: {self.device_name}")
                 else:
                     self.device_name = "Default (Unknown)"
                 self.device_index = default_input_device_index
            else:
                 print("❌ No input devices found for sounddevice.")
                 self.device_index = None # Should trigger error later if not handled
                 self.device_name = "No Input Device"
        self.stream = None

    def _init_pyaudio(self):
        """Initialize pyaudio"""
        print("✅ Using pyaudio for continuous recording")
        self.pyaudio_instance = pyaudio.PyAudio()

        # 显示可用设备
        print("   Available audio devices:")
        default_device_index = -1
        try:
            default_device_info = self.pyaudio_instance.get_default_input_device_info()
            default_device_index = default_device_info['index']
        except IOError:
            print("   Warning: No default PyAudio input device found.")


        for i in range(self.pyaudio_instance.get_device_count()):
            info = self.pyaudio_instance.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                marker = ""
                if self.microphone_index == i : marker += " [SELECTED]"
                elif default_device_index == i and self.microphone_index is None: marker += " [DEFAULT]"
                print(f"     {i}: {info['name']}{marker}")

        if self.microphone_index is not None:
            try:
                info = self.pyaudio_instance.get_device_info_by_index(self.microphone_index)
                if info['maxInputChannels'] > 0:
                    self.device_name = info['name']  # 保存设备名称
                    print(f"   Using device {self.microphone_index}: {info['name']}")
                    self.device_index = self.microphone_index
                else:
                    print(f"❌ Device {self.microphone_index} has no input channels, using default if available.")
                    self.device_index = default_device_index if default_device_index != -1 else None
                    if default_device_index != -1:
                        try:
                            default_info = self.pyaudio_instance.get_device_info_by_index(default_device_index)
                            self.device_name = default_info['name']
                        except:
                            self.device_name = "Default (Unknown)"
                    else:
                        self.device_name = "No Input Device"
            except Exception as e:
                print(f"❌ Error with device {self.microphone_index}: {e}, using default if available.")
                self.device_index = default_device_index if default_device_index != -1 else None
                if default_device_index != -1:
                    try:
                        default_info = self.pyaudio_instance.get_device_info_by_index(default_device_index)
                        self.device_name = default_info['name']
                    except:
                        self.device_name = "Default (Unknown)"
                else:
                    self.device_name = "No Input Device"
        else:
            if default_device_index != -1:
                try:
                    default_info = self.pyaudio_instance.get_device_info_by_index(default_device_index)
                    self.device_name = default_info['name']
                    print(f"   Using default input device: {self.device_name}")
                except:
                    self.device_name = "Default (Unknown)"
                    print(f"   Using default input device: {self.device_name}")
                self.device_index = default_device_index
            else:
                print("   No default input device. Please specify a microphone_index.")
                self.device_index = None # This will likely cause an error later
                self.device_name = "No Input Device"

        self.stream = None

    def _init_speech_recognition(self):
        """Initialize speech_recognition with improvements"""
        print("✅ Using speech_recognition with phrase_time_limit")
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.energy_threshold
        self.recorder.dynamic_energy_threshold = True  # 启用动态阈值
        self.recorder.pause_threshold = 0.3
        self.recorder.non_speaking_duration = 0.2

        mic_names = sr.Microphone.list_microphone_names()
        if not mic_names:
            print("❌ No microphones found by SpeechRecognition library.")
            self.microphone = None
            self.device_name = "No Microphone"
            return

        # 设置麦克风设备
        if self.microphone_index is not None:
            print(f"🎤 Using specified microphone device {self.microphone_index}")
            if 0 <= self.microphone_index < len(mic_names):
                self.device_name = mic_names[self.microphone_index]  # 保存设备名称
                print(f"   Device name: {self.device_name}")
                self.microphone = sr.Microphone(sample_rate=self.sample_rate, device_index=self.microphone_index)
            else:
                print(f"❌ Microphone index {self.microphone_index} out of range ({len(mic_names)} available), using default.")
                self.microphone = sr.Microphone(sample_rate=self.sample_rate)
                self.device_name = "Default Microphone"
        else:
            print("🎤 Using default microphone device")
            self.microphone = sr.Microphone(sample_rate=self.sample_rate)
            self.device_name = "Default Microphone"

        if self.microphone:
            print(f"   Energy threshold: {self.recorder.energy_threshold}")
            print(f"   Dynamic threshold: {self.recorder.dynamic_energy_threshold}")

    def start_recording(self):
        """开始录音"""
        self.running = True
        self.total_frames_recorded = 0

        if self.method == "sounddevice":
            if self.device_index is None: # From _init_sounddevice
                 print("❌ Sounddevice cannot start: No valid device index.")
                 return False
            return self._start_sounddevice()
        elif self.method == "pyaudio":
            if self.device_index is None: # From _init_pyaudio
                 print("❌ PyAudio cannot start: No valid device index.")
                 return False
            return self._start_pyaudio()
        else: # speech_recognition
            if not hasattr(self, 'microphone') or self.microphone is None:
                 print("❌ SpeechRecognition cannot start: Microphone not initialized.")
                 return False
            return self._start_speech_recognition()

    def _start_sounddevice(self):
        """Start sounddevice recording"""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"⚠️  Audio status (sounddevice): {status}")
            if self.running:
                # indata shape: (frames, channels)
                audio_data = indata[:, 0] if self.channels == 1 and indata.ndim > 1 else indata
                self.audio_queue.put(audio_data.copy().flatten())
                self.total_frames_recorded += frames

        try:
            # 使用流式录音
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback,
                blocksize=self.chunk_size,
                dtype=np.float32,
                device=self.device_index  # 使用指定设备
            )
            self.stream.start()
            print("🎙️  Sounddevice recording started")
            return True
        except Exception as e:
            print(f"❌ Sounddevice recording failed: {e}")
            return False

    def _start_pyaudio(self):
        """Start pyaudio recording"""
        def audio_callback(in_data, frame_count, time_info, status):
            if self.running:
                audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
                self.audio_queue.put(audio_data)
                self.total_frames_recorded += frame_count
            return (None, pyaudio.paContinue)

        try:
            stream_kwargs = {
                'format': pyaudio.paInt16,
                'channels': self.channels,
                'rate': self.sample_rate,
                'input': True,
                'frames_per_buffer': self.chunk_size,
                'stream_callback': audio_callback
            }

            # 添加设备索引（如果指定）
            if self.device_index is not None:
                stream_kwargs['input_device_index'] = self.device_index
            else: # Should have been caught by start_recording
                print("❌ PyAudio error: device_index is None before opening stream.")
                return False


            self.stream = self.pyaudio_instance.open(**stream_kwargs)
            self.stream.start_stream()
            print("🎙️  PyAudio recording started")
            return True
        except Exception as e:
            print(f"❌ PyAudio recording failed: {e}")
            return False

    def _start_speech_recognition(self):
        """Start speech_recognition with fixed phrase_time_limit"""
        def audio_callback(recognizer, audio):
            if self.running:
                data = np.frombuffer(
                    audio.get_raw_data(),
                    dtype=np.int16
                ).astype(np.float32) / 32768.0
                self.audio_queue.put(data)
                self.total_frames_recorded += len(data)

        try:
            # 关键修复：使用固定的 phrase_time_limit
            self.stop_listening = self.recorder.listen_in_background(
                self.microphone,
                audio_callback,
                phrase_time_limit=0.8  # 每0.8秒强制回调一次
            )
            print("🎙️  Speech recognition recording started (with phrase_time_limit=0.8s)")
            return True
        except Exception as e:
            print(f"❌ Speech recognition recording failed: {e}")
            return False

    def pause_recording(self):
        """暂停录音（不完全停止流）"""
        if self.running:
            self.running = False
            print("⏸️  Recording paused (streams remain active).")
        else:
            print("⏸️  Recording already paused.")

    def resume_recording(self):
        """恢复录音"""
        if not self.running:
            self.running = True
            
            # 检查流是否仍然活动，如果不活动则重新启动
            stream_active = False
            if self.method == "sounddevice" and self.stream:
                stream_active = self.stream.active
            elif self.method == "pyaudio" and self.stream:
                stream_active = self.stream.is_active()
            elif self.method == "speech_recognition" and hasattr(self, 'stop_listening'):
                stream_active = True  # speech_recognition doesn't have a simple way to check
            
            if not stream_active:
                print("🔄 Restarting audio stream...")
                return self.start_recording()
            else:
                print("▶️  Recording resumed (stream was still active).")
                return True
        else:
            print("▶️  Recording already active.")
            return True

    def stop_recording(self):
        """停止录音"""
        self.running = False
        time.sleep(0.2) # Allow callback to finish current chunk

        if self.method == "sounddevice" and self.stream:
            if self.stream.active:
                self.stream.stop()
            self.stream.close()
            print("   Sounddevice stream stopped and closed.")
        elif self.method == "pyaudio" and self.stream:
            if self.stream.is_active(): # Check before stopping
                self.stream.stop_stream()
            self.stream.close()
            self.pyaudio_instance.terminate()
            print("   PyAudio stream stopped and closed, instance terminated.")
        elif self.method == "speech_recognition" and hasattr(self, 'stop_listening') and self.stop_listening:
            self.stop_listening(wait_for_stop=False)
            print("   SpeechRecognition listening stopped.")

        print(f"🛑 Recording stopped. Total frames: {self.total_frames_recorded}")

    def get_audio(self, timeout=0.1):
        """获取音频数据"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except Empty:
            return None

    def get_queue_size(self):
        """获取队列大小"""
        return self.audio_queue.qsize()

class SegmentedTranscriber:
    """Segmented recording transcriber"""

    def __init__(self, model_name="small", use_faster_whisper=True,
                 enable_llm=False, device="auto", language="en",
                 segment_duration=8.0, overlap_duration=2.0, debug_mode=False): # Added debug_mode

        self.target_language = language
        self.segment_duration = segment_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = 16000
        self.debug_mode = debug_mode # Store debug_mode

        # Device selection
        if device == "auto":
            if FASTER_WHISPER_AVAILABLE and torch.backends.mps.is_available() and torch.backends.mps.is_built(): # Check if MPS is built
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Model loading
        self.model_name = model_name
        self.use_faster_whisper = use_faster_whisper and FASTER_WHISPER_AVAILABLE
        self._load_model()

        # Post-processor
        self.post_processor = LLMPostProcessor() if enable_llm else None

        # Transcription history
        self.transcription_history = deque(maxlen=50)

    def _load_model(self):
        """Load Whisper model"""
        model_loaded_successfully = False
        if self.use_faster_whisper:
            print(f"Loading Faster-Whisper model '{self.model_name}'...")
            try:
                compute_type = "float16" if self.device in ["cuda", "mps"] else ("bfloat16" if self.device == "cpu" and hasattr(torch, 'bfloat16') and torch.cpu.is_bf16_supported() else "float32")
                if self.device == "mps" and compute_type == "float16": # MPS often prefers float32 for stability with some models
                    print("   Note: MPS device with float16 compute_type. If issues arise, consider float32 for Faster-Whisper on MPS.")
                print(f"   Using compute_type: {compute_type}")
                self.model = faster_whisper.WhisperModel(
                    self.model_name,
                    device=self.device,
                    compute_type=compute_type
                )
                model_loaded_successfully = True
            except Exception as e:
                print(f"Faster-Whisper failed: {e}")
                print("Falling back to OpenAI Whisper...")
                self.use_faster_whisper = False # Ensure this is set so the next block runs

        if not self.use_faster_whisper and not model_loaded_successfully: # Check model_loaded_successfully in case faster_whisper was true but failed
            print(f"Loading OpenAI Whisper model '{self.model_name}'...")
            try:
                self.model = whisper.load_model(self.model_name, device=self.device)
                model_loaded_successfully = True
            except Exception as e:
                print(f"OpenAI Whisper model loading failed on {self.device}: {e}")
                if self.device == "mps" or self.device == "cuda": # Fallback from GPU to CPU
                    print(f"Falling back to CPU for OpenAI Whisper...")
                    self.device = "cpu" # Update self.device
                    try:
                        self.model = whisper.load_model(self.model_name, device=self.device)
                        model_loaded_successfully = True
                    except Exception as e_cpu:
                        print(f"OpenAI Whisper model loading failed on CPU as well: {e_cpu}")
                        raise e_cpu # Re-raise if CPU also fails
                else:
                    raise e # Re-raise original error if not on GPU or already on CPU

        if model_loaded_successfully:
            print(f"Model '{self.model_name}' loaded successfully on {self.device} {'(Faster-Whisper)' if self.use_faster_whisper else '(OpenAI Whisper)'}")
        else:
            # This state should ideally not be reached if exceptions are re-raised
            print(f"🚨 CRITICAL: Model '{self.model_name}' could not be loaded on any device.")
            # Consider exiting or raising a more specific error
            raise RuntimeError(f"Failed to load Whisper model '{self.model_name}'.")


    def transcribe_audio(self, audio_data):
        """转录音频数据"""
        if not isinstance(audio_data, np.ndarray):
            print(f"❌ Invalid audio data type: Expected numpy array, got {type(audio_data)}")
            return ""
        if audio_data.size == 0 : # Check if array is empty
            print("❌ Empty audio data array received for transcription.")
            return ""
        if len(audio_data) < self.sample_rate * 0.1:  # Less than 0.1 seconds, likely too short
            if self.debug_mode:
                print(f"Audio data too short ({len(audio_data)/self.sample_rate:.2f}s), skipping transcription.")
            return ""


        try:
            # Detailed audio properties if debug mode is on
            if self.debug_mode:
                 print(f"🔍 SegmentedTranscriber: Transcribing audio. Length={len(audio_data)/self.sample_rate:.2f}s, Min={np.min(audio_data):.4f}, Max={np.max(audio_data):.4f}, Mean={np.mean(audio_data):.4f}, Std={np.std(audio_data):.4f}, Dtype={audio_data.dtype}")

            if np.isnan(audio_data).any() or np.isinf(audio_data).any():
                print("❌ Invalid audio data (NaN/Inf) detected before transcription")
                return ""
            if audio_data.dtype != np.float32: # Ensure float32 for Whisper
                audio_data = audio_data.astype(np.float32)


            text = ""
            if self.use_faster_whisper:
                segments, info = self.model.transcribe(
                    audio_data,
                    language=self.target_language if self.target_language != "auto" else None,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=700, speech_pad_ms=200) # Adjusted VAD params
                )
                text = " ".join([segment.text for segment in segments])
                num_segments = sum(1 for _ in segments) # Correct way to count segments
                print(f"🎵 Faster-Whisper: '{text.strip()}' (Lang: {info.language}, Prob: {info.language_probability:.2f}, Segments: {num_segments})")
            else: # OpenAI Whisper
                transcribe_kwargs = {}
                if self.target_language != "auto":
                    transcribe_kwargs["language"] = self.target_language
                # FP16 is often default or handled internally by whisper.load_model based on device
                # Forcing it might not always be optimal or necessary.
                # if self.device != "cpu":
                #    transcribe_kwargs["fp16"] = True # Can be True or False. None uses default.

                result = self.model.transcribe(audio_data, **transcribe_kwargs)
                text = result["text"]
                detected_lang = result.get('language', 'N/A')
                print(f"🎵 OpenAI Whisper: '{text.strip()}' (Lang: {detected_lang})")

            text = text.strip()

            # LLM post-processing
            if self.post_processor and text:
                context = " ".join(list(self.transcription_history)[-3:])
                original_text = text
                text = self.post_processor.correct_text(text, context)
                if text != original_text:
                    print(f"🔧 LLM corrected: '{original_text}' → '{text}'")

            if text:
                self.transcription_history.append(text)
            return text # Return even if empty, let caller decide

        except Exception as e:
            print(f"❌ Transcription error: {e}")
            import traceback
            if self.debug_mode:
                traceback.print_exc()

            # Fallback for MPS issues with OpenAI Whisper (already partially handled in _load_model)
            if "mps" in str(e).lower() and self.device == "mps" and not self.use_faster_whisper:
                print("🔄 Attempting to switch OpenAI Whisper to CPU mode due to MPS error during transcription...")
                try:
                    self.device = "cpu"
                    self._load_model() # Reload model on CPU
                    if self.model: # If model loaded successfully on CPU
                        print("   Successfully switched to CPU, retrying transcription...")
                        return self.transcribe_audio(audio_data) # Retry
                    else:
                        print("   Failed to load model on CPU during fallback.")
                except Exception as retry_e:
                    print(f"❌ CPU fallback also failed: {retry_e}")
        return ""


class ImprovedRealTimeTranscriber:
    """改进的实时转录系统 - 解决录音连续性问题"""

    def __init__(self, args):
        self.args = args
        self.debug = getattr(args, 'debug', False)

        # 初始化连续音频录制器
        recording_method = getattr(args, 'recording_method', "sounddevice")
        microphone_index = getattr(args, 'microphone_index', None)
        energy_threshold = getattr(args, 'energy_threshold', 300)

        self.audio_recorder = ContinuousAudioRecorder(
            sample_rate=16000,
            method=recording_method,
            microphone_index=microphone_index,
            energy_threshold=energy_threshold
        )

        # 初始化转录器
        self.transcriber = SegmentedTranscriber(
            model_name=getattr(args, 'model', 'small'),
            use_faster_whisper=getattr(args, 'use_faster_whisper', True),
            enable_llm=getattr(args, 'enable_llm', False),
            device=getattr(args, 'device', 'auto'),
            language=getattr(args, 'language', 'en'),
            segment_duration=getattr(args, 'segment_duration', 8.0),
            overlap_duration=getattr(args, 'overlap_duration', 2.0),
            debug_mode=self.debug # Pass debug flag
        )

        # 音频分段管理
        self.current_segment = np.array([], dtype=np.float32)
        self.segment_start_time = None
        self.segment_count = 0
        self.current_wav_path_for_overlap = None # Store path of WAV used for *next* segment's overlap
        self.last_debug_time = 0  # 调试输出时间控制

        # 输出设置
        self.output_file = getattr(args, 'output_file', 'transcription.txt')
        self.save_audio = getattr(args, 'save_audio', False)
        self.audio_output_dir = getattr(args, 'audio_output_dir', 'recordings')
        self.show_timestamps = getattr(args, 'show_timestamps', True)

        # 转录结果
        self.current_transcription = []
        self.transcription_lock = threading.Lock()

        # 控制变量
        self.running = True

        # 创建输出目录
        if self.save_audio:
            os.makedirs(self.audio_output_dir, exist_ok=True)
        self._cleanup_old_files() # Call cleanup after ensuring dir exists

    def _cleanup_old_files(self):
        """清理旧文件"""
        import glob
        import shutil

        # 清理WAV文件
        if self.save_audio and os.path.exists(self.audio_output_dir): # Only if saving audio
            wav_files = glob.glob(os.path.join(self.audio_output_dir, "*.wav"))
            if wav_files:
                print(f"🗑️  Cleaning {len(wav_files)} old audio files from {self.audio_output_dir}")
                for wav_file in wav_files:
                    try:
                        os.remove(wav_file)
                    except Exception as e:
                        print(f"   Failed to delete {wav_file}: {e}")

        # 备份转录文件
        if self.output_file and os.path.exists(self.output_file):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{self.output_file}.backup_{timestamp}"
            try:
                shutil.move(self.output_file, backup_name)
                print(f"📄 Backed up old transcription: {backup_name}")
            except Exception as e:
                print(f"⚠️  Failed to backup {self.output_file}: {e}")

    def audio_collection_thread(self):
        """音频收集线程 - 连续收集音频数据"""
        print("🎵 Audio collection thread started")

        while self.running:
            try:
                # 检查录音器是否在运行
                if not self.audio_recorder.running:
                    time.sleep(0.1)  # 暂停时等待
                    continue
                
                # 从录音器获取音频数据
                audio_chunk = self.audio_recorder.get_audio(timeout=0.05) # Shorter timeout

                if audio_chunk is not None and len(audio_chunk) > 0:
                    self._add_to_current_segment(audio_chunk)

                    # 检查是否需要处理当前段
                    if self._should_process_segment():
                        self._process_current_segment()
                elif audio_chunk is None and not self.running : # Recorder stopped and queue empty
                    break

            except Exception as e:
                print(f"Audio collection error: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()
        print("🎵 Audio collection thread finished.")


    def _add_to_current_segment(self, audio_chunk):
        """添加音频到当前段"""
        if self.segment_start_time is None:
            self.segment_start_time = time.time()
            # No need to print here, _process_current_segment will announce
            self.last_debug_time = time.time()

        self.current_segment = np.concatenate([self.current_segment, audio_chunk])

        if self.debug:
            current_time = time.time()
            if current_time - self.last_debug_time >= 1.0:
                duration = len(self.current_segment) / 16000
                q_size = self.audio_recorder.get_queue_size()
                print(f"🔍 Segment {self.segment_count + 1} buffer: {duration:.1f}s audio. Queue: {q_size}")
                self.last_debug_time = current_time

    def _should_process_segment(self):
        """检查是否应该处理当前段"""
        if self.segment_start_time is None or len(self.current_segment) == 0:
            return False

        # Process if current segment is long enough (e.g. target duration)
        # OR if a certain time has elapsed (original logic)
        # This ensures that even with sparse audio, we eventually process what's there.
        # Let's primarily use time, but ensure there's *some* audio.
        elapsed_time = time.time() - self.segment_start_time
        return elapsed_time >= getattr(self.args, 'segment_duration', 8.0) and len(self.current_segment) > 0


    def _process_current_segment(self):
        """处理当前音频段"""
        if len(self.current_segment) == 0: # Should be caught by _should_process_segment
            return

        segment_to_process = self.current_segment.copy()
        # Crucial: Reset current_segment and its timer *before* async transcription.
        # The overlap logic uses self.current_wav_path_for_overlap (from *previous* segment)
        # and segment_to_process (the *current* audio collected).
        self.current_segment = np.array([], dtype=np.float32)
        self.segment_start_time = None # Reset for the next segment's timing
        
        self.segment_count += 1 # Increment segment counter early for logging
        segment_duration_actual = len(segment_to_process) / 16000
        print(f"📝 Processing segment {self.segment_count} ({segment_duration_actual:.1f}s actual audio)")

        # --- WAV Saving and Overlap Logic ---
        # The WAV saved here (S_n_wav) will be used for the *next* segment's overlap.
        # The audio transcribed for *this* segment (S_n) uses overlap from S_{n-1}_wav.
        
        current_segment_wav_path = None
        if self.save_audio:
            # Save the actual collected audio (segment_to_process),
            # but pad/truncate it to ensure the WAV for overlap is consistently segment_duration long.
            current_segment_wav_path = self._save_audio_for_overlap(segment_to_process, self.segment_count)

        # Prepare audio for transcription
        if self.segment_count == 1: # First segment
            transcription_audio = segment_to_process
            if self.debug:
                print(f"🎯 Segment {self.segment_count}: Direct transcription ({len(transcription_audio)/16000:.1f}s)")
        else: # Subsequent segments, use overlap
            transcription_audio = self._prepare_transcription_with_overlap(
                segment_to_process, # This is the new audio data for the current segment
                self.current_wav_path_for_overlap # Path to the WAV of the previous segment (S_{n-1}_wav)
            )

        # Asynchronous transcription
        transcription_thread = threading.Thread(
            target=self._transcribe_segment,
            args=(transcription_audio, self.segment_count, current_segment_wav_path), # Pass actual WAV path for this segment if saved
            daemon=True
        )
        transcription_thread.start()

        # Update the path for the *next* segment's overlap source
        self.current_wav_path_for_overlap = current_segment_wav_path

        print(f"📤 Segment {self.segment_count} queued for transcription.")


    def _save_audio_for_overlap(self, audio_data_raw, segment_num_being_processed):
        """
        Saves audio data, ensuring it's padded/truncated to segment_duration.
        This WAV is intended as the source for the *next* segment's overlap.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Filename indicates it's the WAV of the segment just processed, to be used for future overlap
        filename = f"segment_{timestamp}_{segment_num_being_processed:03d}_for_overlap.wav"
        filepath = os.path.join(self.audio_output_dir, filename)

        target_samples = int(getattr(self.args, 'segment_duration', 8.0) * 16000)
        current_samples = len(audio_data_raw)
        
        audio_to_save = audio_data_raw # Start with the raw audio

        if current_samples < target_samples:
            silence = np.zeros(target_samples - current_samples, dtype=np.float32)
            audio_to_save = np.concatenate([audio_data_raw, silence])
            if self.debug:
                print(f"💾 Padded segment {segment_num_being_processed} for overlap WAV: {current_samples/16000:.1f}s raw -> {getattr(self.args, 'segment_duration', 8.0):.1f}s saved.")
        elif current_samples > target_samples:
            audio_to_save = audio_data_raw[:target_samples]
            if self.debug:
                print(f"💾 Truncated segment {segment_num_being_processed} for overlap WAV: {current_samples/16000:.1f}s raw -> {getattr(self.args, 'segment_duration', 8.0):.1f}s saved.")
        # Else, current_samples == target_samples, save as is.

        audio_int16 = (audio_to_save * 32767).astype(np.int16)
        try:
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_int16.tobytes())
            
            if self.debug: # Only print full path if debug, otherwise just basename for clarity
                print(f"💾 Saved for overlap: {os.path.basename(filepath)} ({len(audio_to_save)/16000:.1f}s)")
            return filepath
        except Exception as e:
            print(f"❌ Failed to save segment for overlap: {e}")
            return None


    def _prepare_transcription_with_overlap(self, current_segment_audio, previous_segment_wav_path):
        """
        Prepares audio for transcription by prepending overlap from the previous segment's WAV.
        'current_segment_audio' is the new audio collected for the current segment.
        'previous_segment_wav_path' is the path to the WAV file of the segment S_{n-1}.
        """
        if previous_segment_wav_path and os.path.exists(previous_segment_wav_path):
            try:
                overlap_audio = self._extract_overlap_from_wav(previous_segment_wav_path)
                if len(overlap_audio) > 0:
                    # Audio for transcription = Overlap from S_{n-1} + New audio S_n
                    transcription_audio = np.concatenate([overlap_audio, current_segment_audio])
                    if self.debug:
                        overlap_dur = len(overlap_audio)/16000
                        current_dur = len(current_segment_audio)/16000
                        total_dur = len(transcription_audio)/16000
                        print(f"🔗 Segment {self.segment_count}: Using {overlap_dur:.1f}s overlap from '{os.path.basename(previous_segment_wav_path)}' + {current_dur:.1f}s new audio = {total_dur:.1f}s total for Whisper.")
                    return transcription_audio
                else:
                    if self.debug:
                        print(f"⚠️ No overlap audio extracted from {os.path.basename(previous_segment_wav_path)}. Transcribing current segment directly.")
            except Exception as e:
                print(f"❌ Failed to extract overlap from {previous_segment_wav_path}: {e}. Transcribing current segment directly.")
        else:
            if self.debug and previous_segment_wav_path: # If path was provided but not found
                print(f"⚠️ Previous segment WAV '{os.path.basename(previous_segment_wav_path)}' not found. Transcribing current segment directly.")
            elif self.debug and not previous_segment_wav_path: # If no path was provided (e.g. save_audio was off)
                print(f"🎯 Segment {self.segment_count}: No previous WAV path for overlap. Transcribing current segment directly.")


        # Fallback: transcribe only the current segment's audio if overlap fails or not available
        if self.debug and not (previous_segment_wav_path and os.path.exists(previous_segment_wav_path) and len(overlap_audio)>0) : # only print if not already printed in detail
             print(f"🎯 Segment {self.segment_count}: Direct transcription (no overlap) ({len(current_segment_audio)/16000:.1f}s)")
        return current_segment_audio


    def _extract_overlap_from_wav(self, wav_filepath):
        """从WAV文件末尾提取重叠部分"""
        try:
            with wave.open(wav_filepath, 'rb') as wav_file:
                n_frames = wav_file.getnframes()
                frames = wav_file.readframes(n_frames)
                audio_data_int16 = np.frombuffer(frames, dtype=np.int16)
                audio_data_float32 = audio_data_int16.astype(np.float32) / 32768.0

            overlap_samples = int(getattr(self.args, 'overlap_duration', 2.0) * 16000)
            if len(audio_data_float32) >= overlap_samples:
                overlap_audio = audio_data_float32[-overlap_samples:]
                if self.debug:
                    print(f"🔍 Extracted {len(overlap_audio)/16000:.1f}s overlap from {os.path.basename(wav_filepath)}")
                return overlap_audio
            else: # WAV shorter than desired overlap
                if self.debug:
                    print(f"🔍 WAV file {os.path.basename(wav_filepath)} ({len(audio_data_float32)/16000:.1f}s) is shorter than overlap duration ({getattr(self.args, 'overlap_duration', 2.0):.1f}s). Using entire file as overlap.")
                return audio_data_float32

        except Exception as e:
            print(f"❌ Error extracting overlap from {os.path.basename(wav_filepath)}: {e}")
            return np.array([], dtype=np.float32)


    def _transcribe_segment(self, audio_data, segment_num_being_transcribed, wav_path_of_this_segment=None): # wav_path is for info
        """转录音频段"""
        start_time = time.time()

        try:
            # Audio length check already in SegmentedTranscriber.transcribe_audio
            # Additional pre-transcription checks specific to this stage
            if audio_data is None or len(audio_data) == 0:
                 print(f"⚠️ Segment {segment_num_being_transcribed}: Received empty audio data for transcription. Skipping.")
                 self._update_transcription("(Empty audio data)", segment_num_being_transcribed, time.time() - start_time)
                 return

            # Debug: Min/Max/Mean/Std of audio data being sent to Whisper
            if self.debug:
                min_val, max_val, mean_val, std_val = np.min(audio_data), np.max(audio_data), np.mean(audio_data), np.std(audio_data)
                print(f"🧠 Segment {segment_num_being_transcribed} ({len(audio_data)/16000:.1f}s audio) PRE-WHISPER STATS: Min={min_val:.4f}, Max={max_val:.4f}, Mean={mean_val:.4f}, Std={std_val:.4f}, dtype={audio_data.dtype}")

            # Stricter silence/low audio check before calling Whisper
            # Whisper VAD is good, but extremely low signals might still be problematic or slow
            # Max absolute amplitude; 0.001 is approx -60dBFS, 0.005 is approx -46dBFS
            # Allow very quiet audio through if VAD is expected to pick it up.
            # This check can be made less aggressive if Whisper's VAD is preferred.
            # if np.max(np.abs(audio_data)) < 0.005:
            #     print(f"⚠️  Segment {segment_num_being_transcribed}: Audio signal potentially too low (max abs: {np.max(np.abs(audio_data)):.2e}). Relying on Whisper VAD.")
                # self._update_transcription("(Silence or very low audio detected pre-Whisper)", segment_num_being_transcribed, time.time() - start_time)
                # return # Or let Whisper try

            # Execute transcription
            result_text = self.transcriber.transcribe_audio(audio_data) # This now returns text
            transcription_time = time.time() - start_time

            if result_text and result_text.strip():
                self._update_transcription(result_text, segment_num_being_transcribed, transcription_time)
                # No need for extra print here, _update_transcription handles it
            else: # No text from Whisper (empty string or all whitespace)
                # This means Whisper processed it but found no speech, or an error occurred in transcribe_audio
                # which should have printed its own message.
                print(f"⚠️  Segment {segment_num_being_transcribed}: No transcription result from Whisper (likely silence or unclear audio based on Whisper's VAD/analysis). Time: {transcription_time:.1f}s")
                self._update_transcription("(No speech detected by Whisper)", segment_num_being_transcribed, transcription_time)

        except Exception as e:
            # This is a catch-all for unexpected errors within _transcribe_segment itself.
            # Errors within self.transcriber.transcribe_audio() should be handled there.
            print(f"❌ Critical error during transcription task for segment {segment_num_being_transcribed}: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            self._update_transcription("(Transcription error)", segment_num_being_transcribed, time.time() - start_time)


    def _update_transcription(self, new_text, segment_num, transcription_time):
        """更新转录结果"""
        with self.transcription_lock:
            timestamp = datetime.now().strftime('%H:%M:%S')

            # ANSI颜色代码
            CYAN = '\033[96m'
            GREEN = '\033[92m'
            YELLOW = '\033[93m'
            RED = '\033[91m'
            RESET = '\033[0m'
            BOLD = '\033[1m'

            # Determine display based on content of new_text
            segment_label = f"{BOLD}{CYAN}📝 Seg {segment_num} Result:{RESET}"
            display_text_formatted = ""

            if new_text == "(No speech detected by Whisper)":
                display_text_formatted = f"{YELLOW}{new_text}{RESET}"
            elif new_text in ["(Empty audio data)", "(Transcription error)", "(Silence or very low audio detected pre-Whisper)"]:
                display_text_formatted = f"{RED}{new_text}{RESET}"
            elif new_text.strip(): # Actual transcription
                display_text_formatted = f"{GREEN}{new_text}{RESET}"
            else: # Should not happen if previous checks are done, but as a fallback
                display_text_formatted = f"{YELLOW}(Empty result){RESET}"


            print(f"\n{'='*20} Segment {segment_num} {'='*20}")
            print(f"{segment_label} {display_text_formatted}")
            print(f"{CYAN}⏱️  Processing time: {transcription_time:.1f}s{RESET}")
            print(f"{'='*51}\n")


            # Prepare text for history and file
            if self.show_timestamps:
                # Use the actual content of new_text for logging, not the formatted one
                history_entry = f"[{timestamp}] Segment {segment_num}: {new_text}"
            else:
                history_entry = f"Segment {segment_num}: {new_text}"

            self.current_transcription.append(history_entry)

            # Save to file
            if self.output_file:
                self._save_to_file()

    def _save_to_file(self):
        """保存转录结果到文件"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for line in self.current_transcription:
                    f.write(line + '\n')
        except Exception as e:
            print(f"Failed to save file {self.output_file}: {e}")

    def display_status(self):
        """显示状态信息"""
        # (ANSI codes defined in _update_transcription can be reused or defined here)
        BLUE = '\033[94m'; CYAN = '\033[96m'; GREEN = '\033[92m'; YELLOW = '\033[93m'; MAGENTA = '\033[95m'; RESET = '\033[0m'; BOLD = '\033[1m'

        os.system('cls' if os.name == 'nt' else 'clear')

        # Language display
        lang_map = {"en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "es": "Spanish", "fr": "French", "de": "German", "auto": "Auto Detect"}
        language_info = lang_map.get(self.transcriber.target_language, self.transcriber.target_language)

        print(f"{BOLD}{BLUE}=== Real-Time Speech Transcription (Ctrl+C to pause/access menu) ==={RESET}")
        model_type = "Faster-Whisper" if self.transcriber.use_faster_whisper else "OpenAI-Whisper"
        print(f"{CYAN}Config: Lang: {language_info} | Device: {self.transcriber.device} | Model: {getattr(self.args, 'model', 'small')} ({model_type}){RESET}")

        # 修改的部分：显示实际的麦克风设备名称
        mic_info = self.audio_recorder.device_name
        if getattr(self.args, 'microphone_index', None) is not None:
            mic_info = f"{mic_info} (Index: {self.args.microphone_index})"
        
        energy_info = f" | Energy: {getattr(self.args, 'energy_threshold', 300)}" if self.audio_recorder.method == "speech_recognition" else ""
        print(f"{CYAN}Audio: {self.audio_recorder.method} | Mic: {mic_info}{energy_info} | Segments: {getattr(self.args, 'segment_duration', 8.0)}s (Overlap: {getattr(self.args, 'overlap_duration', 2.0)}s){RESET}")

        if self.output_file: print(f"{YELLOW}Output File: {self.output_file}{RESET}")
        if self.save_audio: print(f"{YELLOW}Saved Audio: {self.audio_output_dir}/ (Segments processed: {self.segment_count}){RESET}")
        print()

        # Display recent transcriptions
        print(f"{BOLD}--- Recent Transcriptions ---{RESET}")
        with self.transcription_lock:
            for i, line in enumerate(self.current_transcription[-8:], 1): # Show last 8 lines
                # Simple coloring for segment part vs content part
                if ": " in line:
                    parts = line.split(": ", 1)
                    print(f"{i:2d}. {CYAN}{parts[0]}:{RESET} {parts[1]}") # Color depends on content, handled by _update
                else:
                    print(f"{i:2d}. {line}")
        print(f"{BOLD}-----------------------------{RESET}")


        # Current status
        status_line = f"{MAGENTA}[{datetime.now().strftime('%H:%M:%S')}]{RESET}"
        if not self.audio_recorder.running and self.running:
            status_line += f" {YELLOW}⏸️  Recording PAUSED - Press Ctrl+C to access options menu{RESET}"
        elif self.segment_start_time is not None and len(self.current_segment) > 0 and self.audio_recorder.running:
            current_segment_audio_duration = len(self.current_segment) / 16000
            elapsed_time_current_segment = time.time() - self.segment_start_time
            progress = min((elapsed_time_current_segment / getattr(self.args, 'segment_duration', 8.0)) * 100, 100.0)
            status_line += f" {YELLOW}Recording segment {self.segment_count + 1}... ({elapsed_time_current_segment:.1f}s / {getattr(self.args, 'segment_duration', 8.0)}s target | Audio in buffer: {current_segment_audio_duration:.1f}s){RESET}"
        elif self.running and self.audio_recorder.running:
            status_line += f" {GREEN}Listening... (Waiting for audio to start new segment){RESET}"
        else:
            status_line += f" {RED}Shutting down...{RESET}"

        if self.debug:
            queue_size = self.audio_recorder.get_queue_size()
            status_line += f" | RecQueue: {queue_size}"

        print(f"\n{status_line}")


    def run(self):
        """运行改进的转录系统"""
        print("🚀 Starting real-time transcription system...")
        print(f"   🗣️ Target Language: {self.transcriber.target_language}")
        print(f"   ⚙️  Whisper Model: {getattr(self.args, 'model', 'small')} ({'Faster-Whisper' if self.transcriber.use_faster_whisper else 'OpenAI-Whisper'}) on {self.transcriber.device}")
        print(f"   🎤 Recording Method: {self.audio_recorder.method}")
        print(f"   🎙️  Microphone: {self.audio_recorder.device_name}")  # 添加麦克风设备名称显示
        print(f"   🕒 Segments: {getattr(self.args, 'segment_duration', 8.0)}s, Overlap: {getattr(self.args, 'overlap_duration', 2.0)}s")
        if getattr(self.args, 'enable_llm', False): print(f"   🤖 LLM Post-processing: Enabled ({self.transcriber.post_processor.model_name if self.transcriber.post_processor else 'Error'})")
        if self.debug: print(f"   🐞 Debug Mode: Enabled")
        print()

        # 启动音频录制
        if not self.audio_recorder.start_recording():
            print("❌ CRITICAL: Failed to start audio recording. Exiting.")
            return

        # 启动音频收集线程
        self.collection_thread_instance = threading.Thread(target=self.audio_collection_thread, daemon=True)
        self.collection_thread_instance.start()

        time.sleep(0.5) # Brief pause for threads to initialize
        if not self.collection_thread_instance.is_alive():
            print("❌ CRITICAL: Audio collection thread failed to start. Exiting.")
            self.audio_recorder.stop_recording() # Attempt to clean up
            return

        print("✅ All systems nominal. Begin speaking...")
        print("💡 Press Ctrl+C anytime to pause and access options menu (resume or exit)")
        print()

        try:
            loop_count = 0
            while self.running: # Check self.running, controlled by KeyboardInterrupt
                time.sleep(0.5) # Update status display interval
                loop_count += 1

                # Update display (e.g., every 1 second, so 2 loops of 0.5s)
                if loop_count % 2 == 0:
                    self.display_status()

                if not self.collection_thread_instance.is_alive() and self.running:
                    print("⚠️  Audio collection thread unexpectedly stopped. Attempting to restart...")
                    # Basic restart logic, could be more robust
                    self.collection_thread_instance = threading.Thread(target=self.audio_collection_thread, daemon=True)
                    self.collection_thread_instance.start()
                    if not self.collection_thread_instance.is_alive():
                        print("❌ CRITICAL: Failed to restart audio collection thread. Exiting.")
                        self.running = False # Signal main loop to stop
                        break # Exit while loop
                    time.sleep(0.5) # Give it a moment

        except KeyboardInterrupt:
            # 暂停录音但不完全停止
            print("\n\n🛑 Ctrl+C received. Pausing transcription...")
            
            # 暂停录音
            self.audio_recorder.pause_recording()
            
            # 给用户选择
            while True:
                try:
                    print("\n📋 What would you like to do?")
                    print("   1. Resume recording and transcription")
                    print("   2. Stop and exit program")
                    choice = input("\nEnter your choice (1 or 2): ").strip()
                    
                    if choice == "1":
                        print("🔄 Resuming transcription...")
                        # 重新启动录音
                        if not self.audio_recorder.resume_recording():
                            print("❌ Failed to resume audio recording. Exiting.")
                            self.running = False
                            break
                        
                        # 重新启动音频收集线程（如果已停止）
                        if not self.collection_thread_instance.is_alive():
                            self.collection_thread_instance = threading.Thread(target=self.audio_collection_thread, daemon=True)
                            self.collection_thread_instance.start()
                            time.sleep(0.5)  # 给线程时间启动
                            
                            if not self.collection_thread_instance.is_alive():
                                print("❌ Failed to restart audio collection thread. Exiting.")
                                self.running = False
                                break
                        
                        print("✅ Recording resumed. Continue speaking...")
                        break  # 退出选择循环，继续主循环
                        
                    elif choice == "2":
                        print("🛑 Stopping transcription system...")
                        self.running = False
                        break  # 退出选择循环和主循环
                        
                    else:
                        print("❌ Invalid choice. Please enter 1 or 2.")
                        
                except KeyboardInterrupt:
                    print("\n🛑 Force exit requested.")
                    self.running = False
                    break
            
            # 如果用户选择恢复且系统仍在运行，回到主循环
            if self.running:
                try:
                    loop_count = 0
                    while self.running:
                        time.sleep(0.5)
                        loop_count += 1

                        if loop_count % 2 == 0:
                            self.display_status()

                        if not self.collection_thread_instance.is_alive() and self.running:
                            print("⚠️  Audio collection thread unexpectedly stopped. Attempting to restart...")
                            self.collection_thread_instance = threading.Thread(target=self.audio_collection_thread, daemon=True)
                            self.collection_thread_instance.start()
                            if not self.collection_thread_instance.is_alive():
                                print("❌ CRITICAL: Failed to restart audio collection thread. Exiting.")
                                self.running = False
                                break
                            time.sleep(0.5)
                except KeyboardInterrupt:
                    # 如果在恢复后再次按Ctrl+C，递归调用相同的逻辑
                    return self.run()  # 重新进入主运行逻辑

        finally:
            print("Shutting down audio systems...")
            self.audio_recorder.stop_recording()

            if hasattr(self, 'collection_thread_instance') and self.collection_thread_instance.is_alive():
                print("Waiting for audio collection thread to finish...")
                self.collection_thread_instance.join(timeout=2.0) # Wait for thread
                if self.collection_thread_instance.is_alive():
                    print("⚠️ Audio collection thread did not finish in time.")


            # Process any remaining audio in current_segment
            if len(self.current_segment) > 16000 * 0.2: # Process if more than 0.2s left
                print(f"🔄 Processing final audio segment ({len(self.current_segment)/16000:.1f}s)...")
                # Use a simplified processing for the final bit, no complex overlap needed
                final_audio_to_transcribe = self.current_segment.copy()
                self.current_segment = np.array([], dtype=np.float32) # Clear it
                self.segment_count+=1
                
                # Run transcription for the final segment synchronously for simplicity upon exit
                # Or launch thread and wait briefly if preferred.
                # For now, let's keep it threaded but be mindful on exit.
                final_transcription_thread = threading.Thread(
                    target=self._transcribe_segment,
                    args=(final_audio_to_transcribe, self.segment_count, None), # No WAV path for this final bit necessarily
                    daemon=True # Keep as daemon; main thread will wait for active threads below
                )
                final_transcription_thread.start()


            # Wait for active transcription threads to complete
            active_threads = [t for t in threading.enumerate() if t.daemon and t != threading.current_thread() and t != self.collection_thread_instance]
            if active_threads:
                print(f"Waiting for {len(active_threads)} active transcription threads to complete...")
                for t in active_threads:
                    t.join(timeout=getattr(self.args, 'segment_duration', 8.0) + 5.0) # Generous timeout
                    if t.is_alive():
                        print(f"⚠️ Thread {t.name} did not complete in time.")


            # Final save to file
            if self.output_file:
                with self.transcription_lock: # Ensure thread safety for final save
                    self._save_to_file()
                print(f"✅ Final results saved to: {self.output_file}")

            if self.save_audio:
                print(f"🎵 Audio files saved in: {self.audio_output_dir}/ (Total segments processed: {self.segment_count})")

            # Display final summary
            BLUE = '\033[94m'; CYAN = '\033[96m'; GREEN = '\033[92m'; RESET = '\033[0m'; BOLD = '\033[1m'
            print(f"\n{BOLD}{BLUE}=== Final Transcription Summary ==={RESET}")
            with self.transcription_lock:
                if self.current_transcription:
                    for i, line in enumerate(self.current_transcription, 1):
                        if ": " in line:
                            parts = line.split(": ", 1)
                            print(f"{i:2d}. {CYAN}{parts[0]}:{RESET} {parts[1]}")
                        else:
                            print(f"{i:2d}. {line}")
                else:
                    print(f"{YELLOW}No transcriptions were generated.{RESET}")

            print(f"\n{GREEN}Transcription system shut down.{RESET}")
            print(f"📊 Total segments processed: {self.segment_count}")


def interactive_setup():
    """交互式设置参数"""
    print("🎙️  Welcome to Real-Time Speech Transcription Setup")
    print("=" * 50)
    print("This interactive mode will help you configure the transcription system.")
    print("You can also run with command line arguments for direct execution.")
    print("Example: python transcriber.py --language zh --model small --save_audio")
    print("💡 During transcription, press Ctrl+C to pause and choose to resume or exit.")
    
    # 检查库可用性
    print("\n📦 Checking library availability...")
    print(f"   Sounddevice:      {'✅ Available' if SOUNDDEVICE_AVAILABLE else '❌ Not Available'}")
    print(f"   PyAudio:          {'✅ Available' if PYAUDIO_AVAILABLE else '❌ Not Available'}")
    print(f"   Faster-Whisper:   {'✅ Available' if FASTER_WHISPER_AVAILABLE else '❌ Not Available'}")
    
    class InteractiveArgs:
        def __init__(self):
            pass
    
    args = InteractiveArgs()
    
    # 语言选择
    print("\n🌍 Language Selection:")
    lang_options = {
        "1": ("en", "English"),
        "2": ("zh", "Chinese"),
        "3": ("ja", "Japanese"),
        "4": ("ko", "Korean"),
        "5": ("es", "Spanish"),
        "6": ("fr", "French"),
        "7": ("de", "German"),
        "8": ("auto", "Auto Detect")
    }
    
    for key, (code, name) in lang_options.items():
        print(f"   {key}. {name} ({code})")
    
    while True:
        choice = input("\nSelect language (1-8, default: 1 for English): ").strip()
        if not choice:
            choice = "1"
        if choice in lang_options:
            args.language = lang_options[choice][0]
            print(f"   Selected: {lang_options[choice][1]}")
            break
        print("   Invalid choice, please try again.")
    
    # 模型选择
    print("\n🤖 Whisper Model Selection:")
    model_options = ["tiny", "base", "small", "medium", "large"]
    if FASTER_WHISPER_AVAILABLE:
        model_options.extend(["tiny.en", "base.en", "small.en", "medium.en", "large.en"])
    
    for i, model in enumerate(model_options, 1):
        print(f"   {i}. {model}")
    
    while True:
        choice = input(f"\nSelect model (1-{len(model_options)}, default: 3 for small): ").strip()
        if not choice:
            choice = "3"
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(model_options):
                args.model = model_options[idx]
                print(f"   Selected: {args.model}")
                break
        except ValueError:
            pass
        print("   Invalid choice, please try again.")
    
    # 设备选择
    print("\n💻 Processing Device:")
    device_options = ["auto", "cpu", "cuda", "mps"]
    for i, device in enumerate(device_options, 1):
        print(f"   {i}. {device}")
    
    while True:
        choice = input("\nSelect device (1-4, default: 1 for auto): ").strip()
        if not choice:
            choice = "1"
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(device_options):
                args.device = device_options[idx]
                print(f"   Selected: {args.device}")
                break
        except ValueError:
            pass
        print("   Invalid choice, please try again.")
    
    # 录音方法选择
    print("\n🎵 Recording Method:")
    recording_methods = []
    if SOUNDDEVICE_AVAILABLE:
        recording_methods.append("sounddevice")
    if PYAUDIO_AVAILABLE:
        recording_methods.append("pyaudio")
    recording_methods.append("speech_recognition")
    
    for i, method in enumerate(recording_methods, 1):
        status = "✅ Recommended" if method == "sounddevice" else ""
        print(f"   {i}. {method} {status}")
    
    while True:
        choice = input(f"\nSelect recording method (1-{len(recording_methods)}, default: 1): ").strip()
        if not choice:
            choice = "1"
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(recording_methods):
                args.recording_method = recording_methods[idx]
                print(f"   Selected: {args.recording_method}")
                break
        except ValueError:
            pass
        print("   Invalid choice, please try again.")
    
    # 麦克风选择
    print("\n🎤 Microphone Selection:")
    print("   Scanning available microphones...")
    
    # 显示可用麦克风
    if args.recording_method == "sounddevice" and SOUNDDEVICE_AVAILABLE:
        try:
            devices = sd.query_devices()
            input_devices = []
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append((i, device['name']))
                    default_marker = "(default)" if i == (sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device) else ""
                    print(f"   {len(input_devices)}. [{i}] {device['name']} {default_marker}")
        except Exception as e:
            print(f"   Error listing devices: {e}")
            input_devices = []
    elif args.recording_method == "pyaudio" and PYAUDIO_AVAILABLE:
        try:
            p = pyaudio.PyAudio()
            input_devices = []
            default_idx = -1
            try:
                default_idx = p.get_default_input_device_info()['index']
            except:
                pass
            
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    input_devices.append((i, info['name']))
                    default_marker = "(default)" if i == default_idx else ""
                    print(f"   {len(input_devices)}. [{i}] {info['name']} {default_marker}")
            p.terminate()
        except Exception as e:
            print(f"   Error listing devices: {e}")
            input_devices = []
    else:  # speech_recognition
        try:
            mic_names = sr.Microphone.list_microphone_names()
            input_devices = [(i, name) for i, name in enumerate(mic_names)]
            for i, (idx, name) in enumerate(input_devices, 1):
                print(f"   {i}. [{idx}] {name}")
        except Exception as e:
            print(f"   Error listing devices: {e}")
            input_devices = []
    
    print(f"   0. Use default microphone")
    
    while True:
        choice = input(f"\nSelect microphone (0-{len(input_devices)}, default: 0): ").strip()
        if not choice:
            choice = "0"
        try:
            idx = int(choice)
            if idx == 0:
                args.microphone_index = None
                print("   Selected: Default microphone")
                break
            elif 1 <= idx <= len(input_devices):
                args.microphone_index = input_devices[idx-1][0]
                print(f"   Selected: {input_devices[idx-1][1]} (Index: {args.microphone_index})")
                break
        except ValueError:
            pass
        print("   Invalid choice, please try again.")
    
    # 高级设置
    print("\n⚙️  Advanced Settings:")
    
    # Faster-Whisper选择
    if FASTER_WHISPER_AVAILABLE:
        choice = input("Use Faster-Whisper? (Y/n, default: Y): ").strip().lower()
        args.use_faster_whisper = choice != 'n'
        print(f"   Faster-Whisper: {'Enabled' if args.use_faster_whisper else 'Disabled'}")
    else:
        args.use_faster_whisper = False
    
    # LLM后处理
    choice = input("Enable LLM post-processing? (y/N, default: N): ").strip().lower()
    args.enable_llm = choice == 'y'
    print(f"   LLM post-processing: {'Enabled' if args.enable_llm else 'Disabled'}")
    
    # 音频保存
    choice = input("Save audio segments? (Y/n, default: Y): ").strip().lower()
    args.save_audio = choice != 'n'
    print(f"   Save audio: {'Enabled' if args.save_audio else 'Disabled'}")
    
    # 调试模式
    choice = input("Enable debug mode? (y/N, default: N): ").strip().lower()
    args.debug = choice == 'y'
    print(f"   Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    
    # 其他参数使用默认值
    args.segment_duration = 8.0
    args.overlap_duration = 2.0
    args.energy_threshold = 300
    args.output_file = "transcription.txt"
    args.audio_output_dir = "recordings"
    args.show_timestamps = True
    args.list_mics = False  # 交互式模式不需要列出麦克风
    
    # 段落设置
    print("\n⏱️  Timing Settings:")
    try:
        segment_duration = input(f"Segment duration in seconds (default: {args.segment_duration}): ").strip()
        if segment_duration:
            args.segment_duration = float(segment_duration)
    except ValueError:
        print("   Invalid input, using default.")
    
    try:
        overlap_duration = input(f"Overlap duration in seconds (default: {args.overlap_duration}): ").strip()
        if overlap_duration:
            args.overlap_duration = float(overlap_duration)
    except ValueError:
        print("   Invalid input, using default.")
    
    if args.recording_method == "speech_recognition":
        try:
            energy = input(f"Energy threshold (default: {args.energy_threshold}): ").strip()
            if energy:
                args.energy_threshold = int(energy)
        except ValueError:
            print("   Invalid input, using default.")
    
    # 输出文件
    output_file = input(f"Output file (default: {args.output_file}): ").strip()
    if output_file:
        args.output_file = output_file
    
    if args.save_audio:
        audio_dir = input(f"Audio output directory (default: {args.audio_output_dir}): ").strip()
        if audio_dir:
            args.audio_output_dir = audio_dir
    
    print("\n✅ Setup complete! Starting transcription...")
    print("=" * 50)
    return args


def main():
    parser = argparse.ArgumentParser(
        description="Real-time speech transcription with continuous recording and Whisper.",
        formatter_class=argparse.RawTextHelpFormatter, # Changed for better help text
        epilog="""Examples:
  %(prog)s --language en --model small
  %(prog)s --list_mics
  %(prog)s --debug --save_audio --language auto --model base --segment_duration 10 --overlap_duration 2
"""
    )

    # Model parameters
    model_choices = ["tiny", "base", "small", "medium", "large"]
    if FASTER_WHISPER_AVAILABLE: # Faster-Whisper supports more, e.g. tiny.en, base.en
        model_choices.extend([f"{m}.en" for m in model_choices if "." not in m])
        model_choices.extend(["large-v2", "large-v3", "distil-large-v2", "distil-medium.en", "distil-small.en"]) # Add newer/distilled if desired
        model_choices = sorted(list(set(model_choices)))


    parser.add_argument("--model", default="small",
                       choices=model_choices,
                       help=f"Whisper model size (default: small). Choices: {', '.join(model_choices)}")
    parser.add_argument("--use_faster_whisper", action=argparse.BooleanOptionalAction, default=True, # Use new action
                       help="Use Faster-Whisper if available (default: True). Use --no-use_faster_whisper to disable.")
    parser.add_argument("--enable_llm", action=argparse.BooleanOptionalAction, default=False,
                       help="Enable LLM post-processing (default: False). Requires Ollama and a model like llama3.2.")
    parser.add_argument("--device", default="auto",
                       choices=["auto", "cpu", "mps", "cuda"],
                       help="Compute device for Whisper (default: auto).")

    # Language parameters
    lang_choices = ["auto", "en", "zh", "de", "es", "fr", "ja", "ko", "ru", "it", "pt", "nl", "sv", "pl", "tr", "uk", "he", "ar", "hi"] # Expanded list
    parser.add_argument("--language", default="en",
                       choices=lang_choices,
                       help=f"Target language for transcription (default: en). 'auto' for detection. Choices: {', '.join(lang_choices)}")

    # Recording parameters
    parser.add_argument("--recording_method", default="sounddevice",
                       choices=["sounddevice", "pyaudio", "speech_recognition"],
                       help="Audio recording method (default: sounddevice).")
    parser.add_argument("--microphone_index", default=None, type=int,
                       help="Microphone device index (optional). Use --list_mics to see available devices.")
    parser.add_argument("--energy_threshold", default=300, type=int,
                       help="Energy threshold for speech detection (primarily for speech_recognition method, default: 300).")
    parser.add_argument("--segment_duration", default=8.0, type=float,
                       help="Duration of audio segments processed by Whisper (seconds, default: 8.0).")
    parser.add_argument("--overlap_duration", default=2.0, type=float,
                       help="Duration of overlap between segments (seconds, default: 2.0). Ensures smoother transitions.")

    # Utility parameters
    parser.add_argument("--list_mics", action="store_true", default=False,
                       help="List available microphone devices and exit.")

    # Output parameters
    parser.add_argument("--output_file", default="transcription.txt",
                       help="File to save the transcription (default: transcription.txt).")
    parser.add_argument("--save_audio", action=argparse.BooleanOptionalAction, default=False, # Changed default to False
                       help="Save audio segments to files (default: False). Use --no-save_audio to disable explicitly if default changes.")
    parser.add_argument("--audio_output_dir", default="recordings",
                       help="Directory to save audio files if --save_audio is enabled (default: recordings).")
    parser.add_argument("--show_timestamps", action=argparse.BooleanOptionalAction, default=True,
                       help="Show timestamps in the output and file (default: True).")
    parser.add_argument("--debug", action="store_true", default=False, # Keep as store_true
                       help="Enable detailed debug mode logging (default: False).")

    # 检查是否没有提供任何参数（交互式模式）
    if len(sys.argv) == 1:
        try:
            args = interactive_setup()
        except KeyboardInterrupt:
            print("\n\n🛑 Setup cancelled by user.")
            return
        except Exception as e:
            print(f"\n❌ Error during interactive setup: {e}")
            return
    else:
        args = parser.parse_args()

    # List microphone devices and exit
    if hasattr(args, 'list_mics') and args.list_mics:
        print("🎤 Listing available microphone devices:")
        if SOUNDDEVICE_AVAILABLE:
            print("\n🔊 Sounddevice devices:")
            try:
                devices = sd.query_devices()
                found_input = False
                for i, device in enumerate(devices):
                    if device['max_input_channels'] > 0:
                        default_marker = "(default input)" if i == (sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device) else ""
                        print(f"  Index {i}: {device['name']} (Input Channels: {device['max_input_channels']}) {default_marker}")
                        found_input = True
                if not found_input: print("  No input devices found by sounddevice.")
            except Exception as e: print(f"  ❌ Error listing sounddevice devices: {e}")
        else: print("  Sounddevice library not available.")

        if PYAUDIO_AVAILABLE:
            print("\n🎵 PyAudio devices:")
            try:
                p = pyaudio.PyAudio()
                default_input_idx = -1
                try:
                    default_input_idx = p.get_default_input_device_info()['index']
                except IOError: pass # No default input device

                found_input = False
                for i in range(p.get_device_count()):
                    info = p.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0:
                        default_marker = "(default input)" if i == default_input_idx else ""
                        print(f"  Index {i}: {info['name']} (Input Channels: {info['maxInputChannels']}) {default_marker}")
                        found_input = True
                if not found_input: print("  No input devices found by PyAudio.")
                p.terminate()
            except Exception as e: print(f"  ❌ Error listing PyAudio devices: {e}")
        else: print("  PyAudio library not available.")

        # SpeechRecognition lists all, not just input, but useful for its specific indexing
        print("\n🗣️ SpeechRecognition microphone names (for its own indexing):")
        try:
            import speech_recognition as sr_list_mics # Separate import for this utility
            mic_names = sr_list_mics.Microphone.list_microphone_names()
            if mic_names:
                for index, name in enumerate(mic_names):
                    print(f"  Index {index}: {name}")
            else: print("  No microphones found by SpeechRecognition.")
        except ImportError: print("  SpeechRecognition library not available or failed to import for listing.")
        except Exception as e: print(f"  ❌ Error listing SpeechRecognition microphones: {e}")
        return

    # Parameter validation
    if hasattr(args, 'overlap_duration') and args.overlap_duration < 0:
        print("❌ Error: Overlap duration cannot be negative.")
        return
    if hasattr(args, 'segment_duration') and args.segment_duration <= 0:
        print("❌ Error: Segment duration must be positive.")
        return
    if (hasattr(args, 'overlap_duration') and hasattr(args, 'segment_duration') and 
        args.overlap_duration >= args.segment_duration):
        print("❌ Error: Overlap duration must be less than segment duration for meaningful processing.")
        print(f"   Your settings: Overlap={args.overlap_duration}s, Segment={args.segment_duration}s")
        print("   Consider reducing overlap_duration or increasing segment_duration.")
        return


    # Display library availability
    print("\n📦 Library Availability:")
    print(f"   Sounddevice:      {'✅ Available' if SOUNDDEVICE_AVAILABLE else '❌ Not Available'}")
    print(f"   PyAudio:          {'✅ Available' if PYAUDIO_AVAILABLE else '❌ Not Available'}")
    print(f"   SpeechRec lib:    {'✅ Available' if 'sr' in sys.modules or 'speech_recognition' in sys.modules else '❌ Not Available'}")
    print(f"   Faster-Whisper:   {'✅ Available' if FASTER_WHISPER_AVAILABLE else '❌ Not Available'}")
    print(f"   Ollama (for LLM): {'✅ Available' if 'ollama' in sys.modules else '❔ Not checked (used if --enable_llm)'}")
    print()

    # Microphone access check (simplified, relies on chosen method's init)
    # A more thorough check might try to open a stream briefly with the chosen method.
    # For now, let the ContinuousAudioRecorder handle initialization errors.
    print("🎤 Attempting to initialize chosen audio recording method...")
    # (The actual check happens when ContinuousAudioRecorder is initialized)

    # Create and run the transcriber
    try:
        transcriber_system = ImprovedRealTimeTranscriber(args)
        transcriber_system.run()
    except RuntimeError as e: # Catch runtime errors from model loading or audio init
        print(f"🚨 A critical error occurred during initialization: {e}")
        print("   Please check your setup, model availability, and device compatibility.")
    except Exception as e: # Catch any other unexpected errors
        print(f"💥 An unexpected error occurred: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()