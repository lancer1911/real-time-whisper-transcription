#!/usr/bin/env python3

import speech_recognition as sr
import numpy as np
import time
import pyaudio

def test_microphone_detailed():
    """详细的麦克风测试"""
    print("🔍 详细麦克风诊断开始...\n")
    
    # 1. 列出所有音频设备
    print("1. 可用麦克风设备:")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"   {index}: {name}")
    
    # 2. 测试默认麦克风
    print("\n2. 测试默认麦克风...")
    try:
        r = sr.Recognizer()
        
        # 尝试不同的能量阈值
        for threshold in [4000, 1000, 300, 100, 50]:
            print(f"\n   测试能量阈值: {threshold}")
            r.energy_threshold = threshold
            r.dynamic_energy_threshold = False
            
            with sr.Microphone(sample_rate=16000) as source:
                print(f"   调整环境噪音... (请保持安静)")
                r.adjust_for_ambient_noise(source, duration=1)
                print(f"   调整后的能量阈值: {r.energy_threshold}")
                
                print(f"   录制测试 (请说话，3秒)...")
                try:
                    audio = r.listen(source, timeout=3, phrase_time_limit=3)
                    audio_data = audio.get_raw_data()
                    print(f"   ✅ 录制成功: {len(audio_data)} 字节")
                    
                    # 分析音频数据
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    max_volume = np.max(np.abs(audio_np))
                    avg_volume = np.mean(np.abs(audio_np))
                    print(f"   音量 - 最大: {max_volume:.3f}, 平均: {avg_volume:.3f}")
                    
                    if max_volume > 0.01:
                        print(f"   🎉 阈值 {threshold} 工作正常!")
                        return threshold
                    else:
                        print(f"   ⚠️  音量太小，可能需要更低阈值")
                        
                except sr.WaitTimeoutError:
                    print(f"   ❌ 超时 - 阈值 {threshold} 太高")
                except Exception as e:
                    print(f"   ❌ 错误: {e}")
    
    except Exception as e:
        print(f"❌ 麦克风测试失败: {e}")
        return None
    
    # 3. 使用PyAudio直接测试
    print("\n3. PyAudio直接录音测试...")
    try:
        p = pyaudio.PyAudio()
        
        # 获取默认输入设备信息
        default_device = p.get_default_input_device_info()
        print(f"   默认输入设备: {default_device['name']}")
        print(f"   最大输入通道: {default_device['maxInputChannels']}")
        print(f"   默认采样率: {default_device['defaultSampleRate']}")
        
        # 尝试录音
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        print("   直接录音测试 (2秒, 请说话)...")
        frames = []
        for i in range(0, int(16000 / 1024 * 2)):
            data = stream.read(1024)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # 分析录制的数据
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        max_volume = np.max(np.abs(audio_np))
        avg_volume = np.mean(np.abs(audio_np))
        
        print(f"   ✅ PyAudio录音成功")
        print(f"   音量 - 最大: {max_volume:.3f}, 平均: {avg_volume:.3f}")
        
        if max_volume < 0.001:
            print("   ⚠️  音量极低，可能麦克风未工作")
        elif max_volume < 0.01:
            print("   ⚠️  音量较低，建议使用能量阈值 10-50")
        else:
            print("   ✅ 音量正常")
            
    except Exception as e:
        print(f"   ❌ PyAudio测试失败: {e}")
    
    print("\n🔍 诊断完成")
    return None

if __name__ == "__main__":
    recommended_threshold = test_microphone_detailed()
    if recommended_threshold:
        print(f"\n🎯 推荐的能量阈值: {recommended_threshold}")
        print(f"使用命令:")
        print(f"python transcriber.py --language en --model small --save_audio --output_file transcription.txt --device cpu --energy_threshold {recommended_threshold}")
    else:
        print(f"\n🎯 建议尝试极低能量阈值:")
        print(f"python transcriber.py --language en --model small --save_audio --output_file transcription.txt --device cpu --energy_threshold 10")