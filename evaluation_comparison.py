#!/usr/bin/env python3
"""
Comprehensive TTS Evaluation Script
Compare Fine-tuned vs Pretrained Spark-TTS models
"""

import os
import json
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Text-to-speech evaluation metrics
try:
    from jiwer import wer, cer
    JIWER_AVAILABLE = True
except ImportError:
    print("Installing jiwer for WER/CER calculation...")
    import subprocess
    subprocess.check_call(["pip", "install", "jiwer"])
    from jiwer import wer, cer
    JIWER_AVAILABLE = True

# Speech similarity metrics
try:
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    WAV2VEC_AVAILABLE = True
except ImportError:
    WAV2VEC_AVAILABLE = False

# Audio quality metrics
try:
    import pesq
    PESQ_AVAILABLE = True
except ImportError:
    print("Installing pesq for audio quality...")
    import subprocess
    subprocess.check_call(["pip", "install", "pesq"])
    import pesq
    PESQ_AVAILABLE = True

class TTSEvaluator:
    def __init__(self):
        self.results = {}
        self.texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Hello, this is a test of the Spark TTS model.",
            "The weather today is quite pleasant and sunny.",
            "Artificial intelligence is transforming",
            "voices are becoming indistinguishable",
            "he was a notorious criminal",
            "on the contrary, now this is permanent",
            "hi there how is it going"
        ]
        
    def load_audio_files(self, directory):
        """Load all audio files from directory"""
        audio_files = {}
        for file in sorted(Path(directory).glob("*.wav")):
            try:
                audio, sr = librosa.load(file, sr=16000)
                audio_files[file.name] = audio
            except Exception as e:
                print(f"Error loading {file}: {e}")
        return audio_files
    
    def calculate_audio_metrics(self, audio1, audio2):
        """Calculate audio similarity metrics"""
        # Ensure same length
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]
        
        # Cosine similarity
        cos_sim = np.dot(audio1, audio2) / (np.linalg.norm(audio1) * np.linalg.norm(audio2))
        
        # Spectral similarity
        stft1 = librosa.stft(audio1)
        stft2 = librosa.stft(audio2)
        spectral_sim = np.mean(np.abs(stft1 - stft2))
        
        # Energy ratio
        energy1 = np.sum(audio1**2)
        energy2 = np.sum(audio2**2)
        energy_ratio = min(energy1, energy2) / max(energy1, energy2)
        
        return {
            'cosine_similarity': cos_sim,
            'spectral_distance': spectral_sim,
            'energy_ratio': energy_ratio
        }
    
    def calculate_pesq_score(self, reference, degraded):
        """Calculate PESQ score for audio quality"""
        try:
            # Ensure same length and sample rate
            min_len = min(len(reference), len(degraded))
            ref = reference[:min_len]
            deg = degraded[:min_len]
            
            # PESQ expects 16kHz sample rate
            score = pesq.pesq(16000, ref, deg, 'wb')
            return score
        except Exception as e:
            print(f"PESQ calculation error: {e}")
            return None
    
    def calculate_spectral_centroid(self, audio):
        """Calculate spectral centroid"""
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=16000)[0]
        return np.mean(spectral_centroids)
    
    def calculate_zero_crossing_rate(self, audio):
        """Calculate zero crossing rate"""
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        return np.mean(zcr)
    
    def calculate_mfcc_similarity(self, audio1, audio2):
        """Calculate MFCC similarity"""
        mfcc1 = librosa.feature.mfcc(y=audio1, sr=16000, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=audio2, sr=16000, n_mfcc=13)
        
        # Calculate mean MFCC vectors
        mfcc1_mean = np.mean(mfcc1, axis=1)
        mfcc2_mean = np.mean(mfcc2, axis=1)
        
        # Cosine similarity of MFCC vectors
        mfcc_sim = np.dot(mfcc1_mean, mfcc2_mean) / (np.linalg.norm(mfcc1_mean) * np.linalg.norm(mfcc2_mean))
        return mfcc_sim
    
    def calculate_rhythm_metrics(self, audio):
        """Calculate rhythm and tempo metrics"""
        # Tempo. librosa.beat.beat_track returns a tuple (tempo, beat_frames). tempo is an array.
        # We extract the scalar value from the tempo array.
        tempo, _ = librosa.beat.beat_track(y=audio, sr=16000)
        tempo_value = float(tempo[0]) # Extract the scalar float value
        
        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=16000)[0]
        mean_rolloff = np.mean(rolloff)
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=16000)[0]
        mean_bandwidth = np.mean(bandwidth)
        
        return {
            'tempo': tempo_value, # Now a scalar float, fixing the DataFrame issue
            'spectral_rolloff': mean_rolloff,
            'spectral_bandwidth': mean_bandwidth
        }
    
    def calculate_voice_activity_detection(self, audio):
        """Calculate voice activity metrics"""
        # Simple energy-based VAD
        frame_length = 1024
        hop_length = 512
        
        # Calculate frame energy
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Threshold for voice activity
        threshold = np.mean(energy) * 0.1
        voice_frames = np.sum(energy > threshold)
        total_frames = len(energy)
        
        voice_ratio = voice_frames / total_frames if total_frames > 0 else 0
        
        return {
            'voice_activity_ratio': voice_ratio,
            'mean_energy': np.mean(energy),
            'energy_std': np.std(energy)
        }
    
    def evaluate_models(self):
        """Main evaluation function"""
        print("🎯 Starting Comprehensive TTS Evaluation...")
        print("=" * 60)
        
        # Load audio files
        print("📁 Loading audio files...")
        finetuned_audio = self.load_audio_files("cli_results")
        pretrained_audio = self.load_audio_files("pre_spark_results")
        
        print(f"Fine-tuned model: {len(finetuned_audio)} files")
        print(f"Pretrained model: {len(pretrained_audio)} files")
        
        # Match files by timestamp
        finetuned_files = sorted(finetuned_audio.keys())
        pretrained_files = sorted(pretrained_audio.keys())
        
        results = []
        
        for i, (finetuned_file, pretrained_file) in enumerate(zip(finetuned_files, pretrained_files)):
            if i >= len(self.texts):
                break
                
            text = self.texts[i]
            print(f"\n📝 Evaluating: '{text[:50]}...'")
            
            finetuned_audio_data = finetuned_audio[finetuned_file]
            pretrained_audio_data = pretrained_audio[pretrained_file]
            
            # Basic audio metrics
            audio_metrics = self.calculate_audio_metrics(finetuned_audio_data, pretrained_audio_data)
            
            # PESQ score
            pesq_score = self.calculate_pesq_score(finetuned_audio_data, pretrained_audio_data)
            
            # Spectral features
            finetuned_centroid = self.calculate_spectral_centroid(finetuned_audio_data)
            pretrained_centroid = self.calculate_spectral_centroid(pretrained_audio_data)
            
            finetuned_zcr = self.calculate_zero_crossing_rate(finetuned_audio_data)
            pretrained_zcr = self.calculate_zero_crossing_rate(pretrained_audio_data)
            
            # MFCC similarity
            mfcc_similarity = self.calculate_mfcc_similarity(finetuned_audio_data, pretrained_audio_data)
            
            # Rhythm metrics
            finetuned_rhythm = self.calculate_rhythm_metrics(finetuned_audio_data)
            pretrained_rhythm = self.calculate_rhythm_metrics(pretrained_audio_data)
            
            # Voice activity detection
            finetuned_vad = self.calculate_voice_activity_detection(finetuned_audio_data)
            pretrained_vad = self.calculate_voice_activity_detection(pretrained_audio_data)
            
            # Audio duration
            finetuned_duration = len(finetuned_audio_data) / 16000
            pretrained_duration = len(pretrained_audio_data) / 16000
            
            result = {
                'text': text,
                'finetuned_file': finetuned_file,
                'pretrained_file': pretrained_file,
                'finetuned_duration': finetuned_duration,
                'pretrained_duration': pretrained_duration,
                'duration_difference': abs(finetuned_duration - pretrained_duration),
                'cosine_similarity': audio_metrics['cosine_similarity'],
                'spectral_distance': audio_metrics['spectral_distance'],
                'energy_ratio': audio_metrics['energy_ratio'],
                'pesq_score': pesq_score,
                'mfcc_similarity': mfcc_similarity,
                'finetuned_spectral_centroid': finetuned_centroid,
                'pretrained_spectral_centroid': pretrained_centroid,
                'spectral_centroid_diff': abs(finetuned_centroid - pretrained_centroid),
                'finetuned_zcr': finetuned_zcr,
                'pretrained_zcr': pretrained_zcr,
                'zcr_difference': abs(finetuned_zcr - pretrained_zcr),
                'finetuned_tempo': finetuned_rhythm['tempo'],
                'pretrained_tempo': pretrained_rhythm['tempo'],
                'tempo_difference': abs(finetuned_rhythm['tempo'] - pretrained_rhythm['tempo']),
                'finetuned_voice_activity': finetuned_vad['voice_activity_ratio'],
                'pretrained_voice_activity': pretrained_vad['voice_activity_ratio'],
                'voice_activity_diff': abs(finetuned_vad['voice_activity_ratio'] - pretrained_vad['voice_activity_ratio'])
            }
            
            results.append(result)
            
            print(f"  ✅ Cosine Similarity: {audio_metrics['cosine_similarity']:.4f}")
            print(f"  ✅ PESQ Score: {pesq_score:.4f}" if pesq_score else "  ❌ PESQ: N/A")
            print(f"  ✅ MFCC Similarity: {mfcc_similarity:.4f}")
            print(f"  ✅ Duration: {finetuned_duration:.2f}s vs {pretrained_duration:.2f}s")
        
        return results
    
    def generate_summary_report(self, results):
        """Generate comprehensive summary report"""
        df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE EVALUATION SUMMARY")
        print("="*80)
        
        # Overall statistics
        print(f"\n🎯 Total Samples Evaluated: {len(results)}")
        print(f"📝 Average Cosine Similarity: {df['cosine_similarity'].mean():.4f} ± {df['cosine_similarity'].std():.4f}")
        print(f"🎵 Average MFCC Similarity: {df['mfcc_similarity'].mean():.4f} ± {df['mfcc_similarity'].std():.4f}")
        
        if df['pesq_score'].notna().any():
            print(f"🔊 Average PESQ Score: {df['pesq_score'].mean():.4f} ± {df['pesq_score'].std():.4f}")
        
        print(f"⏱️  Average Duration Difference: {df['duration_difference'].mean():.2f}s ± {df['duration_difference'].std():.2f}s")
        print(f"🎼 Average Spectral Centroid Difference: {df['spectral_centroid_diff'].mean():.2f} ± {df['spectral_centroid_diff'].std():.2f}")
        print(f"🎵 Average Tempo Difference: {float(df['tempo_difference'].mean()):.2f} ± {float(df['tempo_difference'].std()):.2f}")
        
        # Best and worst performing samples
        best_similarity = df.loc[df['cosine_similarity'].idxmax()]
        worst_similarity = df.loc[df['cosine_similarity'].idxmin()]
        
        print(f"\n🏆 Best Similarity: '{best_similarity['text'][:50]}...' ({best_similarity['cosine_similarity']:.4f})")
        print(f"📉 Worst Similarity: '{worst_similarity['text'][:50]}...' ({worst_similarity['cosine_similarity']:.4f})")
        
        # Model comparison
        print(f"\n📈 MODEL COMPARISON:")
        print(f"Fine-tuned avg duration: {df['finetuned_duration'].mean():.2f}s")
        print(f"Pretrained avg duration: {df['pretrained_duration'].mean():.2f}s")
        print(f"Fine-tuned avg spectral centroid: {df['finetuned_spectral_centroid'].mean():.2f}")
        print(f"Pretrained avg spectral centroid: {df['pretrained_spectral_centroid'].mean():.2f}")
        
        return df
    
    def save_results(self, results, df):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results (convert numpy types to Python types)
        results_serializable = []
        for result in results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, (np.float32, np.float64)):
                    serializable_result[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    serializable_result[key] = int(value)
                elif isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                else:
                    serializable_result[key] = value
            results_serializable.append(serializable_result)
        
        results_file = f"evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        # Save CSV
        csv_file = f"evaluation_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        # Save summary report
        report_file = f"evaluation_summary_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write("TTS Model Evaluation Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Samples: {len(results)}\n\n")
            f.write("Key Metrics:\n")
            f.write(f"- Average Cosine Similarity: {df['cosine_similarity'].mean():.4f}\n")
            f.write(f"- Average MFCC Similarity: {df['mfcc_similarity'].mean():.4f}\n")
            if df['pesq_score'].notna().any():
                f.write(f"- Average PESQ Score: {df['pesq_score'].mean():.4f}\n")
            f.write(f"- Average Duration Difference: {df['duration_difference'].mean():.2f}s\n")
        
        print(f"\n💾 Results saved to:")
        print(f"  📄 Detailed JSON: {results_file}")
        print(f"  📊 CSV Data: {csv_file}")
        print(f"  📋 Summary Report: {report_file}")
        
        return results_file, csv_file, report_file

def main():
    print("🚀 Starting TTS Model Evaluation")
    print("Comparing Fine-tuned vs Pretrained Spark-TTS models")
    print("="*60)
    
    evaluator = TTSEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate_models()
    
    # Generate summary
    df = evaluator.generate_summary_report(results)
    
    # Save results
    evaluator.save_results(results, df)
    
    print("\n🎉 Evaluation Complete!")
    print("📊 Check the generated files for detailed analysis")

if __name__ == "__main__":
    main()
