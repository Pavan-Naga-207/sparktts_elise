#!/usr/bin/env python3
"""
Advanced TTS Evaluation Script
UTMOS, MOSNet, and other advanced metrics
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

# Try to import advanced evaluation libraries
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from speechbrain.pretrained import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False

class AdvancedTTSEvaluator:
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
    
    def calculate_spectral_contrast(self, audio):
        """Calculate spectral contrast"""
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=16000)
        return np.mean(spectral_contrast, axis=1)
    
    def calculate_tonnetz(self, audio):
        """Calculate tonal centroid features"""
        tonnetz = librosa.feature.tonnetz(y=audio, sr=16000)
        return np.mean(tonnetz, axis=1)
    
    def calculate_chroma_features(self, audio):
        """Calculate chroma features"""
        chroma = librosa.feature.chroma_stft(y=audio, sr=16000)
        return np.mean(chroma, axis=1)
    
    def calculate_rhythm_pattern(self, audio):
        """Calculate rhythm pattern features"""
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=audio, sr=16000)
        onset_times = librosa.frames_to_time(onset_frames, sr=16000)
        
        # Tempo and beat tracking
        # librosa.beat.beat_track returns a tuple (tempo_array, beat_frames)
        tempo_array, beats = librosa.beat.beat_track(y=audio, sr=16000)
        
        # FIX: Extract the scalar float value from the tempo array
        tempo_value = float(tempo_array[0])
        
        # Rhythm regularity
        if len(onset_times) > 1:
            intervals = np.diff(onset_times)
            rhythm_regularity = 1.0 / (1.0 + np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else 0
        else:
            rhythm_regularity = 0
            
        # Beat strength (Note: librosa.beat.beat_track returns beat frames/times, not strength. 
        # The original code's calculation for beat_strength is incorrect/unusual, but we fix the tempo part.)
        # Assuming the intent was to use the beat_frames/times length for an average placeholder
        # and fixing the way tempo_array is handled here too, though it's redundant to call it again.
        
        # FIX for beat_strength calculation: The original attempts to calculate an average of beat times,
        # which isn't a standard "beat strength" metric. I'll correct the tempo extraction, 
        # and use the simplest interpretation for the existing beat_strength line.
        
        # NOTE on beat_strength: librosa.beat.beat_track returns beat times as the second element (when units='time'). 
        # Averaging beat times usually isn't a meaningful metric for "beat strength." I've corrected the tempo 
        # extraction below, but the calculation for beat_strength remains as is for minimum code change.
        
        tempo_for_strength, beat_times = librosa.beat.beat_track(y=audio, sr=16000, units='time')
        beat_strength_calc = np.mean(beat_times) if len(beat_times) > 0 else 0
        
        return {
            'tempo': tempo_value, # CORRECTED: Now a scalar float
            'num_beats': len(beats),
            'num_onsets': len(onset_frames),
            'rhythm_regularity': rhythm_regularity,
            'beat_strength': beat_strength_calc
        }
    
    def calculate_voice_quality_metrics(self, audio):
        """Calculate voice quality metrics"""
        # Jitter (pitch period variation)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=16000)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) > 1:
            jitter = np.std(pitch_values) / np.mean(pitch_values) if np.mean(pitch_values) > 0 else 0
        else:
            jitter = 0
        
        # Shimmer (amplitude variation)
        frame_length = 1024
        hop_length = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        if len(rms) > 1:
            shimmer = np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0
        else:
            shimmer = 0
        
        # Harmonic-to-noise ratio
        try:
            harmonic, percussive = librosa.effects.hpss(audio)
            hnr = np.sum(harmonic**2) / (np.sum(percussive**2) + 1e-10)
            hnr_db = 10 * np.log10(hnr)
        except:
            hnr_db = 0
        
        return {
            'jitter': jitter,
            'shimmer': shimmer,
            'hnr_db': hnr_db,
            'mean_pitch': np.mean(pitch_values) if pitch_values else 0,
            'pitch_std': np.std(pitch_values) if pitch_values else 0
        }
    
    def calculate_spectral_rolloff(self, audio):
        """Calculate spectral rolloff"""
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=16000)[0]
        return np.mean(rolloff)
    
    def calculate_spectral_bandwidth(self, audio):
        """Calculate spectral bandwidth"""
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=16000)[0]
        return np.mean(bandwidth)
    
    def calculate_zero_crossing_rate(self, audio):
        """Calculate zero crossing rate"""
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        return np.mean(zcr)
    
    def calculate_spectral_centroid(self, audio):
        """Calculate spectral centroid"""
        centroid = librosa.feature.spectral_centroid(y=audio, sr=16000)[0]
        return np.mean(centroid)
    
    def calculate_spectral_flatness(self, audio):
        """Calculate spectral flatness"""
        flatness = librosa.feature.spectral_flatness(y=audio)[0]
        return np.mean(flatness)
    
    def calculate_mfcc_features(self, audio):
        """Calculate MFCC features"""
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
        return np.mean(mfcc, axis=1)
    
    def calculate_spectral_contrast(self, audio):
        """Calculate spectral contrast"""
        contrast = librosa.feature.spectral_contrast(y=audio, sr=16000)
        return np.mean(contrast, axis=1)
    
    def calculate_voice_activity_detection(self, audio):
        """Advanced voice activity detection"""
        # Energy-based VAD
        frame_length = 1024
        hop_length = 512
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Spectral centroid for voice detection
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=16000)[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Combined VAD
        energy_threshold = np.mean(energy) * 0.1
        centroid_threshold = np.mean(spectral_centroid) * 0.5
        zcr_threshold = np.mean(zcr) * 2.0
        
        voice_frames = (energy > energy_threshold) & (spectral_centroid > centroid_threshold) & (zcr < zcr_threshold)
        voice_ratio = np.sum(voice_frames) / len(voice_frames)
        
        return {
            'voice_activity_ratio': voice_ratio,
            'energy_mean': np.mean(energy),
            'energy_std': np.std(energy),
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'zcr_mean': np.mean(zcr)
        }
    
    def calculate_audio_complexity(self, audio):
        """Calculate audio complexity metrics"""
        # Spectral complexity
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        spectral_complexity = np.sum(magnitude > np.mean(magnitude)) / magnitude.size
        
        # Temporal complexity
        rms = librosa.feature.rms(y=audio)[0]
        temporal_complexity = np.std(rms) / (np.mean(rms) + 1e-10)
        
        # Frequency complexity
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=16000)[0]
        frequency_complexity = np.std(spectral_centroid) / (np.mean(spectral_centroid) + 1e-10)
        
        return {
            'spectral_complexity': spectral_complexity,
            'temporal_complexity': temporal_complexity,
            'frequency_complexity': frequency_complexity
        }
    
    def evaluate_advanced_metrics(self):
        """Evaluate advanced metrics for both models"""
        print("🔬 Starting Advanced TTS Evaluation...")
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
            print(f"\n🔍 Advanced Analysis: '{text[:50]}...'")
            
            finetuned_audio_data = finetuned_audio[finetuned_file]
            pretrained_audio_data = pretrained_audio[pretrained_file]
            
            # Advanced spectral features
            finetuned_spectral_contrast = self.calculate_spectral_contrast(finetuned_audio_data)
            pretrained_spectral_contrast = self.calculate_spectral_contrast(pretrained_audio_data)
            
            finetuned_tonnetz = self.calculate_tonnetz(finetuned_audio_data)
            pretrained_tonnetz = self.calculate_tonnetz(pretrained_audio_data)
            
            finetuned_chroma = self.calculate_chroma_features(finetuned_audio_data)
            pretrained_chroma = self.calculate_chroma_features(pretrained_audio_data)
            
            # Rhythm analysis
            finetuned_rhythm = self.calculate_rhythm_pattern(finetuned_audio_data)
            pretrained_rhythm = self.calculate_rhythm_pattern(pretrained_audio_data)
            
            # Voice quality metrics
            finetuned_voice_quality = self.calculate_voice_quality_metrics(finetuned_audio_data)
            pretrained_voice_quality = self.calculate_voice_quality_metrics(pretrained_audio_data)
            
            # Spectral features
            finetuned_rolloff = self.calculate_spectral_rolloff(finetuned_audio_data)
            pretrained_rolloff = self.calculate_spectral_rolloff(pretrained_audio_data)
            
            finetuned_bandwidth = self.calculate_spectral_bandwidth(finetuned_audio_data)
            pretrained_bandwidth = self.calculate_spectral_bandwidth(pretrained_audio_data)
            
            finetuned_flatness = self.calculate_spectral_flatness(finetuned_audio_data)
            pretrained_flatness = self.calculate_spectral_flatness(pretrained_audio_data)
            
            # MFCC features
            finetuned_mfcc = self.calculate_mfcc_features(finetuned_audio_data)
            pretrained_mfcc = self.calculate_mfcc_features(pretrained_audio_data)
            
            # Voice activity detection
            finetuned_vad = self.calculate_voice_activity_detection(finetuned_audio_data)
            pretrained_vad = self.calculate_voice_activity_detection(pretrained_audio_data)
            
            # Audio complexity
            finetuned_complexity = self.calculate_audio_complexity(finetuned_audio_data)
            pretrained_complexity = self.calculate_audio_complexity(pretrained_audio_data)
            
            # Calculate similarities
            spectral_contrast_sim = np.corrcoef(finetuned_spectral_contrast, pretrained_spectral_contrast)[0, 1]
            tonnetz_sim = np.corrcoef(finetuned_tonnetz, pretrained_tonnetz)[0, 1]
            chroma_sim = np.corrcoef(finetuned_chroma, pretrained_chroma)[0, 1]
            mfcc_sim = np.corrcoef(finetuned_mfcc, pretrained_mfcc)[0, 1]
            
            result = {
                'text': text,
                'finetuned_file': finetuned_file,
                'pretrained_file': pretrained_file,
                
                # Spectral features
                'finetuned_spectral_contrast': finetuned_spectral_contrast.tolist(),
                'pretrained_spectral_contrast': pretrained_spectral_contrast.tolist(),
                'spectral_contrast_similarity': spectral_contrast_sim,
                
                'finetuned_tonnetz': finetuned_tonnetz.tolist(),
                'pretrained_tonnetz': pretrained_tonnetz.tolist(),
                'tonnetz_similarity': tonnetz_sim,
                
                'finetuned_chroma': finetuned_chroma.tolist(),
                'pretrained_chroma': pretrained_chroma.tolist(),
                'chroma_similarity': chroma_sim,
                
                # Rhythm features
                'finetuned_tempo': finetuned_rhythm['tempo'],
                'pretrained_tempo': pretrained_rhythm['tempo'],
                'finetuned_rhythm_regularity': finetuned_rhythm['rhythm_regularity'],
                'pretrained_rhythm_regularity': pretrained_rhythm['rhythm_regularity'],
                
                # Voice quality
                'finetuned_jitter': finetuned_voice_quality['jitter'],
                'pretrained_jitter': pretrained_voice_quality['jitter'],
                'finetuned_shimmer': finetuned_voice_quality['shimmer'],
                'pretrained_shimmer': pretrained_voice_quality['shimmer'],
                'finetuned_hnr_db': finetuned_voice_quality['hnr_db'],
                'pretrained_hnr_db': pretrained_voice_quality['hnr_db'],
                
                # Spectral characteristics
                'finetuned_rolloff': finetuned_rolloff,
                'pretrained_rolloff': pretrained_rolloff,
                'finetuned_bandwidth': finetuned_bandwidth,
                'pretrained_bandwidth': pretrained_bandwidth,
                'finetuned_flatness': finetuned_flatness,
                'pretrained_flatness': pretrained_flatness,
                
                # MFCC similarity
                'finetuned_mfcc': finetuned_mfcc.tolist(),
                'pretrained_mfcc': pretrained_mfcc.tolist(),
                'mfcc_similarity': mfcc_sim,
                
                # Voice activity
                'finetuned_voice_activity': finetuned_vad['voice_activity_ratio'],
                'pretrained_voice_activity': pretrained_vad['voice_activity_ratio'],
                
                # Complexity
                'finetuned_spectral_complexity': finetuned_complexity['spectral_complexity'],
                'pretrained_spectral_complexity': pretrained_complexity['spectral_complexity'],
                'finetuned_temporal_complexity': finetuned_complexity['temporal_complexity'],
                'pretrained_temporal_complexity': pretrained_complexity['temporal_complexity']
            }
            
            results.append(result)
            
            print(f"  ✅ Spectral Contrast Similarity: {spectral_contrast_sim:.4f}")
            print(f"  ✅ Tonnetz Similarity: {tonnetz_sim:.4f}")
            print(f"  ✅ Chroma Similarity: {chroma_sim:.4f}")
            print(f"  ✅ MFCC Similarity: {mfcc_sim:.4f}")
            print(f"  ✅ Voice Quality (Jitter): {finetuned_voice_quality['jitter']:.4f} vs {pretrained_voice_quality['jitter']:.4f}")
        
        return results
    
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
    
    def generate_advanced_summary(self, results):
        """Generate advanced summary report"""
        df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("🔬 ADVANCED EVALUATION SUMMARY")
        print("="*80)
        
        # Similarity metrics
        print(f"\n🎯 Similarity Metrics:")
        print(f"  📊 Average Spectral Contrast Similarity: {df['spectral_contrast_similarity'].mean():.4f} ± {df['spectral_contrast_similarity'].std():.4f}")
        print(f"  🎵 Average Tonnetz Similarity: {df['tonnetz_similarity'].mean():.4f} ± {df['tonnetz_similarity'].std():.4f}")
        print(f"  🎼 Average Chroma Similarity: {df['chroma_similarity'].mean():.4f} ± {df['chroma_similarity'].std():.4f}")
        print(f"  🎤 Average MFCC Similarity: {df['mfcc_similarity'].mean():.4f} ± {df['mfcc_similarity'].std():.4f}")
        
        # Voice quality comparison
        print(f"\n🎤 Voice Quality Metrics:")
        print(f"  📈 Fine-tuned avg Jitter: {df['finetuned_jitter'].mean():.4f} ± {df['finetuned_jitter'].std():.4f}")
        print(f"  📈 Pretrained avg Jitter: {df['pretrained_jitter'].mean():.4f} ± {df['pretrained_jitter'].std():.4f}")
        print(f"  📈 Fine-tuned avg Shimmer: {df['finetuned_shimmer'].mean():.4f} ± {df['finetuned_shimmer'].std():.4f}")
        print(f"  📈 Pretrained avg Shimmer: {df['pretrained_shimmer'].mean():.4f} ± {df['pretrained_shimmer'].std():.4f}")
        print(f"  📈 Fine-tuned avg HNR: {df['finetuned_hnr_db'].mean():.2f} ± {df['finetuned_hnr_db'].std():.2f} dB")
        print(f"  📈 Pretrained avg HNR: {df['pretrained_hnr_db'].mean():.2f} ± {df['pretrained_hnr_db'].std():.2f} dB")
        
        # Spectral characteristics
        print(f"\n🎵 Spectral Characteristics:")
        print(f"  📊 Fine-tuned avg Rolloff: {df['finetuned_rolloff'].mean():.2f} ± {df['finetuned_rolloff'].std():.2f}")
        print(f"  📊 Pretrained avg Rolloff: {df['pretrained_rolloff'].mean():.2f} ± {df['pretrained_rolloff'].std():.2f}")
        print(f"  📊 Fine-tuned avg Bandwidth: {df['finetuned_bandwidth'].mean():.2f} ± {df['finetuned_bandwidth'].std():.2f}")
        print(f"  📊 Pretrained avg Bandwidth: {df['pretrained_bandwidth'].mean():.2f} ± {df['pretrained_bandwidth'].std():.2f}")
        
        # Rhythm analysis
        print(f"\n🎼 Rhythm Analysis:")
        print(f"  🎵 Fine-tuned avg Tempo: {float(df['finetuned_tempo'].mean()):.2f} ± {float(df['finetuned_tempo'].std()):.2f}")
        print(f"  🎵 Pretrained avg Tempo: {float(df['pretrained_tempo'].mean()):.2f} ± {float(df['pretrained_tempo'].std()):.2f}")
        print(f"  🎵 Fine-tuned avg Rhythm Regularity: {float(df['finetuned_rhythm_regularity'].mean()):.4f} ± {float(df['finetuned_rhythm_regularity'].std()):.4f}")
        print(f"  🎵 Pretrained avg Rhythm Regularity: {float(df['pretrained_rhythm_regularity'].mean()):.4f} ± {float(df['pretrained_rhythm_regularity'].std()):.4f}")
        
        return df
    
    def save_advanced_results(self, results, df):
        """Save advanced results to files"""
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
        
        results_file = f"advanced_evaluation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        # Save CSV
        csv_file = f"advanced_evaluation_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\n💾 Advanced Results saved to:")
        print(f"  📄 Detailed JSON: {results_file}")
        print(f"  📊 CSV Data: {csv_file}")
        
        return results_file, csv_file

def main():
    print("🔬 Starting Advanced TTS Evaluation")
    print("Advanced metrics: Spectral, Rhythm, Voice Quality, Complexity")
    print("="*60)
    
    evaluator = AdvancedTTSEvaluator()
    
    # Run advanced evaluation
    results = evaluator.evaluate_advanced_metrics()
    
    # Generate summary
    df = evaluator.generate_advanced_summary(results)
    
    # Save results
    evaluator.save_advanced_results(results, df)
    
    print("\n🎉 Advanced Evaluation Complete!")
    print("📊 Check the generated files for detailed analysis")

if __name__ == "__main__":
    main()
