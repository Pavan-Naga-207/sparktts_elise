import os
import json
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from jiwer import wer, cer
except ImportError:
    print("WARNING: jiwer not installed. Skipping WER/CER calculation.")
    wer = cer = lambda *args: None

try:
    import pesq
except ImportError:
    print("WARNING: pesq not installed. Skipping PESQ calculation.")
    pesq = None

try:
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    ASR_PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    ASR_MODEL = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    ASR_AVAILABLE = True
except ImportError:
    print("WARNING: Torch/Transformers not installed. Skipping ASR transcription (WER/CER).")
    ASR_AVAILABLE = False
except Exception as e:
    print(f"WARNING: ASR Model Loading Failed ({e}). Skipping ASR.")
    ASR_AVAILABLE = False

class TTSEvaluator:
    def __init__(self, metadata_path="metadata.csv", gt_dir="gt_audios", ft_dir="cli_results", pt_dir="pre_spark_results"):
        self.metadata_path = metadata_path
        self.gt_dir = Path(gt_dir)
        self.ft_dir = Path(ft_dir)
        self.pt_dir = Path(pt_dir)
        self.results = []
        self.metadata = self.load_metadata()
        
    def load_metadata(self):
        """Load and parse LJSpeech-style metadata file."""
        print(f"📝 Loading metadata from {self.metadata_path}...")
        try:
            df = pd.read_csv(
                self.metadata_path, 
                sep='|', 
                header=None, 
                names=['segment_id', 'gt_text_1', 'gt_text_2']
            )
            return df[['segment_id', 'gt_text_1']].set_index('segment_id').to_dict()['gt_text_1']
        except FileNotFoundError:
            print(f"FATAL ERROR: Metadata file not found at {self.metadata_path}.")
            return {}
        
    def load_audio_files(self, directory):
        """Load all audio files from directory into a dictionary keyed by segment_id"""
        audio_files = {}
        for file in sorted(Path(directory).glob("*.wav")):
            segment_id = file.stem
            try:
                audio, sr = librosa.load(file, sr=16000)
                audio_files[segment_id] = audio
            except Exception as e:
                print(f"Error loading {file.name}: {e}")
        return audio_files

    def transcribe_audio(self, audio):
        """Use Wav2Vec2 to transcribe audio for WER/CER calculation."""
        if not ASR_AVAILABLE:
            return ""
        try:
            input_values = ASR_PROCESSOR(audio, return_tensors="pt", sampling_rate=16000).input_values
            logits = ASR_MODEL(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = ASR_PROCESSOR.batch_decode(predicted_ids)[0].lower()
            return transcription
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
            
    def calculate_pesq_score(self, reference, degraded):
        """Calculate PESQ score for audio quality against a ground truth reference."""
        if not pesq: return np.nan
        try:
            min_len = min(len(reference), len(degraded))
            score = pesq.pesq(16000, reference[:min_len], degraded[:min_len], 'wb')
            return score
        except Exception:
            return np.nan
    
    def calculate_mfcc_similarity(self, audio1, audio2):
        """Calculate MFCC similarity (Fidelity of Timbre)."""
        mfcc1 = librosa.feature.mfcc(y=audio1, sr=16000, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=audio2, sr=16000, n_mfcc=13)
        
        mfcc1_mean = np.mean(mfcc1, axis=1)
        mfcc2_mean = np.mean(mfcc2, axis=1)

        mfcc_sim = np.dot(mfcc1_mean, mfcc2_mean) / (np.linalg.norm(mfcc1_mean) * np.linalg.norm(mfcc2_mean))
        return mfcc_sim
    
    def calculate_spectral_centroid(self, audio):
        """Calculate spectral centroid (Proxy for tone/brightness)."""
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=16000)[0]
        return np.mean(spectral_centroids)
    
    def evaluate_models(self):
        """Main evaluation function using ground truth."""
        print("🎯 Starting Comprehensive TTS Evaluation with Ground Truth...")

        gt_audio = self.load_audio_files(self.gt_dir)
        finetuned_audio = self.load_audio_files(self.ft_dir)
        pretrained_audio = self.load_audio_files(self.pt_dir)
        
        common_segments = set(self.metadata.keys()) & set(gt_audio.keys()) & set(finetuned_audio.keys()) & set(pretrained_audio.keys())
        
        if not common_segments:
            print("FATAL ERROR: No common segments found across metadata, GT, FT, and PT audio folders.")
            return []

        results = []
        
        for i, segment_id in enumerate(sorted(list(common_segments))):
            if i >= 100:
                break 
                
            gt_text = self.metadata[segment_id]
            gt_audio_data = gt_audio[segment_id]
            ft_audio_data = finetuned_audio[segment_id]
            pt_audio_data = pretrained_audio[segment_id]
            
            # 1. Fidelity and Quality Metrics (against GT)
            ft_pesq = self.calculate_pesq_score(gt_audio_data, ft_audio_data)
            pt_pesq = self.calculate_pesq_score(gt_audio_data, pt_audio_data)
            
            ft_mfcc_sim = self.calculate_mfcc_similarity(gt_audio_data, ft_audio_data)
            pt_mfcc_sim = self.calculate_mfcc_similarity(gt_audio_data, pt_audio_data)
            
            # 2. Intelligibility Metrics (against GT Text)
            ft_transcription = self.transcribe_audio(ft_audio_data).upper()
            pt_transcription = self.transcribe_audio(pt_audio_data).upper()
            
            gt_text_upper = gt_text.upper() # WER/CER requires case consistency
            
            ft_wer = wer(gt_text_upper, ft_transcription) if wer else np.nan
            pt_wer = wer(gt_text_upper, pt_transcription) if wer else np.nan
            ft_cer = cer(gt_text_upper, ft_transcription) if cer else np.nan
            pt_cer = cer(gt_text_upper, pt_transcription) if cer else np.nan
            
            # 3. Prosody/Rhythm Metrics (against GT)
            gt_duration = len(gt_audio_data) / 16000
            ft_duration = len(ft_audio_data) / 16000
            pt_duration = len(pt_audio_data) / 16000
            
            gt_centroid = self.calculate_spectral_centroid(gt_audio_data)
            ft_centroid = self.calculate_spectral_centroid(ft_audio_data)
            pt_centroid = self.calculate_spectral_centroid(pt_audio_data)
            
            result = {
                'segment_id': segment_id,
                'gt_text': gt_text,
                'gt_duration': gt_duration,
                
                # Fine-Tuned Model Metrics
                'ft_pesq': ft_pesq,
                'ft_wer': ft_wer,
                'ft_cer': ft_cer,
                'ft_mfcc_sim_gt': ft_mfcc_sim,
                'ft_duration': ft_duration,
                'ft_duration_diff_gt': abs(ft_duration - gt_duration),
                'ft_centroid_diff_gt': abs(ft_centroid - gt_centroid),
                
                # Pre-Trained Model Metrics
                'pt_pesq': pt_pesq,
                'pt_wer': pt_wer,
                'pt_cer': pt_cer,
                'pt_mfcc_sim_gt': pt_mfcc_sim,
                'pt_duration': pt_duration,
                'pt_duration_diff_gt': abs(pt_duration - gt_duration),
                'pt_centroid_diff_gt': abs(pt_centroid - gt_centroid),
            }
            
            results.append(result)

        return results
    
    def generate_summary_report(self, results):
        """Generate comprehensive summary report"""
        df = pd.DataFrame(results)
        
        # Calculate Model Advantage metrics (FT - PT)
        df['pesq_advantage'] = df['ft_pesq'] - df['pt_pesq']
        df['wer_advantage'] = df['pt_wer'] - df['ft_wer'] # Lower WER is better
        df['mfcc_advantage'] = df['ft_mfcc_sim_gt'] - df['pt_mfcc_sim_gt']
        df['duration_diff_advantage'] = df['pt_duration_diff_gt'] - df['ft_duration_diff_gt'] # Lower duration diff is better
        df['centroid_diff_advantage'] = df['pt_centroid_diff_gt'] - df['ft_centroid_diff_gt'] # Lower centroid diff is better
        
        
        # Calculate summary statistics
        summary = {
            # Absolute Quality and Intelligibility (Higher is Better)
            'FT_PESQ_Mean': df['ft_pesq'].mean(),
            'FT_PESQ_Std': df['ft_pesq'].std(),
            'PT_PESQ_Mean': df['pt_pesq'].mean(),
            'PT_PESQ_Std': df['pt_pesq'].std(),

            # Intelligibility (Lower is Better)
            'FT_WER_Mean': df['ft_wer'].mean(),
            'FT_WER_Std': df['ft_wer'].std(),
            'PT_WER_Mean': df['pt_wer'].mean(),
            'PT_WER_Std': df['pt_wer'].std(),
            
            # Fidelity (Higher is Better)
            'FT_MFCC_Sim_Mean': df['ft_mfcc_sim_gt'].mean(),
            'FT_MFCC_Sim_Std': df['ft_mfcc_sim_gt'].std(),
            'PT_MFCC_Sim_Mean': df['pt_mfcc_sim_gt'].mean(),
            'PT_MFCC_Sim_Std': df['pt_mfcc_sim_gt'].std(),

            # Prosody Fidelity (Lower Difference is Better)
            'FT_Duration_Diff_Mean': df['ft_duration_diff_gt'].mean(),
            'FT_Duration_Diff_Std': df['ft_duration_diff_gt'].std(),
            'PT_Duration_Diff_Mean': df['pt_duration_diff_gt'].mean(),
            'PT_Duration_Diff_Std': df['pt_duration_diff_gt'].std(),
            
            # Tone Fidelity (Lower Difference is Better)
            'FT_Centroid_Diff_Mean': df['ft_centroid_diff_gt'].mean(),
            'FT_Centroid_Diff_Std': df['ft_centroid_diff_gt'].std(),
            'PT_Centroid_Diff_Mean': df['pt_centroid_diff_gt'].mean(),
            'PT_Centroid_Diff_Std': df['pt_centroid_diff_gt'].std(),
            
            # Model Advantage (FT - PT)
            'PESQ_Advantage_Mean': df['pesq_advantage'].mean(),
            'WER_Advantage_Mean': df['wer_advantage'].mean(),
            'MFCC_Advantage_Mean': df['mfcc_advantage'].mean(),
            'Duration_Diff_Advantage_Mean': df['duration_diff_advantage'].mean(),
            'Centroid_Diff_Advantage_Mean': df['centroid_diff_advantage'].mean(),
        }

        # Convert to DataFrame for easy printing/saving
        summary_df = pd.DataFrame.from_dict(summary, orient='index', columns=['Value'])
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE EVALUATION SUMMARY (Relative to Ground Truth)")
        print("="*80)
        print(summary_df.to_string(float_format="%.4f"))
        print("\nNote: FT_PESQ/MFCC Advantage > 0 and FT_WER/Duration/Centroid Advantage < 0 is expected for a superior Fine-Tuned Model.")
        
        return df, summary_df
    
    def save_results(self, results, df, summary_df):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed CSV
        csv_file = f"evaluation_results_full_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        # Save summary CSV
        summary_csv_file = f"evaluation_summary_{timestamp}.csv"
        summary_df.to_csv(summary_csv_file, header=False)
        
        print(f"\n💾 Results saved to:")
        print(f"  📊 Detailed CSV: {csv_file}")
        print(f"  📋 Summary CSV: {summary_csv_file}")
        
        return csv_file, summary_csv_file

def main():

    metadata_file = "text.csv"
    gt_audio_dir = "wav_data"
    ft_audio_dir = "cli_results_1"
    pt_audio_dir = "pre_spark_results_1"
    
    if not Path(metadata_file).exists():
        with open(metadata_file, 'w') as f:
            f.write("LJ001-0001|The quick brown fox jumps over the lazy dog.|The quick brown fox jumps over the lazy dog.\n")
            f.write("LJ001-0003|Hello, this is a test of the Spark TTS model.|Hello, this is a test of the Spark TTS model.\n")
            f.write("LJ001-0004|The weather today is quite pleasant and sunny.|The weather today is quite pleasant and sunny.\n")
            
    for d in [gt_audio_dir, ft_audio_dir, pt_audio_dir]:
        Path(d).mkdir(exist_ok=True)
    
    evaluator = TTSEvaluator(metadata_path=metadata_file, gt_dir=gt_audio_dir, ft_dir=ft_audio_dir, pt_dir=pt_audio_dir)
    
    if not evaluator.metadata:
        print("\nExecution aborted due to missing metadata file. Please ensure 'metadata.csv' exists and run again.")
        return

    # Run evaluation
    results = evaluator.evaluate_models()
    
    if results:
        # Generate summary
        df, summary_df = evaluator.generate_summary_report(results)
        
        # Save results
        evaluator.save_results(results, df, summary_df)
    else:
        print("\nEvaluation produced no results. Check file paths and content.")

if __name__ == "__main__":
    main()