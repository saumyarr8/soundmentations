import soundfile as sf
import soundmentations as S
def main():
    # Load audio (automatically mono)
    audio, sr = S.load_audio("examples/data/sample-9s.wav", sample_rate=44100)
    
    # Define your augmentation pipeline with transforms
    pipeline = S.Compose([
        S.Trim(start_time=0.1, end_time=2.0),  # Trim first 0.1s and last 2.0s
    ])
    
    # Apply the pipeline
    augmented_audio = pipeline(audio, sample_rate=sr)
    
    # Save output
    sf.write("examples/data/output-sample-9s.wav", augmented_audio, sr)
    print("Augmented audio saved!")

if __name__ == "__main__":
    main()