import soundfile as sf
import soundmentations as S

def main():
    # Load audio (automatically mono)
    audio, sr = S.load_audio("examples/data/sample-9s.wav", sample_rate=44100)
    print(f"Original audio: {len(audio)/sr:.2f}s duration")
    
    # Define your augmentation pipeline with multiple transforms
    pipeline = S.Compose([
        S.RandomTrim(duration=(1.5, 3.0), p=0.8),        # Random crop 1.5-3s with 80% chance
        S.Gain(gain=3.0, p=0.7),                         # Fixed +3dB gain with 70% chance
    ])

    # Apply the pipeline multiple times to see variations
    c=0
    for i in range(100):
        augmented_audio = pipeline(audio, sample_rate=sr)
        output_file = f"examples/data/output-sample-{i+1}.wav"
        sf.write(output_file, augmented_audio, sr)
        print(f"Augmented audio {i+1} saved: {len(augmented_audio)/sr:.2f}s duration")
        if len(augmented_audio)/sr > 3.0:
            c += 1
    print(f"Number of augmented audios longer than 3 seconds: {c}")
    
    print("All augmented audio files saved!")

if __name__ == "__main__":
    main()