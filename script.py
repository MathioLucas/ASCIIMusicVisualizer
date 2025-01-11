import numpy as np
import librosa
import time
import os
from colorama import init, Fore, Style
init()

class MusicVisualizer:
    def __init__(self, audio_file):
        # Load the audio 
        self.y, self.sr = librosa.load(audio_file)
        # Generate spectrogram
        self.spec = librosa.feature.melspectrogram(y=self.y, sr=self.sr)
        # Convert to log scale
        self.spec_db = librosa.power_to_db(self.spec, ref=np.max)
        self.frame_count = self.spec.shape[1]
        self.characters = " .:;+=xX$&@"  
        
    def normalize_data(self, data):
        min_val = np.min(data)
        max_val = np.max(data)
        return ((data - min_val) * 9 / (max_val - min_val)).astype(int)
    
    def get_color(self, intensity):
        # Map intensity to colors
        if intensity < 3:
            return Fore.BLUE
        elif intensity < 6:
            return Fore.GREEN
        elif intensity < 8:
            return Fore.YELLOW
        else:
            return Fore.RED
    
    def visualize_frame(self, frame_data):
        # Clear screen (works on both Windows and Unix)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        normalized = self.normalize_data(frame_data)
        visualization = ""
        
        # Create the frame
        for row in reversed(normalized):
            for val in row:
                color = self.get_color(val)
                visualization += color + self.characters[val] + Style.RESET_ALL
            visualization += "\n"
            
        print(visualization)
    
    def play(self, fps=24):
        frame_time = 1/fps
        hop_length = int(self.sr/fps)
        
        print("Starting visualization... Press Ctrl+C to stop")
        time.sleep(2)
        
        try:
            for frame in range(self.frame_count):
                start_time = time.time()
                
                # Get current frame data
                frame_data = self.spec_db[:, frame]
                # Reshape for visualization
                frame_data = frame_data.reshape(-1, 4).mean(axis=1)
                
                self.visualize_frame(frame_data)
                
                # Maintain consistent frame rate
                processing_time = time.time() - start_time
                if processing_time < frame_time:
                    time.sleep(frame_time - processing_time)
                    
        except KeyboardInterrupt:
            print("\nVisualization stopped by user")

def main():
    print("ASCII Music Visualizer")
    print("=====================")
    
    # Get audio file path from user
    while True:
        file_path = input("\nEnter the path to your audio file (mp3/wav): ")
        if os.path.exists(file_path):
            break
        print("File not found. Please try again.")
    
    # Create and run visualizer
    visualizer = MusicVisualizer(file_path)
    visualizer.play()

if __name__ == "__main__":
    main()
