import pyaudio
import wave
import sys 
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv(dotenv_path="../.env")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")

def record_audio(duration=5, output_path=os.path.join(OUTPUT_PATH, "recording.wav")):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024

    audio = pyaudio.PyAudio()

    # Open audio stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []

    for _ in range(int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save recording to file
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        
        
# This ensures the following code runs only if the script is executed directly
if __name__ == "__main__":
    # Default values or command line arguments can be processed here
    duration = 5
    output_path = '../output/recording.wav'
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    if len(sys.argv) > 2:
        duration = int(sys.argv[2])

    record_audio(duration, output_path)
