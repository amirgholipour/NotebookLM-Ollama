import pickle
import re
import random
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from pydub import AudioSegment
import torch
from kokoro import KPipeline
import unicodedata
import json

class PodcastAudioGenerator:
    def __init__(self, podcast_text_path):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.podcast_text_path = podcast_text_path
        self.podcast_text = self.load_podcast_text()
        self.speaker_lengths = self.calculate_speaker_lengths()
        self.average_length = self.calculate_average_length()
        self.final_audio = None
        # self.initial =  None
        self.pipeline = KPipeline(lang_code='a',repo_id='hexgrad/Kokoro-82M') # <= make sure lang_code matches voice

    
    def clean_text(self, text):
        """Removes unwanted Unicode characters and replaces them with ASCII equivalents."""
        replacements = {
            # "\u2019": "'",  # Right single quote → straight quote
            # "\u2018": "'",  # Left single quote → straight quote
            # "\u201c": '"',  # Left double quote → straight quote
            # "\u201d": '"',  # Right double quote → straight quote
            # "\u2013": "-",  # En dash → hyphen
            # "\u2014": "-",  # Em dash → hyphen
            # "\u2026": "...",  # Ellipsis → three dots
            # "\u00A0": " ",  # Non-breaking space → normal space
            # "\u200B": "",  # Zero-width space → remove
            # "\uFEFF": "",  # Zero-width no-break space → remove
            "*": "",  # Remove asterisks
        }
    
        for key, value in replacements.items():
            text = text.replace(key, value)
        # Normalize other Unicode anomalies
        text = unicodedata.normalize("NFKC", text)
    
        # Explicitly decode any remaining escaped Unicode sequences
        text = text.encode("utf-8").decode("unicode_escape")
        return text

    def extract_speaker_messages(self,raw_content):
        # Try parsing the input as JSON
        try:
            parsed_content = json.loads(raw_content)
            if isinstance(parsed_content, list) and all(isinstance(item, dict) for item in parsed_content):
                return [(entry["speaker"], entry["message"]) for entry in parsed_content]
        except json.JSONDecodeError:
            pass  # Proceed with regex if JSON parsing fails
    
        # Fallback to regex-based extraction for tuple-like input
        pattern = r'\(\s*"([^"]+)",\s*"((?:\\.|[^"\\])*)"\s*\)'
        matches = re.findall(pattern, raw_content)
        return matches

    def load_podcast_text(self):
        """Loads and cleans podcast text from a pickle file."""
        with open(self.podcast_text_path, 'rb') as file:
            content = pickle.load(file)
        # self.initial  = content
        # print("Raw Content:", content)  # Debugging
        # print(content)
        if isinstance(content, str):
            print("content str")
            
            matches = self.extract_speaker_messages(content)
            
            # print(matches)
    
            # Apply text cleaning for both speaker and message
            return [(self.clean_text(speaker.strip()), self.clean_text(text.strip())) for speaker, text in matches]
    
        # If content is already structured as a list of dictionaries, clean it
        if isinstance(content, list):
            for entry in content:
                entry["speaker"] = self.clean_text(entry["speaker"])
                entry["message"] = self.clean_text(entry["message"])
    
        return content


    
    def calculate_speaker_lengths(self):
        speaker_lengths = defaultdict(list)
        for speaker, text in self.podcast_text:
            speaker_lengths[speaker].append(len(text))
        return speaker_lengths
    
    def calculate_average_length(self):
        lengths = [len(text) for _, text in self.podcast_text]
        return sum(lengths) / len(lengths) if lengths else 0
    
    def determine_speed_range(self, text_length):
        if text_length < self.average_length:
            return 0.8, 0.92  # Shorter than average: lower speed range
        else:
            return 0.92, 1.05  # Average or longer: higher speed range


    
    def generate_audio(self, text, voice):
        text_length = len(text)
        min_speed, max_speed = self.determine_speed_range(text_length)
        speed = random.uniform(min_speed, max_speed)
        print(f"Speed: {speed}")
        
        try:
            generator = self.pipeline(
                text, voice=voice,
                speed=speed, split_pattern=r'\n+'
            )
            
            for _, _, audio in generator:
                audio_tensor = audio.cpu().detach().numpy()
                return np.squeeze(audio_tensor), 24000
        except Exception as e:
            print(f"Error generating audio for text: {text[:50]}... | Error: {e}")
            return None, None
    
    def numpy_to_audio_segment(self, audio_arr, sampling_rate):
        if audio_arr is None or sampling_rate is None:
            return None
        if audio_arr.dtype != np.int16:
            audio_arr = (audio_arr * 32767).astype(np.int16)
        return AudioSegment(
            audio_arr.tobytes(), 
            frame_rate=sampling_rate,
            sample_width=audio_arr.dtype.itemsize, 
            channels=1
        )
    
    def generate_podcast_audio(self, output_path):
        generated_any_audio = False
        for speaker, text in tqdm(self.podcast_text, desc="Generating podcast segments", unit="segment"):
            voice = 'am_liam' if speaker == "Speaker 1" else 'af_heart'
            audio_arr, rate = self.generate_audio(text, voice)
            
            if audio_arr is None or rate is None:
                print(f"Skipping segment for {speaker} due to generation failure.")
                continue
            
            audio_segment = self.numpy_to_audio_segment(audio_arr, rate)
            if audio_segment is None:
                print(f"Skipping segment for {speaker} due to conversion failure.")
                continue
            
            if self.final_audio is None:
                self.final_audio = audio_segment
            else:
                self.final_audio += audio_segment
            
            generated_any_audio = True
        
        if not generated_any_audio:
            print("No audio was generated. Check your input data and model.")
            return []
        # return self.final_audio
        
        self.final_audio.export(output_path, format="mp3", bitrate="192k", parameters=["-q:a", "0"])
        print(f"Podcast audio saved to {output_path}")

# Usage
# generator = PodcastAudioGenerator('./examples/podcast_ready_data.pkl')
# generator.generate_podcast_audio("./examples/_podcast.mp3")
