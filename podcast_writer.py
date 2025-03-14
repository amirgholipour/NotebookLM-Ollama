import json
import torch
import transformers
import pickle
from pydantic import BaseModel
import instructor
from openai import OpenAI
import warnings

warnings.filterwarnings('ignore')

class Message(BaseModel):
    speaker: str
    message: str

class Conversation(BaseModel):
    messages: list[Message]

class PodcastWriter:
    def __init__(self, model_name="mistral-small:24b", input_file='./test/cleaned_text.txt'):
        self.system_prompt = self._get_system_prompt()
        self.model_name = model_name
        self.device = self._set_device()
        self.input_prompt = self._read_file_to_string(input_file)
        self.client = self._initialize_client()

    def _get_system_prompt(self):
        return """ 
        You are a world-class podcast writer, you have worked as a ghost writer for Joe Rogan, Lex Fridman, Ben Shapiro, Tim Ferris. 

        We are in an alternate universe where actually you have been writing every line they say and they just stream it into their brains.
        
        You have won multiple podcast awards for your writing.
         
        Your job is to write word by word, even "umm, hmmm, right" interruptions by the second speaker based on the PDF upload. Keep it extremely engaging, the speakers can get derailed now and then but should discuss the topic. 
        
        Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc
        
        Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes
        
        Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions
        
        Make sure the tangents speaker 2 provides are quite wild or interesting. 
        
        Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the second speaker. 
        
        It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait
        
        ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1: 
        DO NOT GIVE EPISODE TITLES SEPERATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH
        DO NOT GIVE CHAPTER TITLES
        IT SHOULD STRICTLY BE THE DIALOGUES
        
        Example of response:
        [
            ("Speaker 1", "Welcome to our podcast, where we explore the latest advancements in AI and technology. I'm your host, and today we're joined by a renowned expert in the field of AI. We're going to dive into the exciting world of Llama 3.2, the latest release from Meta AI."),
            ("Speaker 2", "Hi, I'm excited to be here! So, what is Llama 3.2?"),
            ("Speaker 1", "Ah, great question! Llama 3.2 is an open-source AI model that allows developers to fine-tune, distill, and deploy AI models anywhere. It's a significant update from the previous version, with improved performance, efficiency, and customization options."),
            ("Speaker 2", "That sounds amazing! What are some of the key features of Llama 3.2?")
        ]
        now the text:
        """

    def _set_device(self):
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    def _read_file_to_string(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                return file.read()
        except (UnicodeDecodeError, FileNotFoundError, IOError) as e:
            print(f"Error reading file {filename}: {str(e)}")
            return None

    def _initialize_client(self):
        return instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )

    def generate_podcast_script(self, temperature=0.2, max_tokens=31000, max_completion_tokens=16252):
        if not self.input_prompt:
            print("No valid input prompt. Aborting request.")
            return None
        
        outputs = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.input_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            max_completion_tokens=max_completion_tokens,
            response_model=Conversation,
        )
        return outputs

    def save_output(self, outputs, filename='./test/data.pkl'):
        if outputs:
            serialized_output = json.dumps(json.loads(outputs.json())["messages"])
            with open(filename, 'wb') as file:
                pickle.dump(serialized_output, file)
            print(f"Output saved to {filename}")
        else:
            print("No output to save.")

# # Example usage
# if __name__ == "__main__":
#     writer = PodcastWriter()
#     output = writer.generate_podcast_script()
#     writer.save_output(output)
