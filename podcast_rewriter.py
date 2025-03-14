import pickle
import warnings
import ollama

warnings.filterwarnings('ignore')

class PodcastRewriter:
    def __init__(self, model_name="llama3.2:3b", input_data_path="./test/data.pkl"):
        self.model = model_name
        self.input_data_path = input_data_path
        self.system_prompt = self._load_system_prompt()
        self.input_prompt = self._load_input_prompt()
        

    def _load_system_prompt(self):
        return """
        You are an international oscar winnning screenwriter.

        You have been working with multiple award winning podcasters.
        
        Your job is to use the podcast transcript written below to re-write it for an AI Text-To-Speech Pipeline. A very dumb AI had written this so you have to step up for your kind.
        
        Make it as engaging as possible, Speaker 1 and 2 will be simulated by different voice engines
        
        Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc
        
        Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes
        
        Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions
        
        Make sure the tangents speaker 2 provides are quite wild or interesting. 
        
        
        REMEMBER THIS WITH YOUR HEART
        Speaker 2 must  introduce her/him self at the biginning of his/her speech, and express greatfulness of being a part of this conversation.
        
        
        It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait
        
        Please re-write to make it as characteristic as possible
        
        START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:
        
        STRICTLY RETURN YOUR RESPONSE AS A LIST OF TUPLES OK? 
        
        IT WILL START DIRECTLY WITH THE LIST AND END WITH THE LIST NOTHING ELSE
        
        Example of response:
        [
            ("Speaker 1", "Welcome to our podcast, where we explore the latest advancements in AI and technology. I'm your host, and today we're joined by a renowned expert in the field of AI. We're going to dive into the exciting world of Llama 3.2, the latest release from Meta AI."),
            ("Speaker 2", "Hi, I'm excited to be here! So, what is Llama 3.2?"),
            ("Speaker 1", "Ah, great question! Llama 3.2 is an open-source AI model that allows developers to fine-tune, distill, and deploy AI models anywhere. It's a significant update from the previous version, with improved performance, efficiency, and customization options."),
            ("Speaker 2", "That sounds amazing! What are some of the key features of Llama 3.2?")
        ]
        the podcast transcript:
        """  # The system prompt content should be added here

    def _load_input_prompt(self):
        with open(self.input_data_path, 'rb') as file:
            return pickle.load(file)

    def generate_podcast_script(self):
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.input_prompt},
        ]

        response = ollama.chat(
            model=self.model,
            messages=conversation,
            options={'num_ctx': 8126*2, 'temperature': 1}
        )

        return response['message']['content']

    def save_output(self, output_path="./test/podcast_ready_data.pkl"):
        output = self.generate_podcast_script()
        with open(output_path, 'wb') as file:
            pickle.dump(output, file)
        print("Podcast script saved successfully.")
        print(output)

# # Example Usage
# if __name__ == "__main__":
#     rewriter = PodcastRewriter()
#     rewriter.save_output()
