# Podcast Generator from PDF

## Overview

This project automates the process of extracting text from a PDF document, converting it into a structured podcast script, refining the conversation, and generating an audio file. The workflow is divided into four main steps:

1. **Extracting Text from PDF** using `PDFProcessor`
2. **Generating a Podcast Script** using `PodcastWriter`
3. **Refining the Script** using `PodcastRewriter`
4. **Generating the Podcast Audio** using `PodcastAudioGenerator`

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- Required dependencies (install with the command below)

### Install Dependencies

Run the following command to install the required packages:

```bash
pip install -r requirements.txt  
```

## Project Structure

```
📂 Project Directory  
│── 📂 test/  
│   ├── 2402.13116v4.pdf  # Example input PDF  
│   ├── podcast_ready_data.pkl  # Processed data for podcast generation  
│   ├── _podcast.mp3  # Generated podcast audio file  
│── 📜 Run_all.ipynb  # Main script to execute the workflow  
│── 📜 pdf_processor.py  # Extracts and cleans text from a PDF  
│── 📜 podcast_writer.py  # Generates a structured podcast script  
│── 📜 podcast_rewriter.py  # Refines the script for better quality  
│── 📜 podcast_audio_generator.py  # Converts the script into audio  
│── 📜 requirements.txt  # Required Python libraries  
│── 📜 README.md  # Project documentation  
```

## Usage

Follow these steps to generate a podcast from a PDF:

### Step A: Process the PDF

1. Create an instance of `PDFProcessor` by providing the PDF file path and model name.
2. Extract, clean, and save the text using the `process_pdf()` method.

```python
from pdf_processor import PDFProcessor  

pdf_path = "./test/2402.13116v4.pdf"  
processor = PDFProcessor(pdf_path, model_name="llama3.2:3b")  
processor.process_pdf()  
```

### Step B: Generate the Podcast Script

1. Initialize `PodcastWriter` with the desired model.
2. Generate a structured podcast script.
3. Save the output.

```python
from podcast_writer import PodcastWriter  

writer = PodcastWriter(model_name="gemma3:27b")  
output = writer.generate_podcast_script()  
writer.save_output(output)  
```

### Step C: Optimize the Podcast Script

1. Initialize `PodcastRewriter` to refine the script.
2. Save the optimized script.

```python
from podcast_rewriter import PodcastRewriter  

rewriter = PodcastRewriter(model_name="phi4:latest")  
rewriter.save_output()  
```

### Step D: Generate the Podcast Audio

1. Load the processed script data.
2. Convert the refined script into an audio podcast.

```python
from podcast_audio_generator import PodcastAudioGenerator  

generator = PodcastAudioGenerator('./test/podcast_ready_data.pkl')  
generator.generate_podcast_audio("./test/_podcast.mp3")  
```

## Output

- The processed text is stored in an intermediate file.
- The final podcast script is saved for review.
- The generated podcast audio is saved as `test/_podcast.mp3`.

## Customization

- Change the model used for each processing step by modifying the `model_name` parameter.
- Adjust the script refinement parameters inside `PodcastRewriter`.
- Modify the voice style and configuration in `PodcastAudioGenerator` if needed.

## Future Improvements

- Add support for multiple voice options in audio generation.
- Implement user-defined script customization features.
- Improve error handling and logging.

## License

This project is open-source. Feel free to modify and enhance it as needed.

## Contact

For any issues or suggestions, please open an issue or reach out via GitHub.
