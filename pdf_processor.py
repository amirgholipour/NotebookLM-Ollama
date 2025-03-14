import PyPDF2
import os
import ollama
import warnings
from typing import Optional, List
from tqdm.notebook import tqdm

warnings.filterwarnings('ignore')


class PDFProcessor:
    def __init__(self, pdf_path: str, output_dir: str = "./test/", chunk_size: int = 1000, model_name: str = "granite3.2:8b"):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.model = model_name
        self.extracted_text = ""
        self.metadata = None
    
    def validate_pdf(self) -> bool:
        """Validates if the file exists and is a PDF."""
        return os.path.exists(self.pdf_path) and self.pdf_path.lower().endswith('.pdf')
    
    def extract_text(self, max_chars: int = 100000) -> Optional[str]:
        if not self.validate_pdf():
            print("Invalid PDF file.")
            return None
        
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                print(f"Processing PDF with {num_pages} pages...")
                
                extracted_text = []
                total_chars = 0
                
                for page_num in range(num_pages):
                    text = pdf_reader.pages[page_num].extract_text()
                    if total_chars + len(text) > max_chars:
                        extracted_text.append(text[:max_chars - total_chars])
                        break
                    extracted_text.append(text)
                    total_chars += len(text)
                    print(f"Processed page {page_num + 1}/{num_pages}")
                
                self.extracted_text = '\n'.join(extracted_text)
                return self.extracted_text
                
        except PyPDF2.PdfReadError:
            print("Error: Invalid or corrupted PDF file")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    
    def get_metadata(self) -> Optional[dict]:
        if not self.validate_pdf():
            return None
        
        try:
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                self.metadata = {
                    'num_pages': len(pdf_reader.pages),
                    'metadata': pdf_reader.metadata
                }
                return self.metadata
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return None
    
    def create_word_bounded_chunks(self, text: str) -> List[str]:
        words = text.split()
        chunks, current_chunk, current_length = [], [], 0
        
        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk, current_length = [word], word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def process_chunk(self, text_chunk: str) -> str:
        sys_prompt = """
        You are a world class text pre-processor, here is the raw data from a PDF, please parse and return it in a way that is crispy and usable to send to a podcast writer.
        
        The raw data is messed up with new lines, Latex math and you will see fluff that we can remove completely. Basically take away any details that you think might be useless in a podcast author's transcript.
        
        Remember, the podcast could be on any topic whatsoever so the issues listed above are not exhaustive
        
        Please be smart with what you remove and be creative ok?
        
        Remember DO NOT START SUMMARIZING THIS, YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED
        
        Be very smart and aggressive with removing details, you will get a running portion of the text and keep returning the processed text.
        
        PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPATILISATION ETC LIKES
        
        ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions ok?
        Here is the text:
        """
        
        conversation = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": text_chunk},
        ]
        
        response = ollama.chat(model=self.model, messages=conversation)
        return response['message']['content']
    
    def process_pdf(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        metadata = self.get_metadata()
        extracted_text = self.extract_text()
        if not extracted_text:
            return
        
        output_file = os.path.join(self.output_dir, "cleaned_text.txt")
        chunks = self.create_word_bounded_chunks(extracted_text)
        
        with open(output_file, 'w', encoding='utf-8') as out_file:
            for chunk_num, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
                processed_chunk = self.process_chunk(chunk)
                out_file.write(processed_chunk + "\n")
                out_file.flush()
        
        print(f"Processed text saved to {output_file}")
