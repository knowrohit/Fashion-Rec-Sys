import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')


def clean_text(text):
    # Remove timestamps, if any
    text = re.sub(r'\d{1,2}:\d{1,2}', '', text)
    
    # Break the transcript into sentences
    sentences = sent_tokenize(text)
    
    # Convert to lowercase and remove special characters and punctuation
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
        cleaned_sentences.append(sentence.strip())
    
    return cleaned_sentences


def process_transcripts(input_filenames, output_filename):
    with open(output_filename, 'w') as output_file:
        for filename in input_filenames:
            with open(filename, 'r') as input_file:
                transcript = input_file.read()
                cleaned_sentences = clean_text(transcript)
                
                for sentence in cleaned_sentences:
                    output_file.write(f"{sentence}\n")


input_filenames = ['/Users/rohittiwari/Downloads/ntcc_major_recsys/vogue_articles_full.txt']
output_filename = 'cleaned_fashion_dataset.txt'

process_transcripts(input_filenames, output_filename)
