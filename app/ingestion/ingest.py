import argparse
import pymupdf
import os
import hashlib
import tiktoken
from pathlib import Path
import json
import ast

enc = tiktoken.get_encoding("cl100k_base")
Dict_path = ".../data/processed"
output_file_path =".../data/proceeded/output"

def extract_pdf_stats(pdf_path):
    with pymupdf.open(pdf_path) as doc:
        num_pages=doc.page_count
        print(f"Number of pages in the PDF: {num_pages}")
        total_chars=sum(len(page.get_text()) for page in doc)
        print(f"Number of total characters in a pdf: {total_chars}")
        #File size in bytes
        file_size_bytes = os.path.getsize(pdf_path)
        print(f"File size in bytes: {file_size_bytes}")

    return num_pages,total_chars,file_size_bytes

#Calculate the text density,vowel ratio Newline ratio and flags for the pdf document
def extract_text_from_pdf(pdf_path):
    with pymupdf.open(pdf_path) as doc:
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        num_chars = len(full_text)
        num_vowels = sum(full_text.count(vowel) for vowel in "aeiouAEIOU")
        num_newlines = full_text.count("\n")
        flags=[]
        file_size_bytes = os.path.getsize(pdf_path)
        chars_per_kb = num_chars / (file_size_bytes / 1024)
        if chars_per_kb<10:
            flags.append("NEEDS OCR")
        vowel_ratio = num_vowels / num_chars if num_chars > 0 else 0
        if vowel_ratio<0.25 or vowel_ratio>0.45:
            flags.append("LOW_QUALITY_TEXT")
        newline_ratio = num_newlines / num_chars if num_chars > 0 else 0
        if newline_ratio>0.20:
            flags.append("POSSIBLE_TABLE_LAYOUT_ISSUE")
        
        print(f"Text Density: {chars_per_kb:.4f}")
        print(f"Vowel Ratio: {vowel_ratio:.4f}")
        print(f"Newline Ratio: {newline_ratio:.4f}")
        print(flags)

#Method to get the document Id based on the file path and its size from SHA256 hash 
def compute_document_id(pdf_path):
    sha256_hash = hashlib.sha256() 
    with open(pdf_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)          
    return sha256_hash.hexdigest()

#Method to iterate through the pdf document and extract the texts which returns the page number and page text as a listt of tuples
def iter_pages(pdf_path):
    with pymupdf.open(pdf_path) as doc:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            yield (page_num + 1, page.get_text())
           

#Method to print the token count for each page in the pdf document using tiktoken library
def estimate_tokens(page_num: int, text: str) -> int:
    token_count = len(enc.encode(text))
    #print(f"Page {page_num} has approximately {token_count} tokens.")
    return token_count

#Method for writing the chunks into individual dictionaries
def chunks_from_tokens(page_text,max_tokens=512):
    tokens = enc.encode(page_text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks





#Method to store the chunk dictionary to a json file , one chunk dictionary per line in the file
def store_dict_to_path(chunk_dict):
    os.makedirs(Dict_path, exist_ok=True)
    file_path = os.path.join(Dict_path, f"{chunk_dict['chunk_id']}.json")
    with open(file_path, "w") as f:
        json.dump(chunk_dict, f, ensure_ascii=False)


#Method to write all the chunks into a single json files

def save_chunks_json(Dict_path:str,output_file_path:str):
    in_dir=Path(Dict_path)
    out_path=Path(output_file_path)
    with out_path.open("w", encoding="utf-8") as out:
        for p in sorted(in_dir.glob("*.json")):
            with p.open("r", encoding="utf-8") as f:
                try:
                    obj = json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)
                    obj = ast.literal_eval(f.read())
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF documents for RAG system")
    # Support both positional and flagged usage.
    parser.add_argument("pdf_path", nargs="?", type=str, help="Path to the PDF document to ingest")
        
    args = parser.parse_args()

    if not args.pdf_path:
        parser.error("the following arguments are required: --pdf_path or pdf_path")
    
    extract_pdf_stats(args.pdf_path)
    extract_text_from_pdf(args.pdf_path)
    document_id = compute_document_id(args.pdf_path)
    print(f"Document ID: {document_id}")
    #chunking logic
    chunks=[]
    chunk_id=1
    for page_num,page_text in iter_pages(args.pdf_path):
        text_len = len(page_text.strip())
      
        if text_len <50:
            print(f"Skipping page {page_num} which has only {text_len} characters.")  
            continue
        token_count = estimate_tokens(page_num, page_text)
        print(f"Page {page_num} has {text_len} chars (~{token_count} tokens)")
        #Creating a chunking dictionary to store the chunk information for each page 
        page_chunks = chunks_from_tokens(page_text)
        print(f"Page {page_num} → {len(page_chunks)} chunks")
        for chunk_texts in page_chunks:
            chunk_dict={}
            chunk_dict["document_id"]=document_id
            chunk_dict["chunk_id"] = f"{document_id}_{chunk_id:05d}"
            chunk_dict["page_num"]=page_num
            chunk_dict["chunk_text"]=chunk_texts
            chunks.append(chunk_dict)
            chunk_id+=1
            #print the output of the chunk dictionary for the first 5 chunks
            if len(chunks)<=5:
                print(chunk_dict)
            store_dict_to_path(chunk_dict)
    print(f"Total chunks created: {len(chunks)}")
    save_chunks_json(Dict_path=Dict_path, output_file_path="/Users/anuraggupta/projects/sentinel-rag/data/processed/output/combined.jsonl")


        
