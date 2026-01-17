import os
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path

def Pdf_processor(path_input):
    all_documents = []
    path_obj = Path(path_input)
    pdf_files = []

    # 1. Determine if input is a File or a Directory
    if path_obj.is_file():
        # Case A: It's a single file (API Upload Scenario)
        if path_obj.suffix.lower() == '.pdf':
            pdf_files.append(path_obj)
        else:
            print(f"Skipping non-PDF file: {path_obj.name}")

    elif path_obj.is_dir():
        # Case B: It's a directory (Batch Ingestion Scenario)
        pdf_files = list(path_obj.glob("**/*.pdf"))
    
    else:
        print(f"Path does not exist: {path_input}")
        return []

    print(f"Found {len(pdf_files)} PDF files to process")

    # 2. Process the list of files (Works for both cases)
    for pdf_file in pdf_files:
        print(f"\nProcessing pdf file {pdf_file.name}")
        try:
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load()

            # Add metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
            
            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages")
            
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents