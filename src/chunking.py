from langchain_text_splitters import RecursiveCharacterTextSplitter

# splitting docs into chunks

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=True
    )
    split_docs=text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    # Example of a split doc
    print("\nExample splitting:")
    print(f"content {split_docs[0].page_content[:200]}...")
    print(f"metadata {split_docs[0].metadata}")

    return split_docs