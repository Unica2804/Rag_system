from src.embedding import EmbeddingProcessor

embedding= EmbeddingProcessor(device='cuda', batch_size=32)
texts=["This is a sample text."]
embeddings= embedding.generate_doc_embeddings(texts)
print(embeddings)
print(f"Embeddings generated on device: {embedding.get_device()}")