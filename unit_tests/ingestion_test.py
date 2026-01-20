from src.Ingestion import data_ingestor

ingestor= data_ingestor("./data")
doc=ingestor.ingest()
for i in range(len(doc)):
    print(doc[i].metadata)