from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
llm=ChatGroq(
    model='openai/gpt-oss-20b',
    temperature=0.1,
    max_retries=2,
    max_tokens=1024    
)

# RAG pipeline

def rag_simple(query:str,retriever:object,llm:object,top_k:int=5):
    results=retriever.retrieve(query,top_k=top_k)
    context="\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        return "No relevant context found to answer question"
    prompt=f"""Use the following context to answer the questions asked by user
                context: {context}

                question: {query}

                Answer:
            """
    response=llm.invoke([prompt.format(context=context,query=query)])
    return response.content
