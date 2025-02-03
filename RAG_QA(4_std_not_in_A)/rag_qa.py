from typing import List, Optional
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from RAG import SimpleDatabase, HuggingFaceEncoder, LocalLLM, parse_pdf

class RAGSystem:
    def __init__(
        self,
        encoder_model: str = 'sentence-transformers/all-mpnet-base-v2',
        llm_model: str = ['your_model'], #TODO: fit in your model
        n_documents: int = 3
    ):
        # Initialize components
        self.database = SimpleDatabase()
        self.encoder = HuggingFaceEncoder(model_name=encoder_model)
        self.llm = LocalLLM(model_name=llm_model)
        self.n_documents = n_documents

    def load_pdf(self, pdf_path: str | Path) -> None:
        """Load and index a PDF file"""
        with open(pdf_path, 'rb') as file:
            # Parse PDF into text chunks
            pages = parse_pdf(file)
            # Encode and store in database
            embeddings = self.encoder.encode(pages)
            self.database.add_documents(pages, embeddings)

        
    def generate_response(self, query: str) -> str:
        """Generate a response using RAG"""
        # Encode the query
        query_embedding = self.encoder.encode(query)
        
        # Retrieve relevant documents
        docs_with_scores = self.database.retrieve_documents(
            query_embedding, 
            n=self.n_documents
        )
        
        if not docs_with_scores:
            return "No relevant documents found to answer the query."
        
        # Format context from retrieved documents
        context = "\n\n".join(f'"""{doc}"""' for doc in docs_with_scores.keys())
        
        # Prepare messages for LLM
        messages = [
            {
                'role': 'system',
                'content': (
                    "You are a helpful assistant. "
                    "Answer the user's questions based on the provided document snippets. "
                    "Only use information from the provided snippets."
                )
            },
            {
                'role': 'user',
                'content': (
                    f"Context:\n{context}\n\n"
                    f"Question: {query}"
                )
            }
        ]
        
        # Generate response
        response = ""
        for token in self.llm.generate(messages):
            response += token
        return response
    
def main():
    rag = RAGSystem()
    
    # Load PDFs
    pdf_path = "Question answering - Hugging Face NLP Course.pdf"
    #TODO: load pdf into the model
    
    #TODO Ask questions, and generate the response

if __name__ == "__main__":
    main()