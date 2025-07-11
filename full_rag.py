import json
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

class RAGChatbot:
    def __init__(self, data_file='intents.json'):
        """Initialize the RAG chatbot with document retrieval and generation capabilities"""
        
        # Load your intent data
        self.load_documents(data_file)
        
        # Initialize embedding model for retrieval
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize generative model (using a light model from Hugging Face)
        print("Loading generative model...")
        self.setup_generative_model()
        
        # Create embeddings for all documents
        print("Creating document embeddings...")
        self.create_document_embeddings()
        
        print("RAG Chatbot initialized successfully!")
    
    def load_documents(self, data_file):
        """Load and process the intent data as documents"""
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Convert intents to documents for RAG
        self.documents = []
        self.metadata = []
        
        for intent in data['intents']:
            tag = intent['tag']
            patterns = intent['patterns']
            responses = intent['responses']
            
            # Each pattern becomes a document
            for pattern in patterns:
                self.documents.append(pattern)
                self.metadata.append({
                    'tag': tag,
                    'responses': responses,
                    'type': 'pattern'
                })
            
            # Each response becomes a document too
            for response in responses:
                self.documents.append(response)
                self.metadata.append({
                    'tag': tag,
                    'responses': responses,
                    'type': 'response'
                })
        
        print(f"Loaded {len(self.documents)} documents")
    
    def setup_generative_model(self):
        """Setup a lightweight generative model"""
        # Using DistilGPT-2 - a smaller, faster version of GPT-2
        model_name = "distilgpt2"
        
        # Check if GPU is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move model to device
        self.generator = self.generator.to(device)
        self.device = device
        
        print(f"Generative model loaded on {device}")
    
    def create_document_embeddings(self):
        """Create embeddings for all documents"""
        self.document_embeddings = self.embedding_model.encode(
            self.documents,
            convert_to_tensor=True,
            show_progress_bar=True
        )
    
    def retrieve_relevant_docs(self, query, top_k=3):
        """Retrieve the most relevant documents for a query"""
        # Encode the query
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(
            query_embedding.cpu().numpy(),
            self.document_embeddings.cpu().numpy()
        )[0]
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_docs = []
        for idx in top_indices:
            relevant_docs.append({
                'document': self.documents[idx],
                'metadata': self.metadata[idx],
                'similarity': similarities[idx]
            })
        
        return relevant_docs
    
    def generate_response(self, query, retrieved_docs):
        """Generate a response using retrieved documents as context"""
        # Prepare context from retrieved documents
        context = ""
        for doc in retrieved_docs:
            if doc['metadata']['type'] == 'response':
                context += doc['document'] + " "
        
        # If no good context found, use a default approach
        if not context.strip():
            # Check if query matches any specific tag
            for doc in retrieved_docs:
                if doc['similarity'] > 0.3:  # Threshold for relevance
                    return np.random.choice(doc['metadata']['responses'])
        
        # Create a prompt for generation
        prompt = f"Context: {context.strip()}\nUser: {query}\nAssistant:"
        
        # Generate response
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.generator.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9
            )
        
        # Decode and clean the response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        # Fallback to retrieved responses if generation is poor
        if len(response) < 5 or not response:
            for doc in retrieved_docs:
                if doc['similarity'] > 0.2:
                    return np.random.choice(doc['metadata']['responses'])
        
        return response if response else "I'm not sure how to help with that. Can you rephrase your question?"
    
    def chat(self, query):
        """Main chat function that combines retrieval and generation"""
        print(f"\nUser: {query}")
        
        # Step 1: Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_docs(query, top_k=3)
        
        # Show retrieved documents (for debugging)
        print("\nRetrieved documents:")
        for i, doc in enumerate(relevant_docs):
            print(f"{i+1}. {doc['document'][:50]}... (similarity: {doc['similarity']:.3f})")
        
        # Step 2: Generate response
        response = self.generate_response(query, relevant_docs)
        
        print(f"\nBot: {response}")
        return response

# Example usage
def main():
    # Initialize the chatbot
    chatbot = RAGChatbot('intents.json')
    
    # Interactive chat loop
    print("\nRAG Chatbot is ready! Type 'quit' to exit.")
    print("=" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        if user_input:
            chatbot.chat(user_input)

if __name__ == "__main__":
    main()