import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

class SimpleRAGChatbot:
    def __init__(self):
        """Initialize simple RAG chatbot"""
        
        # Load your data
        self.load_data()
        
        # Initialize embedding model for document retrieval
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embeddings for retrieval
        self.create_embeddings()
        
        print("Simple RAG Chatbot ready!")
    
    def load_data(self):
        """Load the intent data"""
        with open('intents.json', 'r') as f:
            data = json.load(f)
        
        self.intents = data['intents']
        
        # Create a knowledge base for retrieval
        self.knowledge_base = []
        for intent in self.intents:
            # Add patterns and responses to knowledge base
            for pattern in intent['patterns']:
                self.knowledge_base.append({
                    'text': pattern,
                    'tag': intent['tag'],
                    'responses': intent['responses'],
                    'type': 'pattern'
                })
            
            for response in intent['responses']:
                self.knowledge_base.append({
                    'text': response,
                    'tag': intent['tag'],
                    'responses': intent['responses'],
                    'type': 'response'
                })
        
        print(f"Loaded {len(self.knowledge_base)} knowledge items")
    
    def create_embeddings(self):
        """Create embeddings for all knowledge base items"""
        texts = [item['text'] for item in self.knowledge_base]
        self.embeddings = self.embedding_model.encode(texts)
        print("Created embeddings for knowledge base")
    
    def retrieve_and_respond(self, query, top_k=3):
        """Retrieve relevant items and generate response"""
        
        # 1. RETRIEVAL: Find most relevant documents
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar items
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 2. RESPONSE GENERATION: Use retrieved items to generate response
        best_match = None
        best_score = 0
        
        for idx in top_indices:
            if similarities[idx] > best_score:
                best_score = similarities[idx]
                best_match = self.knowledge_base[idx]
        
        # Generate response based on best match
        if best_match and best_score > 0.3:  # Threshold for relevance
            # Return a random response from the matched intent
            response = random.choice(best_match['responses'])
            
            # Add some context-aware enhancement
            if best_match['tag'] == 'greeting':
                response = self.enhance_greeting(query, response)
            elif best_match['tag'] in ['morning', 'afternoon', 'evening']:
                response = self.enhance_time_greeting(query, response)
            
            return response, best_score
        else:
            # Fallback response
            return "I'm not sure about that. Can you ask me something else?", 0.0
    
    def enhance_greeting(self, query, response):
        """Enhance greeting responses with context"""
        if any(word in query.lower() for word in ['how', 'what', 'tell']):
            return response + " What would you like to know?"
        return response
    
    def enhance_time_greeting(self, query, response):
        """Enhance time-specific greetings"""
        return response + " How can I help you today?"
    
    def chat(self, query):
        """Main chat function"""
        print(f"\nUser: {query}")
        
        # Get response using RAG
        response, confidence = self.retrieve_and_respond(query)
        
        print(f"Bot: {response}")
        print(f"Confidence: {confidence:.3f}")
        
        return response

# Simple test function
def test_chatbot():
    """Test the chatbot with sample queries"""
    chatbot = SimpleRAGChatbot()
    
    test_queries = [
        "Hello",
        "Hi there",
        "Good morning",
        "Good evening",
        "How are you?",
        "What's up?",
        "Tell me something"
    ]
    
    print("\n" + "="*50)
    print("TESTING CHATBOT")
    print("="*50)
    
    for query in test_queries:
        chatbot.chat(query)
    
    print("\n" + "="*50)
    print("INTERACTIVE MODE")
    print("="*50)
    
    # Interactive mode
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        if user_input:
            chatbot.chat(user_input)

if __name__ == "__main__":
    test_chatbot()