# Mental Health RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot designed for mental health support conversations. The system combines document retrieval with generative AI to provide intelligent, context-aware responses.

##  Features

- **Document Retrieval**: Uses sentence transformers to find relevant responses from knowledge base
- **Generative AI**: Employs DistilGPT-2 for contextual response generation
- **Mental Health Focus**: Trained on therapeutic conversation patterns
- **Dual Implementation**: Simple and advanced RAG versions
- **Interactive Chat**: Real-time conversation interface

##  Project Structure

```
mental-health-rag-bot/
├── .env                    # Environment variables (HuggingFace API key)
├── .gitignore             # Git ignore file
├── Dockerfile             # Docker containerization
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── intents.json          # Training data (388 knowledge items)
├── simple_rag.py         # Simple RAG implementation
└── full_rag.py           # Advanced RAG with generative model
```

##  Installation

### Prerequisites
- Python 3.9+
- HuggingFace API key (optional, for future enhancements)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Dhara121/Chatbot.git
   cd mental-health-rag-bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "HUGGINGFACE_API_KEY=your_api_key_here" > .env
   ```

4. **Run the chatbot**
   ```bash
   # Simple version (recommended first)
   python simple_rag.py
   
   # Advanced version with generative model
   python full_rag.py
   ```

### Docker Setup

1. **Build the image**
   ```bash
   docker build -t mental-health-rag-bot .
   ```

2. **Run the container**
   ```bash
   # Simple RAG version
   docker run -it mental-health-rag-bot
   
   # Full RAG version
   docker run -it mental-health-rag-bot python full_rag.py
   
   # Interactive bash
   docker run -it mental-health-rag-bot bash
   ```

##  How RAG Works

### 1. **Retrieval Phase**
- Converts user input into embeddings using `sentence-transformers`
- Searches knowledge base (388 items) for most relevant documents
- Uses cosine similarity to rank relevance

### 2. **Augmentation Phase**
- Provides retrieved documents as context
- Filters results based on confidence threshold (0.3)

### 3. **Generation Phase**
- **Simple RAG**: Returns pre-defined responses from matched intents
- **Full RAG**: Uses DistilGPT-2 to generate contextual responses

##  Usage Examples

### Sample Conversations

```
User: Hello
Bot: Hi there. What brings you here today?
Confidence: 1.000

User: I'm feeling anxious about work
Bot: It sounds like work is causing you some stress. Would you like to talk about what specifically is making you feel anxious?
Confidence: 0.842

User: I can't sleep at night
Bot: Sleep difficulties can be really challenging. Let's explore what might be keeping you awake.
Confidence: 0.756
```

### Interactive Commands

- Type your message and press Enter
- Type `quit`, `exit`, or `bye` to end the conversation
- The bot shows confidence scores for transparency

##  Technical Details

### Models Used
- **Embeddings**: `all-MiniLM-L6-v2` (sentence-transformers)
- **Generation**: `DistilGPT-2` (lightweight GPT-2 variant)
- **Similarity**: Cosine similarity for document retrieval

### Key Components
- **Knowledge Base**: 388 mental health conversation patterns
- **Embedding Dimension**: 384 (MiniLM)
- **Retrieval Top-K**: 3 most relevant documents
- **Confidence Threshold**: 0.3 for response relevance




