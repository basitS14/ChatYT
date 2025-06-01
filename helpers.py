from itertools import chain
import os
import re
from openai import vector_stores
from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import  FAISS
from langchain_chroma import Chroma
from langchain_ollama import  OllamaLLM , ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda , RunnablePassthrough , RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint , HuggingFaceEndpointEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

embedd_model = OllamaEmbeddings(model="nomic-embed-text:latest")
llm = ChatOllama(model="llama3.2:latest")



prompt = PromptTemplate(
    template="""
You are a helpful and conversational assistant that answers questions about YouTube videos based on their transcripts.
Use the provided context from the video transcript and the conversation history to answer the user's question accurately.

If the user's message is general (like greetings, expressions of thanks, or small talk), respond politely and naturally, 
even if the context from the video is not relevant. Maintain a friendly tone throughout.

If the user asks a specific question related to the video and the context is insufficient, say that you don't have enough information from the video.

Context from video transcript:
{context}

Chat History:
{chat_history}

Human: {question}
Assistant:""",
    input_variables=['context', 'chat_history', 'question']
)

def extract_id(link):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", link)
    return match.group(1) if match else None

def get_transcript(link):
    try:
        ytt_api = YouTubeTranscriptApi()
        video_id = extract_id(link)
        print(video_id)
        res = ytt_api.fetch(video_id)
        print("Working...")
        transcript = " ".join([snippet.text for snippet in res])
        print("Transcript Loaded !!!")
        return transcript
    except TranscriptsDisabled:
        print("Transcripts for this video are disabled by the creator")


def create_chunks(transcript):
    num_words = len(transcript.split(" "))
    
    if num_words >= 10000:
        # For very long videos, use larger chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
    elif num_words >= 5000:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, 
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
    
    chunks = splitter.create_documents([transcript])
    print(f"{len(chunks)} chunks created")
    return chunks


def create_vector_store(chunks , link):
    video_id = extract_id(link=link)
    persist_path = f"./chroma_store/{video_id}"

    if os.path.exists(persist_path):
        print(f"Loading vector store for video_id {video_id}")
        return Chroma(persist_directory=persist_path , embedding_function=embedd_model)
    else:
        print("Creating new  Vector Store...")
        vector_store = Chroma.from_documents(   
            documents=chunks,
            embedding=embedd_model,
            persist_directory=persist_path,
        )
        # vector_store.persist()
        print(f"Chroma vector store saved at: {persist_path}")

        return vector_store


def document_retriever(vector_store):
    retriever = vector_store.as_retriever(search_type="mmr" , search_kwargs={'k': 8 , 'fetch_k':20 , 'lambda_mult':0.7})
    # retriever_from_llm =MultiQueryRetriever.from_llm(
    #     retriever=vector_store.as_retriever(search_type="mmr" ,search_kwargs={'k': 5} ), llm=llm 
    # )
    # compressor = LLMChainExtractor.from_llm(llm)
    # compression_retriever = ContextualCompressionRetriever(
    # base_compressor=compressor, base_retriever=retriever_from_llm
    # )
    print("Retrieving Documents...")
    return retriever

def format_docs(retrived_docs):
    doc_context = "\n\n".join(doc.page_content for doc in retrived_docs)
    return doc_context

class ConversationalYouTubeChat:
    def __init__(self, retriever, memory_window=3):
        self.retriever = retriever
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            memory_key="chat_history",
            return_messages=True
        )
        self.output_parser = StrOutputParser()
        
        # Create a custom chain with output parser
        self.chain = (
            {
                "context": self.retriever | format_docs,
                "chat_history": lambda x: self._get_chat_history(),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | self.output_parser
        )
    
    def _get_chat_history(self):
        """Format chat history for the prompt"""
        messages = self.memory.chat_memory.messages
        if not messages:
            return "No previous conversation."
        
        history = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i].content
                ai_msg = messages[i + 1].content
                history.append(f"Human: {human_msg}\nAssistant: {ai_msg}")
        
        return "\n\n".join(history[-3:])  # Keep last 3 exchanges
    
    def chat(self, query):
        """Process a query and return response with conversation context"""
        try:
            # Get response from chain
            response = self.chain.invoke(query)
            
            # Save to memory
            self.memory.save_context(
                {"input": query},
                {"output": response}
            )
            
            return response
        except Exception as e:
            print(f"Error in chat: {e}")
            return "I'm sorry, I encountered an error while processing your question."
    
    def get_chat_history(self):
        """Get the current chat history"""
        return self.memory.chat_memory.messages
    
    def clear_history(self):
        """Clear the conversation history"""
        self.memory.clear()
        print("Chat history cleared!")