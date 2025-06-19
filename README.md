# YouTube Video Chat RAG Application

Demo : [Video Demo](https://www.linkedin.com/posts/abdul-basit-solkar-377929250_built-a-rag-application-that-lets-you-activity-7335249227922456576-xfSh?utm_source=share&utm_medium=member_desktop&rcm=ACoAAD4SzMsBmInwUdcrUpird3AxvTIV7d7D82s)

Imagine asking a long tutorial video, "What's the best way to [do something specific]?" and getting an instant, accurate answer based on the video's content. That's the power of this application. This project was a fantastic dive into applying RAG to real-world problems and making information more accessible.

## Features

* **Interactive Q&A**: Ask specific questions about the content of a YouTube video.
* **Instant & Accurate Answers**: Get precise responses directly derived from the video's transcript.
* **Contextual Understanding**: The application leverages RAG to understand your queries in the context of the video.

## Technologies Used

Under the hood, this application seamlessly integrates powerful tools:

* **LangChain**: Used to connect all the components (LLM, vector store, retrievers) and orchestrate the RAG pipeline.
* **Chroma**: Serves as the vector store to store and retrieve searchable embeddings generated from YouTube video transcripts.
* **Llama 3.2**: Leveraged for its capabilities in generating intelligent and coherent responses to user queries.

## How It Works

1.  **Video Transcription**: The application first obtains the transcript of the target YouTube video.
2.  **Embedding Creation**: The transcript is then broken down into chunks, and each chunk is converted into a numerical vector (embedding).
3.  **Vector Storage**: These embeddings are stored in the Chroma vector store, making them efficiently searchable.
4.  **Retrieval Augmented Generation**: When a user asks a question:
    * The question is converted into an embedding.
    * Relevant transcript chunks are retrieved from the Chroma vector store based on their similarity to the question's embedding.
    * These retrieved chunks, along with the user's question, are fed to the Llama 3.2 language model.
    * Llama 3.2 then generates an informed and contextually relevant answer based on the provided information.
      
## Future Improvements
1. Figure out a way to reduce token usage.
2. Using a cloud-based vector database.
3. Maintaining conversational history.
4. Convert the platform into a chrome extenstion.
