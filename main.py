from helpers import (create_chunks, document_retriever,
                       get_transcript , create_vector_store , ConversationalYouTubeChat)
import sys

print("🎥 Conversational YouTube Chat Bot")
print("=" * 40)

# Get video link from user
link = input("📹 Paste YouTube video link: ")

# Process video
print("\n🔄 Processing video...")
transcript = get_transcript(link=link)

if not transcript:
    print("❌ Could not get transcript for this video.")
    sys.exit() 

chunks = create_chunks(transcript=transcript)
vector_store = create_vector_store(chunks, link=link)

if not vector_store:
    print("❌ Could not create vector store.")
    sys.exit()
    
retriever = document_retriever(vector_store=vector_store)

if not retriever:
    print("❌ Could not create document retriever.")
    sys.exit()


chat_bot = ConversationalYouTubeChat(retriever, memory_window=5)

print("\n✅ Ready to chat! (Type 'quit' to exit, 'clear' to clear history)")
print("💡 I can remember our conversation context!\n")

while True:
    query = input("🗣️  You: ").strip()
    
    if query.lower() in ['quit', 'exit', 'q']:
        print("👋 Goodbye!")
        break
    elif query.lower() == 'clear':
        chat_bot.clear_history()
        continue
    elif not query:
        continue
    
    print("🤖 Bot: ", end="")
    response = chat_bot.chat(query)
    print(response)
    print()