import os # For setting environment variables
from langchain_community.vectorstores import FAISS # Vector database for storing embeddings. FAISS - Facebook AI Similarity Search, se
from langchain_openai import OpenAIEmbeddings # OpenAI's embedding model
from langchain_community.document_loaders import TextLoader # To load text files 
from langchain.text_splitter import RecursiveCharacterTextSplitter # For splitting text into chunks (adapted for pdf)
from langchain_openai import ChatOpenAI # OpenAI's chat model
from langchain.chains import create_retrieval_chain # To create a retrieval chain
from langchain.chains.combine_documents import create_stuff_documents_chain # To combine retrieved documents
from langchain.prompts import ChatPromptTemplate # For creating prompt templates
from langchain_community.document_loaders import PyPDFLoader # For converting pdf to txt

# Set your OpenAI API key (this will be charged)
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Load your knowledge base from a pdf file
loader = PyPDFLoader("Screenwriting_for_Dummies.pdf") # Knowledge file
documents = loader.load() # This loads the entire file into memory

# Split the text into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents) # Creates a list of document chunks

# Convert text chunks into vector embeddings (this costs money!)
embeddings = OpenAIEmbeddings() # Initialises OpenAI's embedding model
vector_store = FAISS.from_documents(chunks, embeddings) # Creates a vector store from chunks
retriever = vector_store.as_retriever(search_kwargs = {"k": 3}) # Creates a retriever that returns top 3 relevant chunks

# Initialise the language model (this costs money per query)
llm = ChatOpenAI(model = "gpt-3.5-turbo") # Uses GPT 3.5 Turbo model

# Create template for how the LLM should use the context to answer questions
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context:

{context}

Question: {input}
""") # Template with placeholders for context and user input

# Set up the document chain to process retrieved documents with the LLM
combine_docs_chain = create_stuff_documents_chain(
    llm, # the large language model
    prompt # the prompt template
) # Creates a chain that stuffs all documents into the prompt

# Create the RAG chain that connects retrieval with generation
rag_chain = create_retrieval_chain(
    retriever,  # The retriever component
    combine_docs_chain  # The document chain component
)  # Connects retrieval and generation in one chain

# Create a simple chat interface
def chat():
    print("RAG Chatbot initialised. Type 'exit' to end")
    while True: # Infinite loop for continuous interaction
        user_input = input("You: ") # Get input from the user
        if user_input.lower() == "exit": # End session if user chooses to exit
            break
    
        # Process the user input through the RAG chain
        response = rag_chain.invoke({"input": user_input}) # This costs money (embedding query and LLM processing)
        print(f"Bot: {response["answer"]}") # Print just the answer part of the response
        
# Run the chat function if this script is exectued directly 
if __name__ == "__main__":
    chat()