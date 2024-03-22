import streamlit as st # simple way to start a web interface. There is no need for HTML, CSS, etc.
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI


OPENAI_API_KEY = "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" #openAI key (retrieved from platform.openai.com/api-keys)

#upload pdf files
st.header("Welcome to LazyReader")

with st.sidebar:
    st.title("My Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions!", type="pdf")

#extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""  # store here
    for page in pdf_reader.pages:
        text += page.extract_text()
        # st.write(text)


#break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators = "\n",
        chunk_size=1000, #1000 characters
        chunk_overlap=150, # duplicate the last 150 characters from the previous chunk
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    

    #generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #creating vector store - FAISS (facebook AI semantic search)
    vector_store = FAISS.from_texts(chunks, embeddings)

    #get user question
    user_input = st.text_input("Type your question here")

    #do similarity search
    if user_input:
        match = vector_store.similarity_search(user_input)
      

        #define the LLM -> fine tuning
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,

            # the lower the value for temperature, the smaller the randomness,
            # this avoids lengthy responses -> can be adjusted to your use case
            temperature = 0, 
            # as the llm generates content, there can be a bit of randomness into it 
            # (some content might not even be related to what you are looking for)

            max_tokens = 1000,
            model_name = "gpt-3.5-turbo" #one of the latest models we have so far
        )

        #output results 
        # chain -> take the question, get relevant documents, pass it to the LLM, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_input)
        st.write(response)