from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub


def create_vectorstore(chunks):
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="WhereIsAI/UAE-Large-V1")
    new_vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    vectorstore = FAISS.load_local('faiss_default', embeddings)
    vectorstore.merge_from(new_vectorstore)

    return vectorstore

def create_conversation_chain(vectorstore=None):

    if(not vectorstore):
        embeddings = HuggingFaceInstructEmbeddings(model_name="WhereIsAI/UAE-Large-V1")
        vectorstore = FAISS.load_local('faiss_default', embeddings)

    #llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id='google/flan-t5-large', model_kwargs={"max_length":512, "temperature":0.1})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain