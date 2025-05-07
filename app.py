from flask import Flask, render_template, request, jsonify
from langchain.llms import Cohere  # Using Cohere instead of ChatCohere
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")

app = Flask(__name__)

llm = Cohere()

# Define the prompt template for a chat-like conversation
chat_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],  # Input and conversation history variables
    template="""
You are a helpful assistant that responds to user questions in a clear and friendly manner.

Conversation history:
{chat_history}

User: {input}
Assistant:"""
)

# Memory to store conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Conversation chain that coordinates between the model and memory
conversation = ConversationChain(
    llm=llm,                  # Language model (Cohere)
    prompt=chat_prompt,       # The prompt that guides the model
    memory=memory,            # The memory that keeps track of the conversation history
    verbose=False             # Disables detailed logging
)

# Function to get the answer from the chatbot
def answer_as_chatbot(message):
    return conversation.predict(input=message)

def load_db():
    try:
        embeddings = CohereEmbeddings(cohere_api_key=os.environ["COHERE_API_KEY"])
        vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
        qa = RetrievalQA.from_chain_type(
            llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )
        return qa
    except Exception as e:
        print("Error:", e)

qa = load_db()

def answer_from_knowledgebase(message):
    res = qa({"query": message})
    return res['result']

def search_knowledgebase(message):
    res = qa({"query": message})
    sources = ""
    for count, source in enumerate(res['source_documents'],1):
        sources += "Source " + str(count) + "\n"
        sources += source.page_content + "\n"
    return sources

@app.route('/kbanswer', methods=['POST'])
def kbanswer():
    message = request.json['message']
    response_message = answer_from_knowledgebase(message)
    return jsonify({'message': response_message}), 200

@app.route('/search', methods=['POST'])
def search():    
    message = request.json['message']
    response_message = search_knowledgebase(message)
    return jsonify({'message': response_message}), 200

@app.route('/answer', methods=['POST'])
def answer():
    message = request.json['message']
    response_message = answer_as_chatbot(message)
    return jsonify({'message': response_message}), 200

@app.route("/")
def index():
    return render_template("index.html", title="Chatbot")

if __name__ == "__main__":
    app.run(debug=True)
