import os
import gradio as gr
from dotenv import dotenv_values
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from ContextualClass.contextual import ContextualRetrieval

# Load API key from environment file
config = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = config["openai_api"]

# Define directories for vectorstores
og_data = "./doc_Data/original"
context_data = "./doc_Data/context"

model = ContextualRetrieval()
contextual_vector = Chroma(persist_directory=context_data, embedding_function=OpenAIEmbeddings(), collection_name=f"general")

def chat_with_model(prompt):
    with open("/home/s6410301020/SeniorProject/FDT_cdti_Chat/digital_doc/baseline.md", "r") as file:
            contexts = file.read()
    answer, _ = model.generate_answer_api_dynamic_with_history(prompt, contexts)
    
    return answer

response = chat_with_model("วันที่ 16 มกรา มีกิจกรรมอะไรครับ")
print("Question 1:")
print(response.content)
print("-"*50)
# print(history)
# print("-"*50)
# print("Question 2:")
# response = chat_with_model("แล้วต้องทำยังไงหรอครับ")
# print(response.content)
# print("-"*50)
# print(history)