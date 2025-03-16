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
    retriever = contextual_vector.as_retriever(search_kwargs={"k": 3})
    # Load vectorstores
    # contextual_vector_results = contextual_vector.similarity_search(prompt, k=3)
    # contextual_vector_answer = model.generate_answer_api(
    #     prompt, [doc.page_content for doc in contextual_vector_results]
    # )

    # Include the selected dropdown value in the response
    # response = f"Selected Option: {dropdown_value}\n\nResponse: {contextual_vector_answer}"
    # history.append((prompt, response))
    # return history, ""
    answer, history = model.generate_answer_api_with_history(prompt, retriever=retriever)
    return answer, history

    # with open("/home/s6410301020/SeniorProject/FDT_cdti_Chat/digital_doc/baseline.md", "r") as file:
    #         contexts = file.read()
    # answer_question = model.create_question(prompt)
    # # answer = model.generate_answer_api_dynamic_with_history(answer_question.content, contexts)
    # return answer_question

response, history = chat_with_model("สวัสดีครับ, ลงทะเบียนเรียนต้องเข้าเว็บไหนครับ")
print("Question 1:")
print(response)
print("-"*50)
print(history)
print("-"*50)
print("Question 2:")
response = chat_with_model("แล้วต้องทำยังไงหรอครับ")
print(response)
print("-"*50)
print(history)