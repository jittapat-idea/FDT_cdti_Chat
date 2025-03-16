import os
import gradio as gr
from dotenv import dotenv_values
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from ContextualClass.contextual import ContextualRetrieval
from intent.prompt_template import RAG_PROMPT_TEMPLATES, DYNAMIC_PROMPT_TEMPLATE
from intent.intent import (
    initialize_model,
    predict_intent
)

# Load API key from environment file
config = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = config["openai_api"]

# Define directories for vectorstores
og_data = "./doc_Data/original"
context_data = "./doc_Data/context"



model = ContextualRetrieval()
tokenizer, intent_model = initialize_model()

# def chat_with_model(prompt, history, selected_option):
#     # Load vectorstores
#     contextual_vector = Chroma(persist_directory=context_data, embedding_function=OpenAIEmbeddings(), collection_name=f"cdti_doc")

#     contextual_vector_results = contextual_vector.similarity_search(prompt, k=6)
#     contextual_vector_answer = model.generate_answer_api(prompt, [doc.page_content for doc in contextual_vector_results])

#     # Include the selected dropdown option in the response
#     response = f"[{selected_option}] {contextual_vector_answer}"
#     history.append((prompt, response))
#     return history, ""

def intention(prompt: str):
    try:        
        # Initialize the model when the application starts. Load the model and tokenizer
        # tokenizer, intent_model = initialize_model()

        intent = predict_intent(prompt, tokenizer, intent_model)
        
        return intent
    except Exception as e:
        print(e)
        return e

def chat_with_model_history(prompt, history):
    dynamic_doc = ["academic_calendar", "student_activities"]

    # intent = intention(prompt=prompt)
    # print(intent)
    # if intent in dynamic_doc:
    #     if intent == "academic_calendar":
    #         with open("D:\MyProject\FDT_cdti_Chat\doc\doc_test.md", "r") as file:
    #             contexts = file.read()
    #         print(contexts)

    #         system_prompt = DYNAMIC_PROMPT_TEMPLATE(intent=intent, context=contexts)
    #         answer, _ = model.generate_answer_api_dynamic_with_history(prompt, system_prompt=system_prompt)
    #         response = f"[{intent}] {answer.content}]"
    #     else:
    #         with open("D:\MyProject\FDT_cdti_Chat\doc\doc_test.md", "r") as file:
    #             contexts = file.read()
    #         print(contexts)

    #         system_prompt = DYNAMIC_PROMPT_TEMPLATE(intent=intent, context=contexts)
    #         answer, _ = model.generate_answer_api_dynamic_with_history(prompt, system_prompt=system_prompt)
    #         response = f"[{intent}] {answer.content}]"

    # else:
    #     system_prompt = RAG_PROMPT_TEMPLATES[intent]
    #     contextual_vector = Chroma(persist_directory=context_data, embedding_function=OpenAIEmbeddings(), collection_name=f"cdti_doc")
    #     retriever = contextual_vector.as_retriever(search_kwargs={"k": 6})

    #     answer, _ = model.generate_answer_api_with_history(prompt, retriever=retriever, system_prompt=system_prompt)

    #     # Include the selected dropdown option in the response
    #     response = f"[{intent}] {answer["answer"]}"

    response = "Hello World"


    history.append((prompt, response))
    return history, ""

# Custom CSS
custom_css = """
#chatbox {
    background-color: #f0f0f0;
}

#chatbox .user {
    background-color: #ffa07a !important;
    padding: 10px;
    border-radius: 10px;
    color: #000000 !important;
}

#chatbox2 .user {
    background-color: #6495ED !important;
    padding: 10px;
    border-radius: 10px;
    color: #FFFFFF !important;
}

#chatbox .bot {
    color: white !important;
    background-color: #FFFFFF !important;
    padding: 10px;
    border-radius: 10px;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# CDTI FDT Chat")

    with gr.Row():
        # chatbot = gr.Chatbot(
        #     label="Chat History",
        #     elem_id="chatbox",
        #     bubble_full_width=False,
        # )

        chatbot_history = gr.Chatbot(
            label="Chat History",
            elem_id="chatbox2"
        )

    # state1 = gr.State([])
    state2 = gr.State([])

    with gr.Row():
        # input_box1 = gr.Textbox(
        #     placeholder="Type your message here...", 
        #     label="Your Input",
        #     elem_id="input_box",
        # )

        # dropdown1 = gr.Dropdown(
        #     choices=["course", "capital", "general", "academic_calendar", "student_activities"],
        #     label="Select an Option",
        #     elem_id="dropdown"
        # )
        

        input_box2 = gr.Textbox(
            placeholder="Type your message here...", 
            label="Your Input",
            elem_id="input_box2",
        )

        dropdown2 = gr.Dropdown(
            choices=["course", "capital", "general", "academic_calendar", "student_activities"],
            label="Select an Option",
            elem_id="dropdown2"
        )
        
    # send_button1 = gr.Button("Send_no_history", elem_id="send_button1")
    send_button2 = gr.Button("Send_history", elem_id="send_button/")

    # send_button1.click(
    #     fn=chat_with_model,
    #     inputs=[input_box1, state1, dropdown1],
    #     outputs=[chatbot, input_box1]
    # )

    send_button2.click(
        fn=chat_with_model_history,
        inputs=[input_box2, state2],
        outputs=[chatbot_history, input_box2]
    )

demo.launch()
