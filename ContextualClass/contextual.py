import os
# import torch
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from rank_bm25 import BM25Okapi
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import time

from dotenv import dotenv_values
config = dotenv_values(".env")

os.environ["OPENAI_API_KEY"] = config["openai_api"]
typhoon_api = config["Typhoon_API"]

# def init_model():
#     if torch.cuda.is_available():
#         print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
#     else:
#         print("No GPU available. Training will run on CPU.")
   
#     # quantization_config = BitsAndBytesConfig(
#     # load_in_8bit=True,
#     # # bnb_4bit_quant_type="nf4",
#     # # bnb_4bit_compute_dtype="float16",
#     # # bnb_4bit_use_double_quant=True,
#     # )

#     llm = HuggingFacePipeline.from_model_id(
#         model_id="scb10x/llama3.1-typhoon2-8b-instruct",
#         device_map="auto",
#         task="text-generation",
#         pipeline_kwargs=dict(
#             max_new_tokens=512,
#             do_sample=True,
#             temperature=0.1,
#             return_full_text=False,
#         )
#         # model_kwargs={"quantization_config": quantization_config},
#     )

#     chat_model = ChatHuggingFace(llm=llm)

#     return chat_model

class ContextualRetrieval:
    """
    A class that implements the Contextual Retrieval system.
    """

    def __init__(self):
        """
        Initialize the ContextualRetrieval system.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, 
            chunk_overlap=250,
            length_function=len,
            add_start_index=True
        )
        self.embeddings = OpenAIEmbeddings()
        # self.llm = init_model()

    
        self.typhoon_api = ChatOpenAI(base_url='https://api.opentyphoon.ai/v1',
                            model='typhoon-v2-70b-instruct',
                            api_key=typhoon_api,
                            max_tokens=1024)

        self.store = {}
        self.count = 0
        
    def process_document(self, document: str) -> Tuple[List[Document], List[Document]]:
        """
        Process a document by splitting it into chunks and generating context for each chunk.
        """
        chunks = self.text_splitter.split_documents([document])
        print(f"Split {len(chunks)} Chunks Successful.")
        contextualized_chunks = self._generate_contextualized_chunks(document, chunks)
        print("Generate Context Chuncks Successful")
        return chunks, contextualized_chunks
    
    def _generate_contextualized_chunks(self, document: str, chunks: List[Document]) -> List[Document]:
        """
        Generate contextualized versions of the given chunks.
        """
        contextualized_chunks = []
        count = 1
        for chunk in chunks:
            context = self._generate_context(document, chunk.page_content)
            contextualized_content = f"{context}\n\n{chunk.page_content}"
            contextualized_chunks.append(Document(page_content=contextualized_content, metadata=chunk.metadata))
            print(f"Chunk {count} Complete")
            count = count + 1
            time.sleep(1)
        return contextualized_chunks
    
    def _generate_context(self, document: str, chunk: str) -> str:
        """
        Generate context for a specific chunk using the language model.
        """
        prompt = ChatPromptTemplate.from_template("""
        คุณเป็นผู้ช่วยที่เชี่ยวชาญในการตอบคำถามเกี่ยวกับเอกสารของสถาบันเทคโนโลยีจิตรลดา (CDTI) งานของคุณคือการให้บริบทที่สรุปและเฉพาะเจาะจงสำหรับข้อความตอนหนึ่งจากเอกสาร
                                                  
        <document> 
        {document} 
        </document>

        นี่คือส่วนย่อยที่เราต้องการจัดวางในบริบทของเอกสารทั้งหมด

        <chunk>
        {chunk} 
        </chunk>

        กรุณาให้บริบทสั้นๆ และกระชับเพื่อจัดวางส่วนย่อยนี้ในเอกสารทั้งหมด โดยมีจุดประสงค์เพื่อปรับปรุงการค้นหาและเรียกคืนส่วนย่อยนี้ กรุณาตอบเฉพาะบริบทที่กระชับเท่านั้น ไม่ต้องเพิ่มข้อมูลอื่นๆ
        แล้วสอดคล้องในแต่ละ Header เช่น ใน Chunk มีการกล่างถึง เอกสาร
                                   
        Context:
        """)
        messages = prompt.format_messages(document=document, chunk=chunk)
        response = self.llm.invoke(messages)
        return response.content
    
    def create_vectoDB(self, chunks: List[Document], path: str) -> Chroma:
        """
        Create a vector DB for the given chunks
        """
        data = f"./doc_Data/{path}"
        vectordb = Chroma.from_documents(chunks, embedding=OpenAIEmbeddings(), persist_directory=data, collection_name="capital")
        vectordb.persist()

    def create_bm25_index(self, chunks: List[Document]) -> BM25Okapi:
        """
        Create a BM25 index for the given chunks.
        """
        tokenized_chunks = [chunk.page_content.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)
    
    # def generate_answer(self, query: str, relevant_chunks: List[str]) -> str:
    #     prompt = ChatPromptTemplate.from_template("""
    #     คุณเป็นผู้ช่วยในการตอบคำถาม ในคณะเทคโนโลยีดิจิทัล คุณจะตอบคำถามตอบข้อมูลใน Context ที่ได้รับ โดยคุณจะสร้างคำตอบที่เข้าใจง่ายต่อผู้ใช้ ถ้าอะไรที่คุณไม่ทราบ
    #     คุณก็จะต้องบอกว่าคุณไม่ทราบ แล้วให้ติดต่อเจ้าหน้าที่

    #     Context: {chunks}
                                                  
    #     Question: {query}
        

    #     Answer:
    #     """)
        # messages = prompt.format_messages(query=query, chunks="\n\n".join(relevant_chunks))
        # response = self.llm.invoke(messages)
        # return response.content

    def history_aware_retriever_func(self, retriever):
        contextualize_q_system_prompt = """
        ### 🔹 **Context (บริบท)**
        - คุณเป็นระบบที่ช่วยสร้างคำถามใหม่จากประวัติการแชทของผู้ใช้  
        - คุณต้องสร้างคำถามที่สามารถเข้าใจได้โดยไม่ต้องมีประวัติการแชท  
        - ห้ามคาดเดาหรือสร้างข้อมูลขึ้นมาเอง  

        ### 🎯 **Objective (เป้าหมาย)**
        - เมื่อได้รับ **ประวัติการแชท (chat history)** และ **คำถามล่าสุดของผู้ใช้**  
        - หากคำถามอ้างอิงถึง **Context** ในประวัติการแชท **ให้สร้างคำถามแบบแยกเดี่ยว** ที่เข้าใจได้โดยไม่ต้องมีบริบทเดิม  
        - ห้ามตอบคำถามนั้น เพียงแค่สร้างคำถามใหม่และส่งคืน  

        ### 🎨 **Style & Tone (รูปแบบและโทนของคำถาม)**  
        - ใช้ภาษาที่เป็น **ทางการแต่เข้าใจง่ายและเป็นกันเอง**  
        - คำถามต้อง **ตรงประเด็น** และไม่คลุมเครือ  
        - ถ้าคำถามเกี่ยวกับ **วัน/เดือน/ปี** ห้ามเปลี่ยนแปลงข้อมูล  
        - ถ้าคำถามเดิมเข้าใจได้อยู่แล้ว ให้คืนคำถามเดิม  

        ### 📜 **Output (รูปแบบคำตอบที่ต้องการ)**
        - **คำถามใหม่ที่สามารถเข้าใจได้โดยไม่ต้องมีประวัติการแชท**  
        - ถ้าคำถามเดิมชัดเจนอยู่แล้ว ให้คืนคำถามเดิม  


        """
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.typhoon_api, retriever, contextualize_q_prompt
        )

        return history_aware_retriever

    def generate_answer_api_with_history(self, query: str, system_prompt, retriever):

        prompt = system_prompt

        qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
        )

        question_answer_chain = create_stuff_documents_chain(self.typhoon_api, qa_prompt)

        history_aware_retriever = self.history_aware_retriever_func(retriever=retriever)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            self.count += 1
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            
            if self.count > 3:
                self.store[session_id] = ChatMessageHistory()
                self.count = 0

            return self.store[session_id]


        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        response = conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": "abc123"}
            }, 
        )
        # response = self.typhoon_api.invoke(messages)
        print(self.store)
        return response, self.store

    def generate_answer_api_dynamic_with_history(self, query: str, system_prompt: str):

        prompt = system_prompt

        qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
        )
        
        conversation_chain = qa_prompt | self.typhoon_api

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            self.count += 1
            if session_id not in self.store:
                self.store[session_id] = ChatMessageHistory()
            
            if self.count > 3:
                self.store[session_id] = ChatMessageHistory()
                self.count = 0

            return self.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            conversation_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        response = conversational_rag_chain.invoke(
            {"input": query},
            config={"configurable": {"session_id": "abc123"}
            }, 
        )

        print(self.store)
        return response, self.store