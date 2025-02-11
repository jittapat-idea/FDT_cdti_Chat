import os
import pandas as pd
from dotenv import dotenv_values
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from ContextualClass.contextual import ContextualRetrieval

# Load API key from environment file
config = dotenv_values(".env")
os.environ["OPENAI_API_KEY"] = config["openai_api"]

# Initialize the Contextual Retrieval system
cr = ContextualRetrieval()

# Define directories for vectorstores
og_data = "./doc_Data/original"
context_data = "./doc_Data/context"

# Load vectorstores
original_vector = Chroma(persist_directory=og_data, embedding_function=OpenAIEmbeddings())
contextual_vector = Chroma(persist_directory=context_data, embedding_function=OpenAIEmbeddings())

# Define queries
# queries = [
#     "ชื่อปริญญาและสาขาวิชาของหลักสูตรนี้คืออะไร ผมอยากได้ทั้งชื่อภาษาไทยและภาษาอังกฤษ",
#     "ผมอยากทราบ วัน เวลาในการดำเนินการเรียนการสอน",
#     "จำนวนหน่วยกิตที่ต้องเรียนทั้งหมด ต้องเรียนทั้งหมดกี่หน่วยกิต",
#     "รายวิชาของกลุ่มวิชาสังคมศาสตร์ มีรายวิชาอะไรบ้าง",
#     "อยากทราบ แผนการศึกษา ของคณะและสาขา ปีที่ 3 ภาคการศึกษาที่ 2 ",
#     "ช่วยอธิบายรายวิชาของ วิชาโครงสร้างของระบบคอมพิวเตอร์ ให้หน่อย",
#     "อยากทราบขั้นตอน เกณฑ์การสำเร็จการศึกษา"
# ]
df = pd.read_excel('/home/s6410301020/SeniorProject/Contextual-Retrieval/ExamQuestion.xlsx')
queries = df["question"]

count = 1

# Open a file to save results
with open("query_results_v2.txt", "w", encoding="utf-8") as file:
    for query in queries:
        file.write(f"\n{count}. Query: {query}\n")
        print(f"\n{count}. Query: {query}")

        # Retrieve from Original Vectorstore
        original_vector_results = original_vector.similarity_search(query, k=2)

        # Retrieve from Contextual Vectorstore
        contextual_vector_results = contextual_vector.similarity_search(query, k=2)

        # Generate answers
        original_vector_answer = cr.generate_answer(query, [doc.page_content for doc in original_vector_results])
        contextual_vector_answer = cr.generate_answer(query, [doc.page_content for doc in contextual_vector_results])

        # Write Original Vectorstore results to file
        file.write("\nOriginal Vector Search Result:\n")
        print("\nOriginal Vector Search Result:")
        for i, doc in enumerate(original_vector_results, 1):
            result_excerpt = doc.page_content
            file.write(f"Context: {i}. {result_excerpt}\n")
            print(f"Context: {i}. {result_excerpt}")

        file.write("\nOriginal Vector Search Answer:\n")
        file.write(original_vector_answer + "\n")
        file.write("\n" + "-"*50)
        print("\nOriginal Vector Search Answer")
        print(original_vector_answer)
        print("\n" + "-"*50)

        # Write Contextual Vectorstore results to file
        file.write("\nContextual Vector Search Result:\n")
        print("\nContextual Vector Search Result:")
        for i, doc in enumerate(contextual_vector_results, 1):
            result_excerpt = doc.page_content
            file.write(f"Context: {i}. {result_excerpt}\n")
            print(f"Context: {i}. {result_excerpt}")

        file.write("\nContextual Vector Search Answer:\n")
        file.write(contextual_vector_answer + "\n")
        file.write("\n" + "-"*50)
        print("\nContextual Vector Search Answer")
        print(contextual_vector_answer)
        print("\n" + "-"*50)

        count = count + 1

print("Results saved to query_results.txt")
