import gradio as gr

# ฟังก์ชันโหลดโมเดล (แก้ไขตามที่จำเป็น)
def load_model():
    return "your_model"

model = load_model()

# ฟังก์ชันสนทนา
def chat(user_input, chat_history):
    """
    รับข้อความจากผู้ใช้และประวัติการสนทนา แล้วตอบกลับ
    """
    chat_history = chat_history or []  # เริ่มประวัติสนทนาใหม่ถ้ายังไม่มี
    chat_history.append(("User", user_input))  # เพิ่มข้อความผู้ใช้ในประวัติสนทนา

    # ใช้โมเดลสร้างคำตอบ
    model_response = f"Model Response to: {user_input}"  # เปลี่ยนเป็นคำตอบจากโมเดลของคุณ
    chat_history.append(("Bot", model_response))  # เพิ่มคำตอบในประวัติสนทนา

    return chat_history, chat_history

# สร้าง UI Gradio
with gr.Blocks() as demo:
    gr.Markdown("## CDTI FDT Chat")

    # ส่วนแสดงประวัติการสนทนา
    chatbox = gr.Chatbot(label="Chat History")  # ไม่มี .style()

    # ช่องสำหรับเขียนข้อความและปุ่มส่ง
    with gr.Row():
        user_input = gr.Textbox(
            show_label=False, placeholder="Type your message here..."
        )
        btn_submit = gr.Button("Send")

    # สถานะสำหรับเก็บประวัติการสนทนา
    chat_history = gr.State()

    # ผูกปุ่มส่งกับฟังก์ชันสนทนา
    btn_submit.click(chat, inputs=[user_input, chat_history], outputs=[chatbox, chat_history])

    # รองรับการกด Enter เพื่อส่งข้อความ
    user_input.submit(chat, inputs=[user_input, chat_history], outputs=[chatbox, chat_history])

# เปิดใช้งานแอป
demo.launch()
