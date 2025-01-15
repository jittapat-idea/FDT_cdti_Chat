import gradio as gr

def chat_with_model(prompt, history):
    # สร้างข้อความตอบกลับจากโมเดล
    response = f"Model Response to: {prompt}"
    history.append((prompt, response))
    return history, ""

# กำหนด Custom CSS
custom_css = """
#chatbox {
    background-color: #f0f0f0;
}

#chatbox .user {
    background-color: #ffa07a !important;  /* สีส้มสำหรับ User */
    padding: 10px;
    border-radius: 10px;
    color: #000000 !important;  /* สีตัวอักษรเป็นสีดำ */
}

#chatbox .bot {
    color: white !important;/* สีตัวอักษรเป็นสีดำ */
    background-color: #404040 !important;    /* สีขาวสำหรับ Bot */
    padding: 10px;
    border-radius: 10px;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# CDTI FDT Chat")
    
    chatbot = gr.Chatbot(
        label="Chat History",
        elem_id="chatbox",
        bubble_full_width=False,  # ทำให้ bubble ไม่เต็มความกว้าง
    )
    state = gr.State([])

    with gr.Column():
        input_box = gr.Textbox(
            placeholder="Type your message here...", 
            label="Your Input",
            elem_id="input_box",
        )
        send_button = gr.Button("Send", elem_id="send_button")

    send_button.click(
        fn=chat_with_model,
        inputs=[input_box, state],
        outputs=[chatbot, input_box]
    )

demo.launch()