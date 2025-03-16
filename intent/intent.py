from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json

model_name = "proideas/CDTI-intent-classification"

# โหลดโมเดล
def load_intent_classifier():
    """
    Load a sequence classification model and tokenizer from Hugging Face Hub.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()  # Set model to evaluation mode
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model from Hugging Face Hub: {e}")

# โหลด label mapping
def predict_intent(prompt, tokenizer, model):
    """
    Predict the intent of a given prompt using a pretrained model.
    """
    intent_mapping = ["Scholarship", "academic_calendar", "course", "general_question", "student_activities"]
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax().item()

    return intent_mapping[predicted_class]


def handle_intent(intent, prompt):
    """
    return intent and prompt to User.
    """
    if intent in ["academic_calendar", "student_activities"]:
        return f"{intent}\n{prompt}"
    elif intent in ["Scholarship", "course", "general_question"]:
        return f"{intent}\n{prompt}"
    

#สำหรับโหลดโมเดลเมื่อเริ่มแอป
def initialize_model():
    """
    Initialize the model when the application starts.
    Returns the tokenizer and model for global use.
    """
    return load_intent_classifier()

def main():

    tokenizer, model = initialize_model()
    prompt = input("กรุณากรอกคำถามของคุณ: ")
    intent = predict_intent(prompt,tokenizer,model)
    print(f"[Intent]:{intent}")
    response = handle_intent(intent, prompt)
    print(f"[Response]: {response}")
    # user_prompts = [
    #     "รายละเอียดเกี่ยวกับเครื่องมือที่เน้นในการเรียนการสอนในหลักสูตรนี้คืออะไร?",
    #     "ช่วยบอกวัตถุประสงค์หลักในการจัดทำหลักสูตรนี้?",
    #     "รายวิชา (การออกแบบระบบฝังตัว) เน้นการเรียนรู้ในหัวข้ออะไร?",
    #     "ในปีการศึกษาที่ 2 ภาคการศึกษาที่ 3 มีวิชาเลือกใดที่นักศึกษาเรียนได้?"
    # ]

    # for prompt in user_prompts:
    #     # Step 1: Predict Intent
    #     intent = predict_intent(prompt, tokenizer, model)
    #     print(f"\n[Intent]: {intent}")

    #     # Step 2: Handle Intent
    #     response = handle_intent(intent, prompt)
    #     print(f"[Response]: {response}")

# Run program
if __name__ == "__main__":
    main()