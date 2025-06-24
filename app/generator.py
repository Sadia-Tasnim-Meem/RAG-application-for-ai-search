from transformers import pipeline

qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")

def generate_answer(question, context):
    prompt = f"Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {question}"

    result = qa_pipeline(prompt, max_length=128, do_sample=False)
    return result[0]['generated_text']