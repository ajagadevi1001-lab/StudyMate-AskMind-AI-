!pip install gradio transformers sentencepiece pdfplumber pillow --quiet

import gradio as gr
import pdfplumber
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import io
from PIL import Image

# ----------------------------------------------------------
# Load Granite Model dynamically using HF token from UI
# ----------------------------------------------------------
def load_model(hf_token, model_name="ibm-granite/granite-3.3-2b-instruct"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=hf_token)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
        return pipe, "Model loaded successfully!"
    except Exception as e:
        return None, f"❌ Error loading model: {str(e)}"

# ----------------------------------------------------------
# PDF → Text Extraction
# ----------------------------------------------------------
def extract_pdf_text(pdf_files):
    text = ""
    if not pdf_files:
        return "No PDF uploaded."
    for pdf in pdf_files:
        with pdfplumber.open(pdf) as pdf_doc:
            for page in pdf_doc.pages:
                text += page.extract_text() or ""
    return text if text.strip() else "No readable text found in PDFs."

# ----------------------------------------------------------
# Chatbot (PDF Q&A)
# ----------------------------------------------------------
def study_chat(q, pdf_text, hf_token):
    if not hf_token:
        return "❌ Please enter your HuggingFace token."

    if not pdf_text.strip():
        return "❌ Please extract text from PDF first."

    pipe, msg = load_model(hf_token)
    if pipe is None:
        return msg

    prompt = (
        "You are StudyMate, an AI assistant. Use ONLY the following study material:\n\n"
        f"{pdf_text}\n\n"
        f"Question: {q}\n"
        "Answer clearly and accurately using info ONLY from the above content."
    )

    output = pipe(prompt, max_length=300)
    return output[0]["generated_text"]

# ----------------------------------------------------------
# Translation (Indian languages)
# ----------------------------------------------------------
def translate_text(text, target_lang, hf_token):
    lang_map = {
        "Hindi": "hi", "Tamil": "ta", "Telugu": "te",
        "Kannada": "kn", "Malayalam": "ml", "Bengali": "bn",
        "Gujarati": "gu", "Marathi": "mr", "Punjabi": "pa"
    }

    if not hf_token:
        return "❌ Please enter your HuggingFace token."

    pipe, msg = load_model(hf_token)
    if pipe is None:
        return msg

    tgt = lang_map[target_lang]
    prompt = f"Translate the following text into {target_lang} ({tgt}):\n{text}"

    output = pipe(prompt, max_length=200)
    return output[0]["generated_text"]

# ----------------------------------------------------------
# Image Analysis
# ----------------------------------------------------------
def analyze_image(img, hf_token):
    if img is None:
        return "No image uploaded."

    if not hf_token:
        return "❌ Please enter your HuggingFace token."

    pipe, msg = load_model(hf_token)
    if pipe is None:
        return msg

    prompt = "Describe the content of this image in detail (objects, scene, meaning)."

    output = pipe(prompt, max_length=200)
    return output[0]["generated_text"]

# ----------------------------------------------------------
# Gradio Interface
# ----------------------------------------------------------
with gr.Blocks(title="StudyMate AI") as demo:

    gr.Markdown("# 📚 StudyMate – AI Academic Assistant")
    gr.Markdown("Upload PDFs → Ask Questions → Translate Text → Analyze Images")

    hf_token = gr.Textbox(label="🔑 Enter Your HuggingFace Token", type="password")

    with gr.Tabs():
        # ---------------- CHATBOT TAB ----------------
        with gr.Tab("📘 Chat with PDFs"):
            pdf_upload = gr.File(label="Upload PDF(s)", file_count="multiple", type="filepath")
            extract_btn = gr.Button("Extract Text")
            pdf_textbox = gr.Textbox(label="Extracted PDF Text", lines=10)

            q = gr.Textbox(label="Ask your question")
            ask_btn = gr.Button("Ask StudyMate")
            chat_output = gr.Textbox(label="StudyMate Answer", lines=6)

            extract_btn.click(extract_pdf_text, inputs=pdf_upload, outputs=pdf_textbox)
            ask_btn.click(study_chat, inputs=[q, pdf_textbox, hf_token], outputs=chat_output)

        # ---------------- TRANSLATION TAB ----------------
        with gr.Tab("🌐 Translate Text"):
            text_input = gr.Textbox(label="Enter text to translate")
            lang = gr.Dropdown(
                ["Hindi", "Tamil", "Telugu", "Kannada", "Malayalam",
                 "Bengali", "Gujarati", "Marathi", "Punjabi"],
                label="Select Language"
            )
            trans_btn = gr.Button("Translate")
            trans_output = gr.Textbox(label="Translated Text")

            trans_btn.click(translate_text, inputs=[text_input, lang, hf_token], outputs=trans_output)

        # ---------------- IMAGE ANALYSIS TAB ----------------
        with gr.Tab("🖼 Image Analysis"):
            img = gr.Image(label="Upload an Image")
            analyze_btn = gr.Button("Analyze Image")
            img_output = gr.Textbox(label="Image Description")

            analyze_btn.click(analyze_image, inputs=[img, hf_token], outputs=img_output)

demo.launch()
