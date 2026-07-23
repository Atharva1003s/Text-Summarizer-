from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re 
import os
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# Stick strictly to CPU on Cloud deployment to save RAM
device = torch.device("cpu")

# Lazy loading variables
model = None
tokenizer = None

def get_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model").to(device)
        tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")
    return model, tokenizer

templates = Jinja2Templates(directory=".")

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()

def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)
    model_inst, tokenizer_inst = get_model()

    inputs = tokenizer_inst(
        dialogue,
        padding=True,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        targets = model_inst.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=1,
            do_sample=False,
            early_stopping=True
        )
    
    summary = tokenizer_inst.decode(targets[0], skip_special_tokens=True)
    return summary

# API Endpoints
@app.post("/summarize")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
