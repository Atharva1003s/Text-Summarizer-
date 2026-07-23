from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re 
from fastapi.templating import Jinja2Templates # UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")


# model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
# tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")



# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

templates = Jinja2Templates(directory=".")

class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # lines
    text = re.sub(r"\s+", " ", text) # spaces
    text = re.sub(r"<.*?>", " ", text) # html tags <p> <h1>
    text = text.strip().lower()
    return text

# def summarize_dialogue(dialogue : str) -> str:
#     dialogue = clean_data(dialogue) # clean
#     model_inst, tokenizer_inst = get_model()

#     # tokenize
#     inputs = tokenizer(
#         dialogue,
#         padding=True,
#         max_length=512,
#         truncation=True,
#         return_tensors="pt"
#     ).to(device)

#     # generate the summary => token ids
#     with torch.no_grad():
#         targets = model_inst.generate(
#           input_ids=inputs["input_ids"],
#           attention_mask=inputs["attention_mask"],
#           max_length=150,
#           num_beams=1,
#           early_stopping=True,
#           do_sample=False
#     )



def get_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        # Load directly from HuggingFace Hub (downloads automatically on Render)
        model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model").to(device)
        tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")
    return model, tokenizer

def summarize_dialogue(dialogue : str) -> str:
    dialogue = clean_data(dialogue) # clean
    model_inst, tokenizer_inst = get_model()

    # tokenize
    inputs = tokenizer(
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
          early_stopping=True,
          do_sample=False
    )
    
    # decoded our output
    summary = tokenizer_inst.decode(targets[0], skip_special_tokens=True)
    return summary


# API endpoints
@app.post("/summarize")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

# NEW (Fixed)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
