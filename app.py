# app.py
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from pydantic import BaseModel
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
import uuid, asyncio, os
from memory_store import MemoryStore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import uvicorn

app = FastAPI(title="ITAP MVP")

# Initialize components
#  - Use small/or quick models for demo; you can swap model names as needed
sentiment_pipeline = pipeline("sentiment-analysis")  # HF default small model
summarizer = pipeline("summarization")                # switch to a small summarizer if needed
qa_pipeline = pipeline("question-answering")          # QA
generator = pipeline("text-generation", model="distilgpt2", max_length=150)  # fast generator
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ROMANCE")   # example
vader = SentimentIntensityAnalyzer()
memory = MemoryStore()

# --- Data models
class ChatRequest(BaseModel):
    user_id: str
    message: str

# Utility to build context
def build_context(user_id: str, message: str, k=3):
    # fetch nearest memories
    n = memory.nearest(message, top_k=k)
    ctx = "\n".join([f"{r[1][1]}: {r[1][2]}" for r in n]) if n else ""
    return ctx

@app.post("/chat")
async def chat(req: ChatRequest):
    # store user message
    uid = req.user_id or str(uuid.uuid4())
    memory.add(uid, "user", req.message)
    # build context
    ctx = build_context(uid, req.message)
    prompt = f"Context:\n{ctx}\n\nUser: {req.message}\nAssistant:"
    # simple generation
    gen = generator(prompt, max_length=200, do_sample=True, num_return_sequences=1)[0]['generated_text']
    reply = gen.split("Assistant:")[-1].strip()
    memory.add(uid, "assistant", reply)
    return {"user_id": uid, "reply": reply}

@app.post("/sentiment")
async def sentiment(text: str = Form(...)):
    # fast rule-based + model
    vader_scores = vader.polarity_scores(text)
    hf = sentiment_pipeline(text)
    return {"vader": vader_scores, "hf": hf}

@app.post("/summarize")
async def summarize(text: str = Form(...)):
    # summarization (truncate long texts for pipeline)
    if len(text.split()) > 800:
        text = " ".join(text.split()[:800])
    summary = summarizer(text, max_length=120, min_length=30, do_sample=False)
    return {"summary": summary[0]['summary_text']}

@app.post("/qa")
async def qa(question: str = Form(...), context: str = Form(...)):
    res = qa_pipeline(question=question, context=context)
    return res

@app.post("/generate")
async def generate(prompt: str = Form(...), max_len: int = Form(120)):
    gen = generator(prompt, max_length=max_len, do_sample=True, num_return_sequences=1)
    return {"output": gen[0]['generated_text']}

@app.post("/translate")
async def translate(text: str = Form(...), target_lang: str = Form("fr")):
    # This endpoint is a demo. Use appropriate model for target_lang mapping in production.
    res = translator(text)
    return {"translation": res[0]['translation_text']}

# Simple real-time ingestion using WebSocket
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # data is text chunk; run sentiment and summarization quick
            sent = vader.polarity_scores(data)
            # respond back with realtime sentiment
            await websocket.send_json({"text": data, "vader": sent})
    except Exception as e:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
