from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your valid Groq API key
client = Groq(api_key="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")

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Updated to a supported model
            messages=[
                {"role": "system", "content": "You are a helpful construction assistant."},
                {"role": "user", "content": req.message},
            ],
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        return {"reply": f"⚠️ Backend error: {str(e)}"}
