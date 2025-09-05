from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os

# FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (you can restrict later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq client using environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set")

client = Groq(api_key=GROQ_API_KEY)

# Request model
class ChatRequest(BaseModel):
    message: str

# Chat endpoint
@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful construction assistant."},
                {"role": "user", "content": req.message},
            ],
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        return {"reply": f"⚠️ Backend error: {str(e)}"}

# Run with uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
