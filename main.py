import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from tavily import AsyncTavilyClient

app = FastAPI(title="Kona AI Free Groq Engine", version="1.1.0")

# Enable open CORS security so your GitHub Pages frontend can securely fetch the stream
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# Pulling secure keys from Hugging Face Settings -> Variables and secrets
# NOTE: Your OPENAI_API_KEY slot should now hold your 'gsk_...' Groq API key!
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Gracefully initialize clients during build startup sequence
if not OPENAI_API_KEY or not TAVILY_API_KEY:
    print("WARNING: System environment keys are currently missing from the Hugging Face secret panel.")
    openai_client = None
    tavily_async_client = None
else:
    # We redirect the official OpenAI SDK client to route traffic straight to Groq's free cloud infrastructure
    openai_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=OPENAI_API_KEY
    )
    tavily_async_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)

class SearchPayload(BaseModel):
    message: str

async def search_and_stream(user_prompt: str):
    if not openai_client or not tavily_async_client:
        yield f"data: {json.dumps({'type': 'error', 'message': 'API keys are missing in Hugging Face settings.'})}\n\n"
        return

    web_context = ""
    sources_payload = []

    try:
        # Step 1: Query live internet data indices using Tavily
        search_result = await tavily_async_client.search(query=user_prompt, max_results=4)
        results = search_result.get('results', [])
        
        for idx, item in enumerate(results):
            source_idx = idx + 1
            web_context += f"[{source_idx}] Source: {item['url']}\nContent: {item['content']}\n\n"
            sources_payload.append({
                "id": source_idx,
                "title": item.get('title', 'Web Resource'),
                "url": item['url']
            })
    except Exception as search_err:
        print(f"Search exception caught: {str(search_err)}")
        web_context = "Live web indexes are temporarily unreachable."

    # Immediately push parsed source citation links to the UI viewport layer
    yield f"data: {json.dumps({'type': 'metadata', 'sources': sources_payload})}\n\n"
    await asyncio.sleep(0.01)

    try:
        # Step 2: Establish connection to Groq's free LLM framework with multilingual guardrails
        system_rules = (
            "You are Kona AI, an advanced AI search engine and real-time data synthesis engine.\n"
            "Generate deeply factual, detailed, clear, and comprehensive responses based on the provided live context data.\n\n"
            "CRITICAL LANGUAGE RULE:\n"
            "Identify the language or regional Indian dialect used by the user in their prompt (e.g., English, Hindi, Telugu, Tamil, Bengali, Marathi, etc.). "
            "You MUST write your entire response and structural reasoning output in that EXACT same language and native script. "
            "Do not use Latin transliteration for regional Indian scripts; use their proper native alphabets (e.g., Devanagari, Telugu script, etc.).\n\n"
            "Integrate precise citation anchor numbers inline using bracket formatting like [1] or [2] to match the facts directly back to their web sources."
        )

        llm_stream = openai_client.chat.completions.create(
            model="llama3-70b-8192",  # Elite, ultra-fast, highly intelligent model hosted on Groq for free
            messages=[
                {"role": "system", "content": f"{system_rules}\n\nContext Data:\n{web_context}"},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2, # Keeps the synthesis strictly grounded in the web data facts
            stream=True
        )

        # Step 3: Yield structured text token frames onto the client connection hook
        for chunk in llm_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield f"data: {json.dumps({'type': 'token', 'text': chunk.choices[0].delta.content})}\n\n"
                await asyncio.sleep(0.002)

    except Exception as llm_err:
        yield f"data: {json.dumps({'type': 'error', 'message': f'LLM routing anomaly: {str(llm_err)}'})}\n\n"

@app.post("/api/v1/search")
async def execute_search(payload: SearchPayload):
    return StreamingResponse(search_and_stream(payload.message), media_type="text/event-stream")

# Standard production binding configuration for Hugging Face container deployments
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)
