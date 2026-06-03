import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from tavily import AsyncTavilyClient

app = FastAPI(
    title="Kona AI Multilingual Engine",
    version="1.0.0",
    docs_url="/internal/docs"
)

# Production CORS Security Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to match your actual Firebase frontend URL in production
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# System verification for active API keys
OPENAI_API_KEY = os.getenv("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")
TAVILY_API_KEY = os.getenv("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")

if not OPENAI_API_KEY or not TAVILY_API_KEY:
    raise RuntimeError("CRITICAL FAILURE: Environment keys for OPENAI_API_KEY or TAVILY_API_KEY are missing.")

# Instantiate core network clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
tavily_async_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)

class SearchQueryPayload(BaseModel):
    message: str

async def process_and_stream_response(user_prompt: str):
    """
    Asynchronously queries live web indexes and constructs a real-time streaming 
    multilingual synthesis back to the viewport layer.
    """
    web_context = ""
    sources_payload = []

    try:
        # Step 1: Execute optimized live search engine inquiry
        search_result = await tavily_async_client.search(
            query=user_prompt, 
            search_depth="basic", 
            max_results=4
        )
        
        results_list = search_result.get('results', [])
        for idx, item in enumerate(results_list):
            source_index = idx + 1
            web_context += f"[{source_index}] Source: {item['url']}\nContent: {item['content']}\n\n"
            sources_payload.append({
                "id": source_index,
                "title": item.get('title', 'Web Article Source'),
                "url": item['url']
            })
            
    except Exception:
        # Fallback to protect uptime if search engine times out
        web_context = "System warning: Live web scraping interface temporarily offline."
        sources_payload = []

    # Stream structured web source links to the user interface immediately
    yield f"data: {json.dumps({'type': 'metadata', 'sources': sources_payload})}\n\n"
    await asyncio.sleep(0.01)

    try:
        # Step 2: Establish connection to downstream LLM with dynamic multilingual guardrails
        system_rules = (
            "You are Kona AI, an advanced, elite global search engine and data synthesis platform.\n"
            "Generate deeply factual, detailed, clear, and comprehensive responses based on the provided live context data.\n\n"
            "CRITICAL INSTRUCTION FOR MULTILINGUAL CAPACITY:\n"
            "1. Automatically detect the language, script, or regional Indian dialect used by the user in their prompt "
            "(e.g., Hindi, Telugu, Tamil, Bengali, Marathi, Gujarati, Spanish, French, Arabic, etc.).\n"
            "2. You MUST write your entire response, reasoning, and structural output in that EXACT same language and script.\n"
            "3. If the user prompts in a localized script (like Devanagari or Telugu script), do not answer in English or Latin transliteration; use their native script.\n\n"
            "Integrate exact citation numbers inline using standard markdown bracket formatting like [1] or [2] "
            "to map facts directly back to their source links."
        )

        llm_stream = openai_client.chat.completions.create(
            model="gpt-4o",  # Enterprise default standard model
            messages=[
                {"role": "system", "content": f"{system_rules}\n\nLive Context Data:\n{web_context}"},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2, # Keeps the model factual and grounded in the web data
            stream=True
        )

        # Step 3: Stream generated response tokens out to the client loop
        for chunk in llm_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text_fragment = chunk.choices[0].delta.content
                yield f"data: {json.dumps({'type': 'token', 'text': text_fragment})}\n\n"
                await asyncio.sleep(0.002)

    except Exception as llm_error:
        yield f"data: {json.dumps({'type': 'error', 'message': str(llm_error)})}\n\n"

@app.post("/api/v1/search")
async def execute_kona_search_v1(payload: SearchQueryPayload):
    return StreamingResponse(
        process_and_stream_response(payload.message),
        media_type="text/event-stream"
    )