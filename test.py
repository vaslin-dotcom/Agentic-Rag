# test_nvidia_think.py
import time
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal
from config import NVIDIA_API_KEY, NVIDIA_BASE_URL

class toolSelector(BaseModel):
    selected_tool: Literal['hybrid_search', 'web_search'] = Field(
        description="name of the tool to call"
    )
    search_query: str = Field(
        description="query to send to the tool"
    )

# hard reasoning prompt — same as your actual think node
think_prompt = """
You are an intelligent agent answering questions about the Mahabharata.

Your job is to decide which tool to use and what query to send to it.

Available tools:
- hybrid_search: PRIMARY tool. Use for any question about Mahabharata characters, events, themes, relationships, dialogues.
- web_search: FALLBACK tool. Use ONLY when the question is about historical context, scholarly interpretations, or anything outside the Mahabharata story itself.

Previous attempts (scratchpad):
[
  {{'thought': "picking tool hybrid_search with query: 'Karna not fighting first 10 days'"}},
  {{'act': "hybrid_search with query: 'Karna not fighting first 10 days'"}},
  {{'graded': "chunks not relevant — requerying with: 'Karna absence Bhishma command Kurukshetra'"}}
]

Current question: Why did Karna not fight in the first 10 days of the Kurukshetra war?

Instructions:
- Study the scratchpad carefully. Do NOT repeat a query that was already tried.
- Pick a fundamentally different query from what was already tried.
- Be specific about characters and parvas involved.

You MUST call the toolSelector tool. Do not respond in plain text.
"""

models = [
    "moonshotai/kimi-k2-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "nvidia/nemotron-3-nano-30b-a3b",
]

print("=" * 60)
print("       NVIDIA THINK NODE MODEL TEST")
print("=" * 60)

for model in models:
    print(f"\nTesting: {model}")
    print("-" * 60)
    try:
        llm = ChatOpenAI(
            model=model,
            temperature=0,
            api_key=NVIDIA_API_KEY,
            base_url=NVIDIA_BASE_URL,
            max_retries=1,
            request_timeout=30
        )

        structured_llm = llm.with_structured_output(toolSelector)

        start = time.time()
        response = structured_llm.invoke(think_prompt)
        elapsed = round(time.time() - start, 2)

        print(f"  speed       : {elapsed}s")
        print(f"  tool        : {response.selected_tool}")
        print(f"  query       : {response.search_query[:80]}")

        # check if it avoided repeating scratchpad queries
        repeated = any(
            q in response.search_query.lower()
            for q in ['karna not fighting first 10 days', 'karna absence bhishma command']
        )
        print(f"  repeated?   : {'YES ❌ bad' if repeated else 'NO ✅ good'}")

    except Exception as e:
        print(f"  FAILED      : {type(e).__name__}: {str(e)[:80]}")

    time.sleep(3)

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)