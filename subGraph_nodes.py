from schemas import *
from langgraph.graph import END 
from llm import get_llm
from tools import *
from prompts import *


MAX_ITERATIONS = 5

def think(state: reactState):
    query = state['query']
    chunks = state['chunks']
    iterations = state['iterations']
    scratchpad=state['scratchpad']

    # 2. safety exit — hit max iterations
    if iterations >= MAX_ITERATIONS:
        return {
            **state,
            'tool': '',           # signals after_think to go to END
            'tool_query': '',
            'iterations': iterations
        }

    # 3. build your prompt — include query AND scratchpad
    #    so the LLM knows what it already tried
    think_prompt_final = think_prompt.format(
        query=query,
        scratchpad=scratchpad
    )

    # 4. call tool picker LLM
    think_llm = get_llm(output_schema=toolSelector,mode='think')
    response = think_llm.invoke(think_prompt_final)

    updated_scratchpad = scratchpad + [{
        'thought': f"picking tool {response.selected_tool} with query {response.search_query}"
    }]
    print("Done think node")
    # 6. return updated state
    print(f"ITERATIONS: {iterations}")
    return {
        **state,
        'tool': response.selected_tool,
        'tool_query': response.search_query,
        'iterations': iterations + 1,
        'scratchpad':updated_scratchpad
    }

def after_think(state: reactState):

    if state['tool'] == '':
        return 'fallback_generation'    
    
    return 'act'       

def act(state:reactState):
    tool_name=state['tool']
    scratchpad=state['scratchpad']
    tool_query=state['tool_query']
    if tool_name=='hybrid_search':
        chunks=hybrid_search(tool_query)
        updated_scratchpad=scratchpad+[{
            'act':'Made a hybrid search in vectorstore and keyword store'
        }]
        chunk_source = 'hybrid'
    else:
        chunks=web_search(tool_query)
        updated_scratchpad=scratchpad+[{
            'act':f'Made a web search with {tool_query}'
        }]
        chunk_source = 'web'
    print('done act node')
    return{
        **state,
        'chunks':chunks,
        'chunk_source': chunk_source,
        'scratchpad':updated_scratchpad
    }
    

def grading(state: reactState):
    chunks = state['chunks']                              # fresh from act
    accumulated_chunks = state.get('accumulated_chunks', [])  # persisted good chunks
    scratchpad = state['scratchpad']
    query = state['query']

    chunks_formatted = "\n\n".join(
        [f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)]
    )

    relevance_check_llm = get_llm(output_schema=relevance_checker_batch, mode='think')
    prompt = relevance_check_prompt_batch.format(
        query=query,
        n=len(chunks),
        chunks_formatted=chunks_formatted
    )

    result = relevance_check_llm.invoke(prompt)
    scores = result.relevance_scores

    # relevant chunks from THIS iteration only
    new_relevant_chunks = [
        chunk for chunk, score in zip(chunks, scores)
        if score == 'yes'
    ]

    # merge with previously accumulated — deduplicate
    updated_accumulated = list(dict.fromkeys(accumulated_chunks + new_relevant_chunks))

    print(f"done grading | new relevant: {len(new_relevant_chunks)} | total accumulated: {len(updated_accumulated)}")

    if len(updated_accumulated) >= 2:
        print(f"[Grading] {len(updated_accumulated)} chunk(s) — proceeding to generation")
        updated_scratchpad = scratchpad + [{
            'graded': f"accumulated {len(updated_accumulated)} relevant chunk(s) — proceeding"
        }]
        return {
            **state,
            'chunks': updated_accumulated,       # 👈 pass all good chunks to generation
            'accumulated_chunks': updated_accumulated,  # 👈 update accumulator
            'scratchpad': updated_scratchpad,
            'grade': 'relevant'
        }
    else:
        print(f"[Grading] only {len(updated_accumulated)} chunk(s) — re-querying")
        re_query_llm = get_llm(mode='generation')
        re_query_prompt_full = re_query_prompt.format(
            query=query,
            scratchpad=scratchpad
        )
        re_query = re_query_llm.invoke(re_query_prompt_full).content

        updated_scratchpad = scratchpad + [{
            'graded': f"only {len(updated_accumulated)} chunk(s) — requerying with: '{re_query}'"
        }]
        return {
            **state,
            'chunks': [],                    # 👈 fresh chunks for next act node
            'accumulated_chunks': updated_accumulated,  # 👈 preserved across iterations
            'query': re_query,
            'scratchpad': updated_scratchpad,
            'grade': 'irrelevant'
        }


def after_grading(state:reactState):
    if state['grade']=='relevant':
        return 'generation'
    else:
        return 'think'

def generation(state:reactState):
    chunks=state['chunks']
    query=state['query']
    answer=state.get('answer','')
    scratchpad=state['scratchpad']
    generation_llm=get_llm(mode='generation')
    generation_prompt_full=generation_prompt.format(
        chunks=chunks,
        query=query,
        answer=answer
    )
    response=generation_llm.invoke(generation_prompt_full)
    updated_scratchpad=scratchpad+[{
        'generation':'generated answer draft'
    }]
    print('done generation')
    return{
        **state,
        'scratchpad':updated_scratchpad,
        'answer':response.content
    }

# subGraph_nodes.py

# subGraph_nodes.py — hallucination_check node
def hallucination_check(state: reactState):
    answer = state['answer']
    scratchpad = state['scratchpad']
    chunks = state['chunks']
    chunk_source = state.get('chunk_source', 'hybrid')
    hallucination_retries = state.get('hallucination_retries', 0)

    hallucination_checker_llm = get_llm(output_schema=hallucination_checker, mode='think')

    # 👇 pick prompt based on source
    if chunk_source == 'web':
        prompt = hallucination_checker_prompt_lenient.format(chunks=chunks, answer=answer)
    else:
        prompt = hallucination_checker_prompt_strict.format(chunks=chunks, answer=answer)

    hallucination = hallucination_checker_llm.invoke(prompt)
    print(f"HALLUCINATION RESULT: {hallucination.hallucination}")
    print(f"HALLUCINATION RETRIES: {hallucination_retries}")

    if hallucination.hallucination == 'yes':
        if hallucination_retries >= 3:
            print("[Hallucination loop] max retries reached — forcing pass")
            return {
                **state,
                'hallucination': 'no',
                'hallucination_retries': 0,
                'scratchpad': scratchpad + [{'hallucination check': 'max retries — accepting best answer'}]
            }
        return {
            **state,
            'hallucination': hallucination.hallucination,
            'hallucination_retries': hallucination_retries + 1,
            'scratchpad': scratchpad + [{'hallucination check': f'hallucinated — retry {hallucination_retries + 1}/3'}]
        }
    else:
        return {
            **state,
            'hallucination': 'no',
            'hallucination_retries': 0,
            'scratchpad': scratchpad + [{'hallucination check': 'no hallucination detected'}]
        }

def after_hallucination_check(state:reactState):
    print(f"ROUTING TO: {'generation' if state['hallucination'] == 'yes' else 'quality_check'}")
    
    if state['hallucination']=='yes':
        
        return 'generation'
    else:
        
        return 'quality_check'

# subGraph_nodes.py
def quality_check(state: reactState):
    answer = state['answer']
    query = state['query']
    scratchpad = state['scratchpad']
    original_query=state['original_query']

    quality_checker_llm = get_llm(output_schema=quality_checker, mode='think')
    quality_checker_prompt_full = quality_checker_prompt.format(
        answer=answer,
        query=original_query
    )
    quality = quality_checker_llm.invoke(quality_checker_prompt_full)

    print(f"QUALITY RESULT: {quality.quality}")


    if quality.quality == 'yes':
        updated_scratchpad = scratchpad + [{
            'quality check': 'query fully answered'
        }]
        return {
            **state,
            'quality': quality.quality,
            'scratchpad': updated_scratchpad,
            'final_response': answer
        }
    else:
        # 👇 check if new_query is same as current query — if so stop looping
        if quality.new_query.strip().lower() == query.strip().lower():
            print("[Quality loop detected] same query repeated — treating as answered")
            return {
                **state,
                'quality': 'yes',          # force stop
                'final_response': answer,
                'scratchpad': scratchpad + [{
                    'quality check': 'loop detected — returning current answer as final'
                }]
            }

        updated_scratchpad = scratchpad + [{
            'quality_check': f'query {query} is answered {answer} want to know more about {quality.new_query}'
        }]
        return {
            **state,
            'query': quality.new_query,
            'chunks': [],
            'quality': quality.quality,
            'scratchpad': updated_scratchpad
        }

def after_quality_check(state:reactState):
    if state['quality']=='yes':
        return END 
    else:
        return 'think'    

def fallback_generation(state: reactState):
    chunks = state['chunks']
    query = state['query']
    answer=state['answer']
    scratchpad = state['scratchpad']

    # if we have no chunks at all, say so honestly
    if not chunks:
        if answer:  # use previous answer if available
            return {
                **state,
                'final_response': answer,
                'scratchpad': scratchpad + [{'fallback': 'no chunks but returning previous answer draft'}]
            }
        return {
            **state,
            'final_response': "I could not find sufficient information in the knowledge base to answer this question confidently.",
            'scratchpad': scratchpad + [{'fallback': 'hit max iterations with no chunks'}]
        }

    generation_llm = get_llm(mode='generation')
    fallback_prompt_full = fallback_prompt.format(
        query=query,
        chunks=chunks,
        answer=answer
    )
    response = generation_llm.invoke(fallback_prompt_full)
    updated_scratchpad = scratchpad + [{
        'fallback': 'hit max iterations — generating best-effort answer from available chunks'
    }]

    print('done fallback generation')
    return {
        **state,
        'final_response': response.content,
        'scratchpad': updated_scratchpad
    }


        




