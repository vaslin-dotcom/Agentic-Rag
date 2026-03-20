from schemas import plannerOutput,plannerState,reactState
from prompts import planner_prompt,final_generation_prompt
from llm import get_llm
from langgraph.types import Send
from subGraph import react

def planner(state:plannerState):
    question=state['original_question']
    planner_prompt_full=planner_prompt.format(
        question=question
    )
    planner_llm=get_llm(output_schema=plannerOutput,mode='think')
    result=planner_llm.invoke(planner_prompt_full)
    print(f"completed planning with {len(result.questions)} questions")
    return {
    'flow': [{'planner': f'{question} divided into {result.questions} with mode {result.execution_mode}'}],
    'execution_mode': result.execution_mode,
    'questions': result.questions
}
def executor(state:plannerState):
    if state['execution_mode'] == 'parallel':
        return {}
    else:
        index=state['current_index']
        qn=state['questions'][index]
        result=react.invoke(
        {
            'original_query': qn,
            'query': qn,
            'chunks': [],
            'final_response': '',
            'tool': '',
            'tool_query': '',
            'iterations': 0,
            'scratchpad': [],
            'grade': '',
            'hallucination': '',
            'quality': '',
            'answer': '',
            'accumulated_chunks': [],
            'hallucination_retries':0
            }
        )
        print(f"completed question {index+1}")
        return{
            'flow':[{'executor':f'completed question {index+1}'}],
            'responses':[result['final_response']],
            'current_index':index+1
        }
    
def after_executor(state:plannerState):
    if state['execution_mode'] == 'parallel':
        return [
            Send('collect_response', {
                'original_query': qn,
                'query': qn,
                'chunks': [],
                'final_response': '',
                'tool': '',
                'tool_query': '',
                'iterations': 0,
                'scratchpad': [],
                'grade': '',
                'hallucination': '',
                'quality': '',
                'answer': '',
                'accumulated_chunks': [],
                'hallucination_retries': 0
            })
            for qn in state['questions']
        ]
    if state['current_index']<len(state['questions']):
            return 'executor'
    print("completed execution")
    return 'synthesizer'

def collect_response(state:reactState):
     result=react.invoke(state)
     return{
          'responses':[result['final_response']],
          'flow':[{'executor':"Completed parallel execution "}]
     }

def after_collect(state:plannerState):
     print('completed execution')
     return 'synthesizer'

def  synthesizer(state:plannerState):
     final_generation_llm=get_llm(mode='generation')
     final_generation_prompt_final=final_generation_prompt.format(
          original_question=state['original_question'],
          responses=state['responses']
     )
     result=final_generation_llm.invoke(final_generation_prompt_final)
     print("Completed sythesizing")
     return{
          'final_response':result.content,
          'flow':[{'synthesizer':'Synthesizer generate final output'}]
     }