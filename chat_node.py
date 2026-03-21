from schemas import *
from langgraph.types import interrupt
from llm import get_llm
from prompts import *
from mainGraph import AgenticRAG
from langgraph.graph import END

def human_node(state: chatState):
    user_input = interrupt("Your question: ") 
    query_processor_llm=get_llm(mode='chat')
    query_processor_prompt_full=query_processor_prompt.format(
        input=user_input,
        prev_chats=state['history']
    )
    result=query_processor_llm.invoke(query_processor_prompt_full)
    return {
        'message':result.content,
        'input':user_input
        }
def intent_node(state:chatState):
    intent_detector_llm=get_llm(output_schema=intentDetector,mode='chat')
    intent_detector_prompt_full=intent_detector_prompt.format(
        message=state['message']
    )
    result=intent_detector_llm.invoke(intent_detector_prompt_full)
    return{
        'intent':result.intent
    }
def after_intent(state:chatState):
    if state['intent']=='rag':
        return 'rag'
    if state['intent']=='out_of_scope':
        return 'decline'
    else:
        return 'salutation'
def decline_node(state:chatState):
    msg="I am a Mahabharata Encyclopedia- Ask only questions related to it"
    print("AI OUTPUT:")
    print(msg)
    return{
        'history':[{'user':state['input']},{'AI':msg}]
    }
def rag(state:chatState):
    result = AgenticRAG.invoke({
        'original_question': state['message'],
        'execution_mode': '',
        'questions': [],
        'responses': [],
        'final_response': '',
        'flow': [],
        'current_index': 0
    })
    print('AI OUTPUT:')
    response=result['final_response']
    print(response)
    return{
        'history':[{'user':state['input']},{'AI':response}]
    }
def salutation(state:chatState):
    salutation_llm=get_llm(output_schema=convo,mode='chat')
    salutation_prompt_full=salutation_prompt.format(
        message=state['message']
    )
    result=salutation_llm.invoke(salutation_prompt_full)
    print("AI OUTPUT")
    print(result.convo_msg)
    return{
        'history':[{'user':state['input']},{'AI':result.convo_msg}],
        'exit':result.exit
    }
def after_salutation(state:chatState):
    if state['exit']==True:
        return END 
    else:
        return 'human'
