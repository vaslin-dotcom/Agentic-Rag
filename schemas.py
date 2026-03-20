from typing import TypedDict,List,Literal,Optional,Annotated
from pydantic import BaseModel,Field
from operator import concat

class reactState(TypedDict):
    query:str
    accumulated_chunks: List[str]
    original_query:str
    chunks:List[str]
    final_response:str
    tool:str
    tool_query:str
    iterations:int 
    scratchpad:List[dict]
    grade: str              
    hallucination: str      
    quality: str  
    answer:str 
    hallucination_retries: int   
    chunk_source: str      


class toolSelector(BaseModel):
    selected_tool: Literal['hybrid_search', 'web_search'] = Field(
        description="name of the tool to call"
    )
    search_query: str = Field(
        description="query to send to the tool"
    )
class relevance_checker_batch(BaseModel):
    relevance_scores: List[Literal['yes', 'no']] = Field(
        description='Relevance score for each chunk in order. Return one yes/no per chunk.'
    )
class hallucination_checker(BaseModel):
    hallucination:Literal['yes','no']=Field(description='check if the llm is hallucinating')
class quality_checker(BaseModel):
    quality:Literal['yes','no']=Field(description='Check if the chunk fully answers the query')
    new_query: str = Field(default='', description='If quality is no, write the missing part as a new query. If quality is yes, leave empty.')

class plannerOutput(BaseModel):
    execution_mode:Literal['sequential','parallel']=Field(description="Mode of execution for the query")
    questions:List[str]=Field(description="list of queries from the recieved query")

class plannerState(TypedDict):
    execution_mode:str
    questions:List[str]
    responses:Annotated[List[str],concat]
    final_response:str
    flow:Annotated[List[dict],concat]
    original_question:str
    current_index:int