from mainGraph_nodes import *
from langgraph.graph import StateGraph,END
from langgraph.types import Send
from schemas import plannerState

mainGraph=StateGraph(plannerState)
mainGraph.add_node('planner',planner)
mainGraph.add_node('executor',executor)
mainGraph.add_node('synthesizer',synthesizer)
mainGraph.add_node('collect_response',collect_response)
mainGraph.set_entry_point('planner')
mainGraph.add_edge('planner','executor')
mainGraph.add_conditional_edges('collect_response',after_collect,{
    'synthesizer':'synthesizer'
})
mainGraph.add_edge('synthesizer',END)
mainGraph.add_conditional_edges('executor',after_executor,{
    'executor':'executor',
    'synthesizer':'synthesizer'
})

AgenticRAG=mainGraph.compile()

if __name__ == '__main__':
    result = AgenticRAG.invoke({
        'original_question': "Compare how the Mahabharata text describes Karna's death and what modern scholars say about the symbolism of his death",
        'execution_mode': '',
        'questions': [],
        'responses': [],
        'final_response': '',
        'flow': [],
        'current_index': 0
    })
    print(result['final_response'])
    print('\n--- flow ---')
    for step in result['flow']:
        print(step)
    
    # from IPython.display import Image
    # img = AgenticRAG.get_graph().draw_mermaid_png()
    # with open('AGENTIC_RAG.png', 'wb') as f:
    #     f.write(img)
    # print("saved to subgraph.png")