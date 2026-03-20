from subGraph_nodes import *
from langgraph.graph import StateGraph,END
from schemas import *

graph=StateGraph(reactState)
graph.add_node('think',think)
graph.set_entry_point('think')
graph.add_node('act',act)
graph.add_node('grade',grading)
graph.add_node('generation',generation)
graph.add_node('hallucination_check',hallucination_check)
graph.add_node('quality_check',quality_check)
graph.add_node('fallback_generation', fallback_generation)
graph.add_edge('act','grade')
graph.add_edge('fallback_generation', END)
graph.add_edge('generation','hallucination_check')
graph.add_conditional_edges('think', after_think, {
    'act': 'act',
    'fallback_generation': 'fallback_generation'            # 👈 changed from END
})
graph.add_conditional_edges('grade',after_grading,{
    'generation':'generation',
    'think':'think'
})
graph.add_conditional_edges('hallucination_check',after_hallucination_check,{
    'generation':'generation',
    'quality_check':'quality_check'
})
graph.add_conditional_edges('quality_check',after_quality_check,{
    'think':'think',
    END:END
})

react=graph.compile()

if __name__=='__main__':
    # from IPython.display import Image
    # img = react.get_graph().draw_mermaid_png()
    # with open('subgraph.png', 'wb') as f:
    #     f.write(img)
    # print("saved to subgraph.png")
    qn='How do Western scholars interpret the Bhagavad Gita philosophically?'
    result = react.invoke({
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
    })
    print(result['final_response'])
    print("===========================================================================")
    print('                            scratchpad                                      ')
    print('===========================================================================')
    for dict in result['scratchpad']:
        print(dict)