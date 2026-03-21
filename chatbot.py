from chat_node import *
from langgraph.graph import StateGraph,END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

checkpointer = MemorySaver()
chatgraph=StateGraph(chatState)
chatgraph.add_node('human',human_node)
chatgraph.set_entry_point('human')
chatgraph.add_node('intent',intent_node)
chatgraph.add_node('rag',rag)
chatgraph.add_node('decline',decline_node)
chatgraph.add_node('salutation',salutation)
chatgraph.add_edge('human','intent')
chatgraph.add_edge('rag','human')
chatgraph.add_edge('decline','human')
chatgraph.add_conditional_edges('intent',after_intent,{
    'rag':'rag',
    'decline':'decline',
    'salutation':'salutation'
})
chatgraph.add_conditional_edges('salutation',after_salutation,{
    END:END,
    'human':'human'
})

chatbot=chatgraph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "chat_session_1"}}
initial_state = {
    'input': '',
    'message': '',
    'intent': '',
    'exit': False,
    'history': []
}

print("=" * 50)
print("   Welcome to the Mahabharata Encyclopedia")
print("=" * 50)
print("Ask me anything about the great epic.")
print("Type 'bye' or 'exit' to leave.\n")

result = chatbot.invoke(initial_state, config=config)

while True:
    # get user input
    user_input = input("You: ").strip()
    
    if not user_input:
        continue
    
    # resume graph with user input — Command passes value to interrupt()
    result = chatbot.invoke(
        Command(resume=user_input),
        config=config
    )
    
    # check if user exited via salutation node
    if result.get('exit') == True:
        break

print("\nFarewell, seeker of wisdom.")