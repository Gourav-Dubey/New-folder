from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

# ===== MEMORY =====
notes = {}
todos = []
chat_history = []

# ===== TOOLS =====

@tool
def save_note(content: str) -> str:
    """Save a note. Just pass the note content."""
    import time
    title = f"Note {len(notes)+1}"
    notes[title] = content
    return f"✅ Note saved: '{content}'"

@tool
def get_notes() -> str:
    """Get all saved notes"""
    if not notes:
        return "No notes saved yet."
    return "\n".join([f"📝 {t}: {c}" for t, c in notes.items()])

@tool
def add_todo(task: str) -> str:
    """Add a single task to todo list"""
    # Check duplicate
    for t in todos:
        if t["task"].lower() == task.lower():
            return f"⚠️ Task already exists: {task}"
    todos.append({"task": task, "done": False})
    return f"✅ Task added: {task}"

@tool
def get_todos() -> str:
    """Get all todos"""
    if not todos:
        return "No tasks yet."
    result = []
    for i, t in enumerate(todos):
        status = "✅" if t["done"] else "⬜"
        result.append(f"{status} {i+1}. {t['task']}")
    return "\n".join(result)

@tool
def complete_todo(index: int) -> str:
    """Mark a todo as complete by index (1-based)"""
    if 0 < index <= len(todos):
        todos[index-1]["done"] = True
        return f"✅ Task {index} marked complete: {todos[index-1]['task']}"
    return f"⚠️ Invalid task number: {index}. You have {len(todos)} tasks."

@tool
def delete_todo(index: int) -> str:
    """Delete a todo by index (1-based)"""
    if 0 < index <= len(todos):
        task = todos.pop(index-1)
        return f"🗑️ Deleted: {task['task']}"
    return f"⚠️ Invalid task number: {index}"

@tool
def web_search(query: str) -> str:
    """Search the web for real-time information"""
    from langchain_community.tools.tavily_search import TavilySearchResults
    search = TavilySearchResults(max_results=3)
    results = search.invoke(query)
    return "\n\n".join([f"Source: {r['url']}\n{r['content']}" for r in results])


# ===== AGENT =====

tools = [save_note, get_notes, add_todo, get_todos, complete_todo, delete_todo, web_search]
tools_map = {t.name: t for t in tools}

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a helpful personal AI assistant with these tools:
- save_note(content): Save a note
- get_notes(): Get all notes
- add_todo(task): Add ONE task only
- get_todos(): Get all tasks
- complete_todo(index): Mark task done
- delete_todo(index): Delete a task
- web_search(query): Search internet

IMPORTANT RULES:
- ALWAYS use tools when user asks to add/save/show/search
- For "save note: X" → call save_note with X as content
- For "add todo: X" → call add_todo with X as task — only ONCE
- Never call the same tool twice in one response
- Be concise and friendly
- Always respond in English"""

def run_agent(user_input: str) -> str:
    chat_history.append(HumanMessage(content=user_input))
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + chat_history[-10:]

    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            chat_history.append(AIMessage(content=response.content))
            return response.content

        # Execute tools — only unique ones
        seen_tools = set()
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            if tool_name in seen_tools:
                continue
            seen_tools.add(tool_name)

            if tool_name in tools_map:
                try:
                    result = tools_map[tool_name].invoke(tool_call["args"])
                except Exception as e:
                    result = f"Error: {str(e)}"
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                ))

    return "Sorry, could not complete the task."
