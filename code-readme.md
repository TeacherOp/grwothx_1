# Code Reference Guide - Building AI Tools with Claude

This document explains the core functions, imports, and code patterns used throughout this project. Use it as a quick reference when building your own AI tools.

---

## Table of Contents
1. [Essential Imports & Setup](#essential-imports--setup)
2. [Basic API Call Pattern](#basic-api-call-pattern)
3. [Tool Use Pattern (Structured Output)](#tool-use-pattern-structured-output)
4. [Conversation Management](#conversation-management)
5. [Agent Architecture](#agent-architecture)
6. [File Operations](#file-operations)
7. [Error Handling](#error-handling)
8. [Common Utilities](#common-utilities)

---

## Essential Imports & Setup

### Core Imports
```python
import os                    # For environment variables and file operations
from dotenv import load_dotenv  # For loading .env files
import anthropic            # Claude API client
import json                 # For JSON operations
import csv                  # For CSV file operations
from datetime import datetime  # For timestamps
from pathlib import Path    # For modern file path handling
```

### Environment Setup Pattern
```python
# Load environment variables from .env file
load_dotenv()

# Initialize Claude client
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# Model configuration
MODEL = "claude-sonnet-4-5-20250929"
```

**Key Concepts:**
- `.env` file stores sensitive data (API keys)
- `load_dotenv()` loads variables from `.env` into `os.environ`
- Always use `os.environ.get()` to retrieve environment variables

---

## Basic API Call Pattern

### Simple Message Creation
```python
message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)

# Extract response content
response_text = message.content[0].text
```

### With System Prompt
```python
message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    system="You are a helpful assistant specialized in Python programming.",
    messages=[
        {"role": "user", "content": "Explain list comprehensions"}
    ]
)
```

### With Temperature Control
```python
message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    temperature=0.1,  # 0.0-1.0: Lower = more focused, Higher = more creative
    messages=[
        {"role": "user", "content": "Your message"}
    ]
)
```

**Key Parameters:**
- `model`: Which Claude model to use
- `max_tokens`: Maximum length of response (higher = longer responses)
- `temperature`: Creativity level (0.0 = deterministic, 1.0 = creative)
- `system`: Sets behavior/personality of the AI
- `messages`: Conversation history (must alternate user/assistant)

---

## Tool Use Pattern (Structured Output)

### Why Use Tools?
Tool use is the **recommended way** to get structured, validated JSON output from Claude. It guarantees:
- ✅ Consistent JSON structure
- ✅ No hallucinated fields
- ✅ Required fields always present
- ✅ Type validation

### Basic Tool Definition
```python
tools = [
    {
        "name": "return_analysis_result",
        "description": "Submit the final analysis results",
        "input_schema": {
            "type": "object",
            "properties": {
                "field_name": {
                    "type": "string",
                    "description": "Description of this field"
                },
                "yes_no_field": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Field with limited choices"
                },
                "items_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of items"
                }
            },
            "required": ["field_name", "yes_no_field"]
        }
    }
]
```

### Using Tools in API Call
```python
response = client.messages.create(
    model=MODEL,
    max_tokens=4000,
    system="Your system prompt here",
    messages=[{"role": "user", "content": "Your request"}],
    tools=tools,
    tool_choice={"type": "tool", "name": "return_analysis_result"}  # Force tool use
)
```

### Extracting Tool Results
```python
def process_response(response):
    """Extract tool use results from API response"""
    for content in response.content:
        if hasattr(content, 'type') and content.type == 'tool_use':
            # content.name = tool name
            # content.input = the structured data
            # content.id = unique tool use ID
            return content.input
    return None

# Usage
result = process_response(response)
print(result['field_name'])
```

### Multi-Turn Tool Conversation
```python
# Initial call
response = client.messages.create(
    model=MODEL,
    max_tokens=4000,
    system=system_prompt,
    messages=messages,
    tools=tools
)

# Add assistant response to conversation
messages.append({
    "role": "assistant",
    "content": response.content
})

# Add tool results back as user message
tool_results = []
for content_block in response.content:
    if content_block.type == 'tool_use':
        # Execute your tool function
        result = execute_tool(content_block.name, content_block.input)

        tool_results.append({
            "type": "tool_result",
            "tool_use_id": content_block.id,
            "content": json.dumps(result)
        })

messages.append({
    "role": "user",
    "content": tool_results
})

# Continue conversation
next_response = client.messages.create(
    model=MODEL,
    max_tokens=4000,
    system=system_prompt,
    messages=messages,
    tools=tools
)
```

**Tool Schema Types:**
- `string`: Text field
- `number`: Numeric value
- `boolean`: True/false
- `array`: List of items
- `object`: Nested structure
- `enum`: Limited set of choices

---

## Conversation Management

### Basic Context Preservation
```python
# Initialize conversation
messages = []

# Add user message
messages.append({
    "role": "user",
    "content": "User's question"
})

# Get response
response = client.messages.create(
    model=MODEL,
    max_tokens=1024,
    messages=messages
)

# Add assistant response
messages.append({
    "role": "assistant",
    "content": response.content[0].text
})

# Continue conversation - context is preserved
messages.append({
    "role": "user",
    "content": "Follow-up question"
})

response = client.messages.create(
    model=MODEL,
    max_tokens=1024,
    messages=messages  # Full history sent
)
```

### Saving Conversations
```python
def save_conversation(conversation_id, messages, metadata=None):
    """Save conversation to JSON file"""
    conversation_data = {
        "conversation_id": conversation_id,
        "created_at": metadata.get("created_at") if metadata else datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "message_count": len(messages),
        "messages": messages
    }

    filename = f"conversations/conversation_{conversation_id}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(conversation_data, f, indent=2, ensure_ascii=False)

    return filename
```

### Loading Conversations
```python
def load_conversation(conversation_id):
    """Load conversation from file"""
    filename = f"conversations/conversation_{conversation_id}.json"
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["messages"], data
    except FileNotFoundError:
        return None, None
```

**Message Structure:**
```python
# Minimum structure for API
{"role": "user", "content": "text"}
{"role": "assistant", "content": "text"}

# With metadata for storage
{
    "role": "user",
    "content": "text",
    "timestamp": "2024-01-01T10:00:00"
}
```

---

## Agent Architecture

### Basic Agent Class Pattern
```python
class AIAgent:
    """Base pattern for AI agents"""

    def __init__(self, api_key: str):
        """Initialize agent with API client"""
        self.client = anthropic.Anthropic(api_key=api_key)
        self.messages = []
        self.results = []

    def create_tools(self):
        """Define tools the agent can use"""
        return [
            {
                "name": "tool_name",
                "description": "What this tool does",
                "input_schema": {
                    # Tool schema here
                }
            }
        ]

    def analyze(self, input_data):
        """Main method to perform agent's task"""
        system_prompt = self.build_system_prompt(input_data)

        messages = [
            {"role": "user", "content": "Your request based on input_data"}
        ]

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=4000,
            system=system_prompt,
            messages=messages,
            tools=self.create_tools()
        )

        return self.process_response(response)

    def process_response(self, response):
        """Extract and return results"""
        # Implementation specific to agent
        pass
```

### Agent with Tool Execution
```python
class SEOAgent:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.messages = []

    def execute_tool(self, tool_name: str, tool_input: dict):
        """Execute tools called by Claude"""
        if tool_name == "image_generator":
            return self.generate_image(tool_input["prompt"])
        elif tool_name == "blog_creator":
            return self.create_blog(tool_input)
        # ... more tools

    def run(self):
        """Main agent loop with tool execution"""
        max_iterations = 10
        iteration = 0
        task_complete = False

        while not task_complete and iteration < max_iterations:
            iteration += 1

            # API call
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=self.system_prompt,
                messages=self.messages,
                tools=self.get_tools()
            )

            # Add response to conversation
            self.messages.append({
                "role": "assistant",
                "content": response.content
            })

            # Check for tool uses
            tool_results = []
            for block in response.content:
                if block.type == 'tool_use':
                    result = self.execute_tool(block.name, block.input)

                    # Track completion
                    if block.name == "final_tool" and result.get("status") == "success":
                        task_complete = True

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })

            # Add tool results back
            if tool_results:
                self.messages.append({
                    "role": "user",
                    "content": tool_results
                })
            else:
                # No tools, might be done
                break

        return {"status": "success" if task_complete else "incomplete"}
```

**Key Concepts:**
- Agents encapsulate AI behavior and tools
- Use iteration loops for multi-step tasks
- Always track completion conditions
- Tool results feed back into conversation

---

## File Operations

### Reading CSV Files
```python
import csv

def read_csv_data(filename):
    """Read CSV file and return list of dictionaries"""
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    return rows

# Usage
data = read_csv_data('input.csv')
for row in data:
    discipline = row.get('Discipline', '').strip()
    skill = row.get('Skill', '').strip()
```

### Writing to CSV (Append Mode)
```python
def save_result_to_csv(result: dict, filename: str):
    """Append single result to CSV"""
    # Ensure directory exists
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists (for header)
    file_exists = Path(filename).exists()

    # Write to CSV
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['field1', 'field2', 'field3']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if new file
        if not file_exists:
            writer.writeheader()

        writer.writerow(result)
```

### JSON Operations
```python
# Save dictionary to JSON
def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Load JSON file
def load_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

# Pretty print JSON
print(json.dumps(data, indent=2))
```

### Creating Timestamped Filenames
```python
from datetime import datetime

# Create unique filename with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"output_data_{timestamp}.csv"
# Result: output_data_20240115_143022.csv
```

### Path Operations with pathlib
```python
from pathlib import Path

# Check if file/directory exists
if Path('file.txt').exists():
    print("File exists")

# Create directory if it doesn't exist
Path('output_data').mkdir(parents=True, exist_ok=True)

# Get parent directory
parent = Path('path/to/file.txt').parent

# List files with pattern
for file in Path('data').glob('*.csv'):
    print(file)
```

---

## Error Handling

### Try-Except Pattern for API Calls
```python
def safe_api_call(client, messages):
    """API call with error handling"""
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=messages
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error during API call: {str(e)}")
        return None
```

### Creating Error Results
```python
def create_error_result(item_id: str, error: str) -> dict:
    """Create standardized error result"""
    return {
        "id": item_id,
        "status": "error",
        "result": None,
        "error_message": f"Error: {error}",
        "timestamp": datetime.now().isoformat()
    }
```

### Retry with Delay (Rate Limiting)
```python
import time

def process_with_rate_limit(items, process_func, delay=2):
    """Process items with delay between calls"""
    results = []
    total = len(items)

    for idx, item in enumerate(items, 1):
        print(f"Processing {idx}/{total}...")

        try:
            result = process_func(item)
            results.append(result)
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            results.append(create_error_result(item, str(e)))

        # Delay between API calls (except last item)
        if idx < total:
            time.sleep(delay)

    return results
```

---

## Common Utilities

### Progress Tracking
```python
def process_batch_with_progress(items, process_func):
    """Process items with progress display"""
    results = []
    total = len(items)

    print(f"Processing {total} items...")
    print("-" * 50)

    for idx, item in enumerate(items, 1):
        print(f"\n[{idx}/{total}] Processing: {item.get('name', 'Unknown')}")

        result = process_func(item)
        results.append(result)

        # Status indicator
        if result.get('status') == 'success':
            print(f"✅ Completed successfully")
        else:
            print(f"⚠️ Completed with warnings/errors")

    print("\n" + "=" * 50)
    print(f"✨ Batch processing complete! ({len(results)} items)")

    return results
```

### Formatting System Prompts with Variables
```python
# Template string
SYSTEM_PROMPT = """You are an expert analyst.

Context Information:
- Client: {client_name}
- Task Type: {task_type}
- Data Source: {data_source}

Your goal is to {goal_description}
"""

# Fill in template
system_prompt = SYSTEM_PROMPT.format(
    client_name="Acme Corp",
    task_type="Market Analysis",
    data_source="Q4 2023 Sales Data",
    goal_description="identify growth opportunities"
)
```

### String Manipulation for URLs/Slugs
```python
from slugify import slugify

# Create URL-friendly slug
title = "How to Build AI Tools: A Complete Guide"
slug = slugify(title)
# Result: "how-to-build-ai-tools-a-complete-guide"
```

### Extracting Specific Response Types
```python
def extract_text_from_response(response):
    """Extract all text blocks from response"""
    text_blocks = []
    for content in response.content:
        if hasattr(content, 'type') and content.type == 'text':
            text_blocks.append(content.text)
    return '\n'.join(text_blocks)

def extract_tool_uses(response):
    """Extract all tool uses from response"""
    tool_uses = []
    for content in response.content:
        if hasattr(content, 'type') and content.type == 'tool_use':
            tool_uses.append({
                'name': content.name,
                'input': content.input,
                'id': content.id
            })
    return tool_uses
```

---

## Quick Reference: Common Patterns

### Pattern 1: Simple One-Shot Analysis
```python
# Load environment
load_dotenv()
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Make request
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Analyze this data: ..."}]
)

# Get result
result = response.content[0].text
print(result)
```

### Pattern 2: Structured Output with Tool
```python
# Define tool
tools = [{
    "name": "return_result",
    "description": "Return structured analysis",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "score": {"type": "number"}
        },
        "required": ["summary", "score"]
    }
}]

# Request with tool
response = client.messages.create(
    model=MODEL,
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "tool", "name": "return_result"},
    messages=[{"role": "user", "content": "Analyze..."}]
)

# Extract structured data
for content in response.content:
    if content.type == 'tool_use':
        result = content.input
        print(f"Summary: {result['summary']}")
        print(f"Score: {result['score']}")
```

### Pattern 3: Batch Processing with CSV
```python
# Read CSV
with open('input.csv', 'r') as f:
    reader = csv.DictReader(f)
    items = list(reader)

# Process each item
results = []
for item in items:
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": f"Analyze: {item['data']}"}]
    )
    results.append({
        'input': item['data'],
        'output': response.content[0].text
    })
    time.sleep(2)  # Rate limiting

# Save results
with open('output.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['input', 'output'])
    writer.writeheader()
    writer.writerows(results)
```

### Pattern 4: Conversational Agent
```python
messages = []

def chat(user_input):
    # Add user message
    messages.append({"role": "user", "content": user_input})

    # Get response
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system="You are a helpful assistant",
        messages=messages
    )

    # Extract and store response
    assistant_message = response.content[0].text
    messages.append({"role": "assistant", "content": assistant_message})

    return assistant_message

# Use it
print(chat("Hello!"))
print(chat("What's the weather like?"))  # Has context from previous
```

---

## Key Takeaways

1. **Always use environment variables** for API keys via `.env` and `load_dotenv()`

2. **Tool use is better than asking for JSON** - it's validated and structured

3. **Maintain conversation context** by keeping all messages in a list and passing it to each API call

4. **Use descriptive system prompts** to set behavior and provide context

5. **Handle errors gracefully** with try-except blocks and meaningful error messages

6. **Implement rate limiting** with `time.sleep()` between batch API calls

7. **Save incrementally** when processing large batches (don't wait until the end)

8. **Use pathlib** for modern, cross-platform file path handling

9. **For multi-turn interactions**, maintain the conversation loop: user → assistant → tool results → user → assistant...

10. **Temperature matters**: Use low (0.0-0.3) for analytical tasks, higher (0.7-1.0) for creative content

---

## Next Steps

Now that you understand these patterns, you can:
- Mix and match patterns to build custom agents
- Create multi-step workflows by combining tool uses
- Build complex applications with chat interfaces
- Process large datasets with batch operations
- Save and resume long-running conversations

Refer back to this guide whenever you need to implement a specific pattern!
