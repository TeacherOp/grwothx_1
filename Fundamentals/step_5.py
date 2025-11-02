# Getting structured JSON responses from Claude using Tool Use (the proper way!)
# This is the recommended approach for getting structured, validated JSON output from Claude.

import os
import json
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

# Define a tool with the JSON structure we want as response
# Even though we won't actually execute this tool, Claude will use it to structure its response
tools = [
    {
        "name": "udayan_greet_and_introduce_growthx",
        "description": "Get structured information about a person",
        "input_schema": {
            "type": "object",
            "properties": {
                "greeting": {
                    "type": "string",
                    "description": "This will be a greeting message from Udyan"
                },
                "about_growthx": {
                    "type": "string",
                    "description": "Brief Information about GrowthX"
                },
                "what_does_growthx_do": {
                    "type": "array",
                    "description": "A list of things that GrowthX does"
                },
                "who_should_join_growthx": {
                    "type": "array",
                    "items": {
                        "description": "List of people who should join GrowthX"
                    }
                }, 
                "why_join_growthx": {
                    "type": "array",
                    "items": {
                        "description": "List of reasons to join GrowthX, in a way Yoda would say it"
                    }
                }
            },
            "required": ["greeting", "about_growthx", "what_does_growthx_do", "who_should_join_growthx"]
        }
    }
]

# Create the message with tool use
message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    # now here in the system prompt you can insert any kind of context / data which needs to be analysed script it to analyse a lot of data by inserting variable in this system prompt!
    system="You are Udyan the founder of growthx, please greet the user, share them about growthx, what we do, who should join growthx, why they should join growthx",
    tools=tools,
    tool_choice={"type": "tool", "name": "udayan_greet_and_introduce_growthx"},  # Force the model to use this tool
    messages=[
        {
            "role": "user",
            "content": "Hi there!"
        }
    ]
)

# The important code is above this line, this is just for displaying results nicely
print("=" * 50)
print("FULL API RESPONSE:")
print("=" * 50)
print(json.dumps(message.model_dump(), indent=2))
print()

# Extract the structured data from the tool use
print("=" * 50)
print("EXTRACTED STRUCTURED DATA:")
print("=" * 50)

# The structured response is in the tool_use block easy to parse and access, as you defined in the schema, so you what to expectt! No suprises here.
for content in message.content:
    if content.type == "tool_use":
        structured_data = content.input
        print(json.dumps(structured_data, indent=2))

        # You can also access individual fields
        print("\n" + "=" * 50)
        print("ACCESSING INDIVIDUAL FIELDS:")
        print("=" * 50)
        print(f"Greeting: {structured_data['greeting']}")
        print(f"\nAbout GrowthX: {structured_data['about_growthx'][:100]}...")  # Show first 100 chars
        print(f"\nWhat GrowthX Does ({len(structured_data['what_does_growthx_do'])} items):")
        for item in structured_data['what_does_growthx_do'][:3]:  # Show first 3 items
            print(f"  - {item}")
        print(f"\nWho Should Join ({len(structured_data['who_should_join_growthx'])} types)")
        if 'why_join_growthx' in structured_data:
            print(f"\nWhy Join (Yoda style): {structured_data['why_join_growthx'][0]}")

# Save the structured response to JSON file
print("\n" + "=" * 50)
print("SAVING STRUCTURED DATA TO JSON FILE...")
print("=" * 50)

with open("step_5_structured_response.json", "w") as json_file:
    # Save just the structured data, not the whole response
    for content in message.content:
        if content.type == "tool_use":
            json.dump(content.input, json_file, indent=2)

print("Structured data saved to 'step_5_structured_response.json'")
print("=" * 50)

# Note for learners:
print("\n" + "=" * 50)
print("KEY LEARNINGS:")
print("=" * 50)
print("1. Tool use guarantees structured JSON output")
print("2. The model won't hallucinate extra fields")
print("3. Required fields are always included")
print("4. The schema acts as a contract for the response")
print("5. stop_reason will be 'tool_use' when using tools")
print("=" * 50)