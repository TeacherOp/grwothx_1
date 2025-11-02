# Now we will try to get the moodel to respond in a specific format which already available as a tool parameter in all llm models
import os
import json # we are importing json to save the response to a json file
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)
message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    system="You are the founder of GrowthX, and your name is Udayan, and you greet people by saying 'What's up champ!' Always repspond with html formatted data", # we added a system prompt here, which sets the behavior of the AI, can include instructions, context, etc.
    temperature=0.1, # we added temperature here, which controls the randomness of the output, lower values make it more focused and deterministic, higher values make it more random
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)

# The important code is abovve this line, this is just for displaying results nicely and saving to a file
# Print the raw API response object
print("=" * 50) # this literally means print 50 equals signs
print("RAW API RESPONSE OBJECT:")
print("=" * 50)
print(message.content)
print()

# Save raw response to JSON file
print("=" * 50)
print("SAVING RAW RESPONSE TO JSON FILE...")
print("=" * 50)

# Convert the message object to dictionary and save as JSON
with open("step_3.json", "w") as json_file:
    json.dump(message.model_dump(), json_file, indent=2)

print("Response saved to 'step_3.json'")
print("=" * 50)