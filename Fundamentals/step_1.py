import os
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
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "Hello, Claude"}
    ]
)


# The important code is abovve this line, this is just for displaying results nicely and saving to a file
# Print the raw API response object
print("=" * 50) # this literally means print 50 equals signs
print("RAW API RESPONSE OBJECT:")
print("=" * 50)
print(message)
print()

# Extract and print only the message content
print("=" * 50)
print("EXTRACTED MESSAGE CONTENT:")
print("=" * 50)
print(message.content)
print()

# Print server_tool_use data
print("=" * 50)
print("SERVER TOOL USE DATA:")
print("=" * 50)
print(message.usage.server_tool_use)
print("=" * 50)

# Save all responses to a text file
print("\n" + "=" * 50)
print("SAVING RESPONSE TO FILE...")
print("=" * 50)

# Create the content to save 
file_content = f"""{'=' * 50}
RAW API RESPONSE OBJECT:
{'=' * 50}
{message}

{'=' * 50}
EXTRACTED MESSAGE CONTENT:
{'=' * 50}
{message.content}

{'=' * 50}
SERVER TOOL USE DATA:
{'=' * 50}
{message.usage.server_tool_use}
{'=' * 50}
"""

# Write to file
with open("step_1.txt", "w") as file:
    file.write(file_content)

print("Response saved to 'step_1.txt'")
print("=" * 50)