"""
SEO Blog Agent

This module contains the main AI agent that orchestrates blog creation.
It handles the conversation with Claude, tool execution, and progress tracking.
"""

import json
from typing import Dict, Any, Optional, Callable
from anthropic import Anthropic
from config import Config
from tools.tool_executor import ToolExecutor
from tools.blog_tools import fetch_existing_blogs


class SEOBlogAgent:
    """
    AI agent for generating SEO-optimized blog posts

    This agent uses Claude to generate comprehensive blog posts with images,
    following SEO best practices and brand guidelines.
    """

    def __init__(self, progress_callback: Optional[Callable] = None):
        """
        Initialize the SEO Blog Agent

        Args:
            progress_callback: Optional callback function for progress updates
        """
        self.anthropic = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.tool_executor = ToolExecutor()
        self.progress_callback = progress_callback
        self.messages = []
        self.brand_context = Config.load_brand_context()
        self.existing_blogs = fetch_existing_blogs()

    def send_progress(self, event_type: str, data: Dict[str, Any]):
        """Send progress update through callback if available"""
        if self.progress_callback:
            self.progress_callback(event_type, data)

    def get_tool_definitions(self) -> list:
        """Get tool definitions for Claude API"""
        return [
            {
                "name": "image_generator",
                "description": "Generate a blog header image using Google Imagen 4.0 Ultra AI model",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": """Describe the image in natural language following these guidelines for best results with Imagen 4.0 Ultra:

                            DO:
                            • Be specific and descriptive (e.g., "a golden retriever sitting on a beach at sunset" not just "dog")
                            • Specify artistic style (e.g., "photorealistic", "watercolor painting", "3D render", "flat illustration")
                            • Include composition details (e.g., "wide-angle shot", "close-up", "aerial view")
                            • Describe lighting and atmosphere (e.g., "soft morning light", "dramatic shadows", "bright and airy")
                            • Mention colors and mood (e.g., "warm tones", "vibrant colors", "muted palette")

                            DON'T:
                            • Use keyword lists or tags - write in complete sentences
                            • Include text/words to appear in the image (Imagen doesn't render text well)
                            • Request specific brands, logos, or copyrighted characters
                            • Ask for multiple unrelated objects in one scene - keep it cohesive

                            Example: "Photorealistic wide-angle shot of a modern office space with large windows overlooking a city skyline. Soft afternoon sunlight streaming through the windows creating warm shadows. Professional atmosphere with plants and minimalist furniture."
                            """
                        }
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "image_uploader",
                "description": "Upload a local image to Supabase bucket and get public URL",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "local_path": {
                            "type": "string",
                            "description": "Local path to the image file to upload"
                        },
                        "file_name": {
                            "type": "string",
                            "description": "Optional custom filename for the uploaded image"
                        }
                    },
                    "required": ["local_path"]
                }
            },
            {
                "name": "blog_creator",
                "description": "Create a blog post and save it to CSV file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Blog title (max 255 chars)"
                        },
                        "slug": {
                            "type": "string",
                            "description": "URL-friendly version of the title"
                        },
                        "meta_title": {
                            "type": "string",
                            "description": "SEO meta title (max 100 chars)"
                        },
                        "meta_description": {
                            "type": "string",
                            "description": "SEO meta description (max 255 chars)"
                        },
                        "content": {
                            "type": "string",
                            "description": "Full blog content in HTML format (use <h2>, <h3>, <p>, <ul>, <li>, <strong>, <em> tags)"
                        },
                        "excerpt": {
                            "type": "string",
                            "description": "Short summary of the blog (max 500 chars)"
                        },
                        "featured_image": {
                            "type": "string",
                            "description": "URL of the featured image from image_uploader"
                        },
                        "category": {
                            "type": "string",
                            "description": "Blog category"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of relevant tags"
                        },
                        "author": {
                            "type": "string",
                            "description": "Author name",
                            "default": "ReplyDaddy Team"
                        }
                    },
                    "required": ["title", "slug", "meta_title", "meta_description", "content", "excerpt", "featured_image", "category", "tags"]
                }
            },
            {
                "name": "blog_inserter",
                "description": "Insert a blog from CSV file into Supabase database",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "csv_file_path": {
                            "type": "string",
                            "description": "Path to the CSV file returned from blog_creator"
                        }
                    },
                    "required": ["csv_file_path"]
                }
            }
        ]

    def get_system_prompt(self, user_provided_topic: bool = False) -> str:
        """
        Get the system prompt for the agent

        Args:
            user_provided_topic: Whether the user provided a specific topic
        """
        # Use exact prompts from original seobot_ai.py
        if user_provided_topic:
            return """You are an expert SEO content strategist and blog writer for ReplyDaddy.com.
    You have access to tools for web search, image generation, blog creation, and database insertion.
    Your goal is to create high-quality, SEO-optimized blog content with custom AI-generated images.

    You should:
    1. First analyze the brand context and existing blogs
    2. Focus on the specific topic provided by the user
    3. Use web_search to gather current data and insights about this topic - SAVE ALL SOURCE URLS
    4. Generate a custom blog header image:
       - Use image_generator with a DESCRIPTIVE NARRATIVE prompt (not keywords!). Example: 'A modern office workspace with a laptop displaying Reddit's interface, warm natural lighting streaming through windows, creating an inspiring atmosphere for digital marketing'
       - Then use image_uploader with the returned local_path to upload it and get the public URL
    5. Create comprehensive blog content using blog_creator tool with:
       - The uploaded image URL as featured_image
       - Content must include these sections IN THE HTML:
         * Main article body with inline citations [1], [2] for all facts and statistics
         * FAQ section: <h2>Frequently Asked Questions</h2> with 5+ Q&As in HTML format
         * References section: <h2>References</h2> with numbered list of all sources
    6. IMPORTANT: After blog_creator returns success, you MUST use blog_inserter tool with the file_path to insert the blog into the database

    Be creative with categories - choose what fits best for the content you're creating.
    The task is ONLY complete after successfully inserting the blog into Supabase."""
        else:
            return """You are an expert SEO content strategist and blog writer for ReplyDaddy.com.
    You have access to tools for web search, image generation, blog creation, and database insertion.
    Your goal is to create high-quality, SEO-optimized blog content with custom AI-generated images.

    You should:
    1. First analyze the brand context and existing blogs
    2. Think of a unique, valuable blog topic that hasn't been covered
    3. Use web_search to gather current data and insights - SAVE ALL SOURCE URLS
    4. Generate a custom blog header image:
       - Use image_generator with a DESCRIPTIVE NARRATIVE prompt (not keywords!). Example: 'A modern office workspace with a laptop displaying Reddit's interface, warm natural lighting streaming through windows, creating an inspiring atmosphere for digital marketing'
       - Then use image_uploader with the returned local_path to upload it and get the public URL
    5. Create comprehensive blog content using blog_creator tool with:
       - The uploaded image URL as featured_image
       - Content must include these sections IN THE HTML:
         * Main article body with inline citations [1], [2] for all facts and statistics
         * FAQ section: <h2>Frequently Asked Questions</h2> with 5+ Q&As in HTML format
         * References section: <h2>References</h2> with numbered list of all sources
    6. IMPORTANT: After blog_creator returns success, you MUST use blog_inserter tool with the file_path to insert the blog into the database

    Be creative with categories - choose what fits best for the content you're creating.
    The task is ONLY complete after successfully inserting the blog into Supabase."""

    def generate_blog(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a complete blog post

        Args:
            topic: Optional specific topic for the blog. If None, generates automatically.

        Returns:
            Dictionary with blog generation results
        """
        try:
            # Send starting progress
            self.send_progress("start", {"message": "Starting blog generation..."})

            # Format existing blogs summary
            existing_blogs_summary = "\n".join([
                f"- {blog['title']} (Category: {blog.get('category', 'N/A')})"
                for blog in self.existing_blogs[:20]
            ]) if self.existing_blogs else "No existing blogs found."

            # Create dynamic instructions based on user input (exact from original)
            if topic:
                topic_instruction = f"""1. Analyze the brand and existing content
2. Create a comprehensive blog about: "{topic}" """
                topic_context = f" specifically related to '{topic}'"
                self.send_progress("topic", {"topic": topic, "auto_generated": False})
            else:
                topic_instruction = """1. Analyze the brand and existing content
2. Think of a NEW, unique blog idea that would be valuable for our audience"""
                topic_context = ""
                self.send_progress("topic", {"topic": "Auto-generating based on trends", "auto_generated": True})

            # Use exact user message format from original
            user_message = f"""You are an SEO expert Here's your task:

BRAND CONTEXT:
{self.brand_context}

EXISTING BLOGS (last 20 - avoid duplicating these):
{existing_blogs_summary}

YOUR MISSION:
{topic_instruction}
3. Use web_search tool to research current trends, statistics, and insights{topic_context}
4. Generate a custom header image for your blog:
   - Use image_generator with a SCENE DESCRIPTION (not keywords!). Describe it like a photograph: subject, setting, lighting, mood, composition
   - Then use image_uploader with the local_path to get the public URL
5. Create a comprehensive 2000-3000 word blog using blog_creator tool with:
   - SEO-optimized title and meta tags
   - Engaging, informative content with statistics and examples
   - Proper HTML formatting (use <h2>, <h3>, <p>, <ul>, <li>, <strong>, <em> tags)
   - NO Markdown - use HTML tags for all formatting
   - Dynamic category that you choose based on the content
   - Relevant tags for discoverability
   - The uploaded image URL as the featured_image
   - FAQ SECTION: Add an <h2>Frequently Asked Questions</h2> section at the end with at least 5 Q&As in HTML format
   - CITATIONS: Include inline citations [1], [2] etc. for all statistics and claims
   - SOURCES: List all source URLs at the end in a <h2>References</h2> section
6. IMPORTANT: After creating the blog with blog_creator, you MUST use blog_inserter with the returned file_path to add it to our database

Be creative and provide genuine value to readers!

IMPORTANT CONTENT STRUCTURE EXAMPLES:
Your content HTML must follow this exact structure:

1. Main article body with inline citations like:
   <p>According to recent studies, Reddit has over 500 million monthly users [1], making it...</p>

2. FAQ Section (REQUIRED):
   <h2>Frequently Asked Questions</h2>
   <h3>What is Reddit marketing?</h3>
   <p>Reddit marketing involves...</p>
   <h3>How much does Reddit advertising cost?</h3>
   <p>Reddit ads typically cost...</p>
   (Include at least 5 Q&As)

3. References Section (REQUIRED):
   <h2>References</h2>
   <ol>
   <li><a href="url1">Source Title 1</a></li>
   <li><a href="url2">Source Title 2</a></li>
   </ol>"""

            self.messages = [{"role": "user", "content": user_message}]

            # Get tools including web search
            tools = self.get_tool_definitions()

            # Add web search tool (using exact format from original)
            tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 10}] + tools

            # Main conversation loop
            max_iterations = 10
            iteration = 0
            blog_published = False

            while not blog_published and iteration < max_iterations:
                iteration += 1

                # Make API call
                self.send_progress("thinking", {"message": "AI is processing..."})

                response = self.anthropic.messages.create(
                    model=Config.MODEL_NAME,
                    max_tokens=16000,
                    temperature=0.7,  # Higher temperature for creative content
                    system=self.get_system_prompt(user_provided_topic=bool(topic)),
                    messages=self.messages,
                    tools=tools,
                    tool_choice={"type": "auto"}
                )

                # Process response
                assistant_message = {"role": "assistant", "content": response.content}
                self.messages.append(assistant_message)

                # Check for tool uses
                tool_results = []
                for content_block in response.content:
                    if hasattr(content_block, 'type') and content_block.type == 'tool_use':
                        tool_name = content_block.name
                        tool_input = content_block.input
                        tool_use_id = content_block.id

                        self.send_progress("tool_use", {
                            "tool": tool_name,
                            "input": tool_input
                        })

                        # Execute tool
                        if tool_name == "web_search" or tool_name == "web_search_20250305":
                            # Web search is handled by Anthropic (server-side tool)
                            result = {"status": "success", "message": "Web search completed by Anthropic"}
                        else:
                            # Execute our custom tools
                            result = self.tool_executor.execute(tool_name, tool_input)

                        # Track if blog was inserted
                        if tool_name == "blog_inserter" and result.get("status") == "success":
                            blog_published = True
                            self.send_progress("complete", {
                                "message": "Blog published successfully!",
                                "blog_url": result.get("blog_url"),
                                "slug": result.get("slug")
                            })

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": json.dumps(result)
                        })

                # Add tool results to messages if any
                if tool_results:
                    self.messages.append({
                        "role": "user",
                        "content": tool_results
                    })
                else:
                    # No tools used, check if there's text output
                    for content_block in response.content:
                        if hasattr(content_block, 'type') and content_block.type == 'text':
                            self.send_progress("message", {"text": content_block.text[:200] + "..."})

            if blog_published:
                return {
                    "status": "success",
                    "message": "Blog generated and published successfully",
                    "messages": self.messages
                }
            else:
                return {
                    "status": "warning",
                    "message": "Blog generation completed but not published to database",
                    "messages": self.messages
                }

        except Exception as e:
            self.send_progress("error", {"message": str(e)})
            return {
                "status": "error",
                "message": f"Error generating blog: {str(e)}"
            }