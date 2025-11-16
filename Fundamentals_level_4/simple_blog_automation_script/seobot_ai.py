import os
import json
import csv
import base64
import mimetypes
import requests
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from supabase import create_client, Client
from anthropic import Anthropic
from slugify import slugify
from google import genai
from google.genai import types
from PIL import Image
import io

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Google Gemini/Imagen API key
BUCKET_NAME = os.getenv("BUCKET_NAME")
BUCKET_ENDPOINT = os.getenv("BUCKET_ENDPOINT")
BUCKET_REGION = os.getenv("BUCKET_REGION")

if not all([ANTHROPIC_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def load_brand_context():
    """Load brand context from file"""
    try:
        with open('brand_context.txt', 'r') as file:
            return file.read()
    except FileNotFoundError:
        print("Warning: brand_context.txt not found.")
        return "ReplyDaddy is an AI-powered Reddit marketing platform."

def fetch_existing_blogs():
    """Fetch existing published blogs from Supabase"""
    try:
        response = supabase.table('blog_posts').select('title, slug, category, excerpt').eq('status', 'published').execute()
        return response.data if response.data else []
    except Exception as e:
        print(f"Error fetching existing blogs: {e}")
        return []

def calculate_read_time(content: str) -> int:
    """Calculate read time based on word count"""
    words_per_minute = 200
    word_count = len(content.split())
    return max(1, round(word_count / words_per_minute))

def optimize_image_for_blog(image_path: str, target_width: int = 1920, target_height: int = 1080) -> str:
    """Resize and optimize image for blog header (16:9 aspect ratio)"""
    try:
        # Open the image
        img = Image.open(image_path)

        # Calculate aspect ratios
        original_ratio = img.width / img.height
        target_ratio = target_width / target_height

        # Crop to 16:9 aspect ratio if needed
        if abs(original_ratio - target_ratio) > 0.01:  # If ratios differ
            if original_ratio > target_ratio:
                # Image is wider than 16:9, crop width
                new_width = int(img.height * target_ratio)
                left = (img.width - new_width) // 2
                img = img.crop((left, 0, left + new_width, img.height))
            else:
                # Image is taller than 16:9, crop height
                new_height = int(img.width / target_ratio)
                top = (img.height - new_height) // 2
                img = img.crop((0, top, img.width, top + new_height))

        # Resize to target dimensions
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

        # Save optimized image
        base, ext = os.path.splitext(image_path)
        optimized_path = f"{base}_optimized{ext}"

        # Save with optimization
        if ext.lower() in ['.jpg', '.jpeg']:
            img.save(optimized_path, 'JPEG', quality=85, optimize=True)
        elif ext.lower() == '.png':
            img.save(optimized_path, 'PNG', optimize=True)
        else:
            img.save(optimized_path)

        return optimized_path
    except Exception as e:
        print(f"Warning: Could not optimize image: {e}")
        return image_path  # Return original if optimization fails

def image_generator(prompt: str) -> Dict[str, Any]:
    """Tool: Generate image using Google Imagen 4.0 Ultra with optimized prompting"""
    try:
        if not GEMINI_API_KEY:
            return {
                "status": "error",
                "message": "GEMINI_API_KEY not configured"
            }

        client = genai.Client(api_key=GEMINI_API_KEY)

        # Use the new Imagen 4.0 Ultra model
        model = "models/imagen-4.0-ultra-generate-001"

        # Enhance prompt for better blog headers following Imagen best practices
        # Imagen 4.0 Ultra supports various aspect ratios and high quality generation
        enhanced_prompt = f"""Professional blog header image: {prompt}

        Photorealistic capture with cinematic composition. Wide-angle perspective suitable for web banner. Modern corporate aesthetic with vibrant yet professional color palette. Soft, even lighting with clear focal point. Clean minimalist design with subtle depth. High resolution detail optimized for digital displays."""

        print(f"Generating image with Imagen 4.0 Ultra...")
        print(f"Prompt: {enhanced_prompt[:200]}...")

        # Generate image using the new API format
        result = client.models.generate_images(
            model=model,
            prompt=enhanced_prompt,
            config=dict(
                number_of_images=1,
                output_mime_type="image/jpeg",  # JPEG for blog headers
                aspect_ratio="16:9",             # Wide format for blog headers
                image_size="1K",                 # 1024x768 for 16:9 aspect ratio
            ),
        )

        # Check if image was generated
        if not result.generated_images:
            return {
                "status": "error",
                "message": "No images generated"
            }

        # Get the first (and only) generated image
        generated_image = result.generated_images[0]

        # Save locally
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        local_filename = f"generated_images/blog_header_{timestamp}.jpg"
        os.makedirs('generated_images', exist_ok=True)

        # Save the image using the built-in save method
        generated_image.image.save(local_filename)
        print(f"Image saved to: {local_filename}")

        # Optimize image for blog header (resize to 1920x1080)
        try:
            optimized_path = optimize_image_for_blog(local_filename)
            if optimized_path != local_filename:
                # Delete original if optimization succeeded
                os.remove(local_filename)
                local_filename = optimized_path
                dimensions = "1920x1080"
                message = "Image generated with Imagen 4.0 Ultra and optimized for blog header"
            else:
                dimensions = "1024x768 (16:9)"
                message = "Image generated successfully with Imagen 4.0 Ultra"
        except Exception as e:
            dimensions = "1024x768 (16:9)"
            message = f"Image generated (optimization skipped: {e})"

        return {
            "status": "success",
            "message": message,
            "local_path": local_filename,
            "mime_type": "image/jpeg",
            "dimensions": dimensions,
            "model": "Imagen 4.0 Ultra"
        }

    except Exception as e:
        print(f"Image generation error: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to generate image: {str(e)}"
        }

def image_uploader(local_path: str, file_name: Optional[str] = None) -> Dict[str, Any]:
    """Tool: Upload image to Supabase bucket"""
    try:
        if not all([BUCKET_NAME]):
            return {
                "status": "error",
                "message": "Bucket configuration missing"
            }
        
        # Read the file
        with open(local_path, 'rb') as f:
            file_data = f.read()
        
        # Get file extension
        _, ext = os.path.splitext(local_path)
        
        # Generate unique filename if not provided
        if not file_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"blog_header_{timestamp}{ext}"
        
        # Upload to Supabase storage
        bucket = supabase.storage.from_(BUCKET_NAME)
        
        # Upload file
        response = bucket.upload(
            path=f"blog-images/{file_name}",
            file=file_data,
            file_options={"content-type": mimetypes.guess_type(local_path)[0] or "image/png"}
        )
        
        # Get public URL
        public_url = bucket.get_public_url(f"blog-images/{file_name}")
        
        return {
            "status": "success",
            "message": "Image uploaded successfully",
            "public_url": public_url,
            "file_path": f"blog-images/{file_name}"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to upload image: {str(e)}"
        }

def generate_schema_markup(
    title: str,
    description: str,
    author: str,
    published_date: str,
    image_url: str,
    faq_items: List[Dict[str, str]] = None
) -> str:
    """Generate JSON-LD schema markup for SEO"""
    schema = {
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": title,
        "description": description,
        "author": {
            "@type": "Person",
            "name": author,
            "url": "https://replydaddy.com/about"
        },
        "datePublished": published_date,
        "dateModified": published_date,
        "image": image_url,
        "publisher": {
            "@type": "Organization",
            "name": "ReplyDaddy",
            "logo": {
                "@type": "ImageObject",
                "url": "https://replydaddy.com/logo.png"
            }
        },
        "mainEntityOfPage": {
            "@type": "WebPage",
            "@id": f"https://replydaddy.com/blog/{title.lower().replace(' ', '-')}"
        }
    }

    # Add FAQ schema if FAQ items provided
    if faq_items:
        faq_schema = {
            "@context": "https://schema.org",
            "@type": "FAQPage",
            "mainEntity": [
                {
                    "@type": "Question",
                    "name": item["question"],
                    "acceptedAnswer": {
                        "@type": "Answer",
                        "text": item["answer"]
                    }
                } for item in faq_items
            ]
        }
        # Return both schemas
        return f'<script type="application/ld+json">{json.dumps(schema)}</script>\n<script type="application/ld+json">{json.dumps(faq_schema)}</script>'

    return f'<script type="application/ld+json">{json.dumps(schema)}</script>'

def blog_creator(
    title: str,
    slug: str,
    meta_title: str,
    meta_description: str,
    content: str,
    excerpt: str,
    featured_image: str,
    category: str,
    tags: List[str],
    author: str = "ReplyDaddy Team"
) -> Dict[str, Any]:
    """Tool: Create blog and save to CSV"""
    try:
        slug = slugify(slug or title)

        # Extract FAQ items from content for schema generation (if needed)
        # For now, we'll generate basic schema without parsing FAQs from HTML
        schema_markup = generate_schema_markup(
            title,
            meta_description or excerpt,
            author,
            datetime.now(timezone.utc).isoformat(),
            featured_image,
            None  # FAQ schema can be added later if we parse from content
        )

        # Prepend schema markup to content
        enhanced_content = schema_markup + "\n" + content

        read_time = calculate_read_time(content)
        timestamp = datetime.now(timezone.utc).isoformat()

        blog_data = {
            'slug': slug[:255],
            'title': title[:255],
            'meta_title': (meta_title or title)[:100],
            'meta_description': (meta_description or excerpt)[:255],
            'content': enhanced_content,  # Content now includes schema markup
            'excerpt': excerpt,
            'featured_image': featured_image[:500] if featured_image else '',
            'category': category[:100],
            'tags': '{' + ','.join([f'"{tag}"' for tag in tags]) + '}',  # PostgreSQL array format
            'author': author[:100],
            'status': 'published',
            'featured': 'false',
            'read_time': str(read_time),
            'view_count': '0',
            'published_at': timestamp,
            'updated_at': timestamp,
            'created_at': timestamp
        }
        
        csv_filename = f"blog_{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = os.path.join('generated_blogs', csv_filename)
        
        os.makedirs('generated_blogs', exist_ok=True)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = blog_data.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(blog_data)
        
        return {
            "status": "success",
            "message": f"Blog created successfully: {title}",
            "file_path": csv_path,
            "slug": slug
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create blog: {str(e)}"
        }

def blog_inserter(csv_file_path: str) -> Dict[str, Any]:
    """Tool: Insert blog from CSV to Supabase"""
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            blog_data = next(reader)
        
        existing = supabase.table('blog_posts').select('id').eq('slug', blog_data['slug']).execute()
        if existing.data:
            return {
                "status": "error",
                "message": f"Blog with slug '{blog_data['slug']}' already exists"
            }
        
        tags_str = blog_data['tags'].strip('{}')
        tags_list = [tag.strip().strip('"') for tag in tags_str.split(',') if tag.strip()]
        
        blog_post = {
            'slug': blog_data['slug'],
            'title': blog_data['title'],
            'meta_title': blog_data['meta_title'],
            'meta_description': blog_data['meta_description'],
            'content': blog_data['content'],
            'excerpt': blog_data['excerpt'],
            'featured_image': blog_data['featured_image'],
            'category': blog_data['category'],
            'tags': tags_list,
            'author': blog_data['author'],
            'status': blog_data['status'],
            'featured': blog_data['featured'] == 'true',
            'read_time': int(blog_data['read_time']),
            'view_count': int(blog_data['view_count']),
            'published_at': blog_data['published_at'],
            'updated_at': blog_data['updated_at'],
            'created_at': blog_data['created_at']
        }
        
        response = supabase.table('blog_posts').insert(blog_post).execute()
        
        if response.data:
            return {
                "status": "success",
                "message": f"Blog inserted successfully",
                "url": f"https://replydaddy.com/blog/{blog_data['slug']}"
            }
        else:
            return {
                "status": "error",
                "message": "Failed to insert blog: No data returned"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error inserting blog: {str(e)}"
        }

TOOLS = [
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
                    • For people, describe appearance naturally (e.g., "woman in her 30s with curly hair wearing a blue dress")

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
                    "description": "URL-friendly slug"
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
                    "description": "Full blog content in HTML format including: main content with inline citations [1], [2], FAQ section (<h2>Frequently Asked Questions</h2>) with at least 5 Q&As, and References section (<h2>References</h2>) with numbered source list. Use proper HTML tags."
                },
                "excerpt": {
                    "type": "string",
                    "description": "Brief summary or excerpt of the blog"
                },
                "featured_image": {
                    "type": "string",
                    "description": "First create image using the image generator tool and get the url using image uploader tool URL to featured image (max 500 chars)"
                },
                "category": {
                    "type": "string",
                    "description": "Blog category (you decide based on content)"
                },
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Array of relevant tags"
                },
                "author": {
                    "type": "string",
                    "description": "Author name",
                    "default": "ReplyDaddy Team"
                }
            },
            "required": ["title", "slug", "content", "excerpt", "category", "tags"]
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
                    "description": "Path to the CSV file containing blog data"
                }
            },
            "required": ["csv_file_path"]
        }
    }
]

def handle_tool_call(tool_name: str, tool_args: Dict) -> Dict:
    """Handle tool calls from AI"""
    if tool_name == "blog_creator":
        return blog_creator(**tool_args)
    elif tool_name == "blog_inserter":
        return blog_inserter(**tool_args)
    elif tool_name == "image_generator":
        return image_generator(**tool_args)
    elif tool_name == "image_uploader":
        return image_uploader(**tool_args)
    else:
        # Web search is handled natively by Anthropic, no need to handle it here
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}

def main():
    print("🚀 Starting AI Blog Generator...")
    print("-" * 50)
    
    # Interactive prompt for article topic
    print("\n📝 Article Topic Selection:")
    print("Press ENTER to auto-generate a topic, or type a specific topic/title:")
    user_topic = input("> ").strip()
    
    if user_topic:
        print(f"\n✅ Generating article about: {user_topic}")
    else:
        print("\n🔄 Auto-generating topic based on trends and existing content...")
    
    print("-" * 50)
    
    brand_context = load_brand_context()
    existing_blogs = fetch_existing_blogs()
    
    existing_blogs_summary = "\n".join([
        f"- {blog['title']} (Category: {blog.get('category', 'N/A')})"
        for blog in existing_blogs[:20]
    ])
    
    # Modify system prompt based on whether user provided a topic
    if user_topic:
        system_prompt = """You are an expert SEO content strategist and blog writer for ReplyDaddy.com. 
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
        system_prompt = """You are an expert SEO content strategist and blog writer for ReplyDaddy.com. 
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
    
    # Create dynamic instructions based on user input
    if user_topic:
        topic_instruction = f"""1. Analyze the brand and existing content
2. Create a comprehensive blog about: "{user_topic}" """
        topic_context = f" specifically related to '{user_topic}'"
    else:
        topic_instruction = """1. Analyze the brand and existing content
2. Think of a NEW, unique blog idea that would be valuable for our audience"""
        topic_context = ""
    
    user_message = f"""You are an SEO expert. Here's your task:

BRAND CONTEXT:
{brand_context}

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

IMPORTANT CONTENT STRUCTURE:
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

    try:
        print("🤖 AI is thinking and creating content...")
        # Add web search as a native tool alongside custom tools
        all_tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 10}] + TOOLS
        
        # Start the conversation
        messages = [{"role": "user", "content": user_message}]
        
        # Keep track of whether the blog has been inserted
        blog_inserted = False
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while not blog_inserted and iteration < max_iterations:
            iteration += 1
            
            # Make API call
            response = anthropic.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=16000,
                temperature=0,
                system=system_prompt,
                messages=messages,
                tools=all_tools,
                tool_choice={"type": "auto"}
            )
            
            # Process response
            assistant_message = {"role": "assistant", "content": []}
            tool_results = []
            
            for content in response.content:
                if content.type == "text":
                    print(f"\n💭 AI: {content.text[:200]}...")
                    assistant_message["content"].append({
                        "type": "text",
                        "text": content.text
                    })
                    
                elif content.type == "tool_use":
                    print(f"\n🔧 Using tool: {content.name}")
                    
                    # Handle the tool call
                    result = handle_tool_call(content.name, content.input)
                    print(f"   Result: {result['message']}")
                    
                    # Add tool use to assistant message
                    assistant_message["content"].append({
                        "type": "tool_use",
                        "id": content.id,
                        "name": content.name,
                        "input": content.input
                    })
                    
                    # Add tool result
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content.id,
                        "content": json.dumps(result)
                    })
                    
                    # Check if blog was inserted
                    if content.name == "blog_inserter" and result["status"] == "success":
                        blog_inserted = True
                        print(f"\n✅ Blog published at: {result['url']}")
            
            # Add assistant message to conversation
            messages.append(assistant_message)
            
            # If there were tool calls, add the results and continue
            if tool_results:
                messages.append({
                    "role": "user",
                    "content": tool_results
                })
                print("\n🔄 Continuing conversation...")
            else:
                # No tool calls, conversation is done
                break
        
        if blog_inserted:
            print("\n✨ Blog successfully created and published!")
        else:
            print("\n⚠️ Blog creation process completed but blog was not inserted")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())