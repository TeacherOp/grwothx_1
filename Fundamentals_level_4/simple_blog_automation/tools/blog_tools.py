"""
Blog Creation and Database Tools

This module contains tools for creating blog posts and inserting them into the database.
Separated from the main script for better modularity and teaching purposes.
"""

import os
import csv
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from slugify import slugify
from config import Config


def fetch_existing_blogs() -> List[Dict[str, Any]]:
    """
    Fetch existing published blogs from Supabase

    This function retrieves all published blog titles to prevent duplicate content generation.
    It helps the AI understand what content already exists.

    Returns:
        List of existing blog posts with title, slug, category, and excerpt
    """
    try:
        # Import here to avoid circular dependency and make it optional
        from supabase import create_client

        if not all([Config.SUPABASE_URL, Config.SUPABASE_SERVICE_KEY]):
            print("Supabase not configured - cannot fetch existing blogs")
            return []

        supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_SERVICE_KEY)

        response = supabase.table('blog_posts').select('title, slug, category, excerpt').eq('status', 'published').execute()

        if response.data:
            print(f"Found {len(response.data)} existing published blogs")
            return response.data
        return []

    except Exception as e:
        print(f"Error fetching existing blogs: {e}")
        return []


def calculate_read_time(content: str) -> int:
    """
    Calculate estimated reading time based on word count

    Uses average reading speed of 200 words per minute.

    Args:
        content: The blog content text

    Returns:
        Estimated reading time in minutes (minimum 1)
    """
    words_per_minute = 200
    word_count = len(content.split())
    return max(1, round(word_count / words_per_minute))


def generate_schema_markup(
    title: str,
    description: str,
    author: str,
    published_date: str,
    image_url: str,
    slug: str,
    faq_items: List[Dict[str, str]] = None
) -> str:
    """
    Generate JSON-LD schema markup for SEO

    Creates structured data that helps search engines understand the content better.
    This improves search visibility and can enable rich snippets.

    Args:
        title: Article title
        description: Article description/excerpt
        author: Author name
        published_date: ISO format publication date
        image_url: Featured image URL
        slug: URL slug for the article
        faq_items: Optional list of FAQ items with 'question' and 'answer' keys

    Returns:
        HTML script tags with JSON-LD structured data
    """
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
            "@id": f"https://replydaddy.com/blog/{slug}"
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
    content: str,
    excerpt: str,
    category: str,
    tags: list,
    featured_image: str = "",
    meta_title: Optional[str] = None,
    meta_description: Optional[str] = None,
    author: str = "ReplyDaddy Team"
) -> Dict[str, Any]:
    """
    Tool: Create a blog post and save it to CSV file

    Creates a blog post with all metadata and saves it to a CSV file.
    The CSV format matches the Supabase blog_posts table structure.

    Args:
        title: Blog title (max 255 chars)
        slug: URL-friendly version of the title
        content: Full blog content in HTML format
        excerpt: Short summary (max 500 chars)
        category: Blog category
        tags: List of tags
        featured_image: URL of the featured image
        meta_title: SEO meta title
        meta_description: SEO meta description
        author: Author name

    Returns:
        Dictionary with status, message, and file_path
    """
    try:
        from slugify import slugify

        # Ensure slug is URL-friendly
        slug = slugify(slug or title)

        # Generate schema markup if not already in content
        published_date = datetime.now(timezone.utc).isoformat()
        schema_markup = generate_schema_markup(
            title,
            meta_description or excerpt,
            author,
            published_date,
            featured_image,
            slug,
            None  # FAQ items can be parsed from content if needed
        )

        # Prepend schema markup to content
        enhanced_content = schema_markup + "\n" + content

        # Calculate read time
        read_time = calculate_read_time(content)

        # Create blog data matching Supabase table structure
        blog_data = {
            'slug': slug[:255],
            'title': title[:255],
            'meta_title': (meta_title or title)[:100],
            'meta_description': (meta_description or excerpt)[:255],
            'content': enhanced_content,
            'excerpt': excerpt[:500],
            'featured_image': featured_image[:500] if featured_image else '',
            'category': category[:100],
            'tags': '{' + ','.join([f'"{tag}"' for tag in tags]) + '}',  # PostgreSQL array format
            'author': author[:100],
            'status': 'published',
            'featured': 'false',
            'read_time': str(read_time),
            'view_count': '0',
            'published_at': published_date,
            'updated_at': published_date,
            'created_at': published_date
        }

        # Create a unique CSV file for this blog (matching original behavior)
        csv_filename = f"blog_{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = Config.DATA_DIR / csv_filename

        # Write to CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = blog_data.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(blog_data)

        # Also save as markdown file for reference
        markdown_file = Config.BLOGS_DIR / f"{slug}.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            f.write(f"**Author:** {author}\n")
            f.write(f"**Category:** {category}\n")
            f.write(f"**Tags:** {', '.join(tags) if isinstance(tags, list) else tags}\n")
            f.write(f"**Published:** {published_date}\n\n")
            if featured_image:
                f.write(f"![Featured Image]({featured_image})\n\n")
            f.write(f"## {excerpt}\n\n")
            f.write(content)

        print(f"Blog saved locally: {markdown_file}")
        print(f"CSV saved: {csv_path}")

        return {
            "status": "success",
            "message": f"Blog created successfully: {title}",
            "file_path": str(csv_path),  # IMPORTANT: Return file_path for blog_inserter
            "slug": slug
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create blog: {str(e)}"
        }


def blog_inserter(csv_file_path: str) -> Dict[str, Any]:
    """
    Tool: Insert a blog from CSV file into Supabase database

    Reads the blog data from CSV and inserts it into the Supabase database.
    This provides persistence and allows the blog to be displayed on the website.

    Args:
        csv_file_path: Path to the CSV file containing blog data

    Returns:
        Dictionary with status, message, and URL if successful
    """
    try:
        # Import here to avoid circular dependency and make it optional
        from supabase import create_client

        if not all([Config.SUPABASE_URL, Config.SUPABASE_SERVICE_KEY]):
            return {
                "status": "warning",
                "message": "Supabase not configured - blog saved locally only"
            }

        supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_SERVICE_KEY)

        # Read the blog data from CSV
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            blog_data = next(reader)  # Get the first (and only) row

        # Check if blog already exists
        existing = supabase.table('blog_posts').select('id').eq('slug', blog_data['slug']).execute()
        if existing.data:
            return {
                "status": "error",
                "message": f"Blog with slug '{blog_data['slug']}' already exists"
            }

        # Parse tags from PostgreSQL array format
        tags_str = blog_data['tags'].strip('{}')
        tags_list = [tag.strip().strip('"') for tag in tags_str.split(',') if tag.strip()]

        # Prepare blog post data for Supabase
        blog_post = {
            'slug': blog_data['slug'],
            'title': blog_data['title'],
            'meta_title': blog_data['meta_title'],
            'meta_description': blog_data['meta_description'],
            'content': blog_data['content'],
            'excerpt': blog_data['excerpt'],
            'featured_image': blog_data['featured_image'],
            'category': blog_data['category'],
            'tags': tags_list,  # Array of strings for PostgreSQL
            'author': blog_data['author'],
            'status': blog_data['status'],
            'featured': blog_data['featured'] == 'true',
            'read_time': int(blog_data['read_time']),
            'view_count': int(blog_data['view_count']),
            'published_at': blog_data['published_at'],
            'updated_at': blog_data['updated_at'],
            'created_at': blog_data['created_at']
        }

        # Insert into Supabase
        response = supabase.table('blog_posts').insert(blog_post).execute()

        if response.data:
            blog_url = f"https://replydaddy.com/blog/{blog_data['slug']}"
            print(f"✅ Blog published at: {blog_url}")

            return {
                "status": "success",
                "message": "Blog inserted successfully",
                "url": blog_url
            }
        else:
            return {
                "status": "error",
                "message": "Failed to insert blog: No data returned"
            }

    except FileNotFoundError:
        return {
            "status": "error",
            "message": f"CSV file not found: {csv_file_path}"
        }
    except Exception as e:
        print(f"Error inserting blog: {str(e)}")
        return {
            "status": "error",
            "message": f"Error inserting blog: {str(e)}"
        }