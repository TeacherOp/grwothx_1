"""
Image Generation and Upload Tools

This module contains tools for generating AI images and uploading them to cloud storage.
Separated from the main script for better modularity and teaching purposes.
"""

import os
from datetime import datetime
from typing import Dict, Any
from PIL import Image
import io
from google import genai
from config import Config


def optimize_image_for_blog(image_path: str, target_width: int = 1920, target_height: int = 1080) -> str:
    """
    Optimize image for blog header use

    Resizes image to standard blog header dimensions (1920x1080) while maintaining quality.
    This ensures consistent display across all blog posts.

    Args:
        image_path: Path to the original image
        target_width: Target width in pixels (default 1920)
        target_height: Target height in pixels (default 1080)

    Returns:
        Path to the optimized image
    """
    try:
        # Open and resize image
        img = Image.open(image_path)

        # Calculate aspect ratios
        original_ratio = img.width / img.height
        target_ratio = target_width / target_height

        if original_ratio > target_ratio:
            # Image is wider - fit to height and crop width
            new_height = target_height
            new_width = int(target_height * original_ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # Crop from center
            left = (new_width - target_width) // 2
            img = img.crop((left, 0, left + target_width, target_height))
        else:
            # Image is taller - fit to width and crop height
            new_width = target_width
            new_height = int(target_width / original_ratio)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # Crop from center
            top = (new_height - target_height) // 2
            img = img.crop((0, top, target_width, top + target_height))

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
    """
    Tool: Generate image using Google Imagen 4.0 Ultra with optimized prompting

    Uses the latest Imagen model to generate high-quality blog header images.
    Images are generated in 16:9 aspect ratio, perfect for blog headers.

    Args:
        prompt: Description of the image to generate

    Returns:
        Dictionary with status, message, local_path, and other metadata
    """
    try:
        if not Config.GEMINI_API_KEY:
            return {
                "status": "error",
                "message": "GEMINI_API_KEY not configured"
            }

        client = genai.Client(api_key=Config.GEMINI_API_KEY)

        # Use the new Imagen 4.0 Ultra model
        model = "models/imagen-4.0-ultra-generate-001"

        # Enhance prompt for better blog headers following Imagen best practices
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
        local_filename = str(Config.IMAGES_DIR / f"blog_header_{timestamp}.jpg")

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


def image_uploader(local_path: str, file_name: str = None) -> Dict[str, Any]:
    """
    Tool: Upload image to Supabase bucket

    Uploads a local image file to Supabase storage and returns the public URL.
    This allows the blog to reference images from a CDN rather than local storage.

    Args:
        local_path: Path to the local image file
        file_name: Optional custom filename for the uploaded image

    Returns:
        Dictionary with status, message, and public_url if successful
    """
    try:
        # Import here to avoid circular dependency and make it optional
        from supabase import create_client

        if not all([Config.BUCKET_NAME, Config.SUPABASE_URL, Config.SUPABASE_SERVICE_KEY]):
            return {
                "status": "error",
                "message": "Supabase bucket configuration missing"
            }

        supabase = create_client(Config.SUPABASE_URL, Config.SUPABASE_SERVICE_KEY)

        # Use provided filename or generate from path
        if not file_name:
            file_name = os.path.basename(local_path)

        # Read file
        with open(local_path, 'rb') as f:
            file_data = f.read()

        # Upload to Supabase bucket
        # Path in bucket: blog-images/[filename]
        bucket_path = f"blog-images/{file_name}"

        response = supabase.storage.from_(Config.BUCKET_NAME).upload(
            path=bucket_path,
            file=file_data,
            file_options={"content-type": "image/jpeg"}
        )

        # Get public URL
        public_url = supabase.storage.from_(Config.BUCKET_NAME).get_public_url(bucket_path)

        return {
            "status": "success",
            "message": "Image uploaded successfully",
            "public_url": public_url,
            "bucket_path": bucket_path
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to upload image: {str(e)}"
        }