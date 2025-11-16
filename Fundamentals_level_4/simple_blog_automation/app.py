"""
Flask Web Application for AI-Powered SEO Blog Generator

This is a simple Flask web application that demonstrates how to build
an AI agent system with a web interface for blog generation.
Students will learn:
- Flask basics (routes, templates, sessions)
- Server-Sent Events for streaming updates
- AI agent orchestration for content creation
- Real-time progress tracking
"""

import json
import queue
import threading
import traceback
import sys
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

# Import our configuration and agent
from config import Config
from agent.seo_agent import SEOBlogAgent

# Initialize Flask app
app = Flask(__name__)

# Apply configuration
app.secret_key = Config.SECRET_KEY

# Ensure directories exist
Config.ensure_directories()


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate_blog():
    """
    Generate a blog post with streaming progress updates using Server-Sent Events (SSE)

    This endpoint demonstrates how to stream real-time updates from an AI agent
    to the frontend, providing visibility into the blog generation process.
    """
    try:
        # Get topic from request (optional)
        data = request.get_json()
        topic = data.get('topic', '').strip() if data else ''

        # If empty string, set to None for auto-generation
        if not topic:
            topic = None

        # Create a queue for streaming progress events
        progress_queue = queue.Queue()

        def progress_callback(event_type, data):
            """Callback function to receive progress updates from agent"""
            progress_queue.put({'event': event_type, 'data': data})

        def generate():
            """
            Generator function for Server-Sent Events

            This function runs the blog generation in a background thread and streams
            progress updates to the client as they occur.
            """
            try:
                # Container to hold the result
                result_container = {'result': None, 'error': None}

                def run_generation():
                    """Run blog generation in background thread"""
                    try:
                        print(f"Starting blog generation - Topic: '{topic if topic else 'Auto-generate'}'")

                        # Create agent with progress callback
                        agent = SEOBlogAgent(progress_callback=progress_callback)

                        # Generate blog
                        result = agent.generate_blog(topic)
                        result_container['result'] = result

                        print(f"Blog generation completed: {result.get('status', 'unknown')}")
                    except Exception as e:
                        error_msg = f"Error in blog generation: {str(e)}"
                        print(error_msg, file=sys.stderr)
                        traceback.print_exc()
                        result_container['error'] = error_msg
                    finally:
                        # Signal completion
                        progress_queue.put({'event': 'done', 'data': {}})

                # Start generation in background thread
                thread = threading.Thread(target=run_generation)
                thread.start()

                # Stream progress events to client
                while True:
                    try:
                        # Get event from queue (wait max 0.1 seconds)
                        event = progress_queue.get(timeout=0.1)

                        if event['event'] == 'done':
                            # Generation complete
                            if result_container['error']:
                                # Send error event
                                yield f"data: {json.dumps({'event': 'error', 'data': {'message': result_container['error']}})}\n\n"
                            else:
                                # Send completion event
                                result = result_container['result']
                                yield f"data: {json.dumps({'event': 'complete', 'data': result})}\\n\\n"
                            break
                        else:
                            # Stream progress event to client
                            yield f"data: {json.dumps(event)}\\n\\n"

                    except queue.Empty:
                        # Send keepalive to prevent timeout
                        yield f": keepalive\n\n"

                        # Check if thread is done
                        if not thread.is_alive() and progress_queue.empty():
                            # Make sure to send final status if thread completed
                            if result_container['result'] or result_container['error']:
                                if result_container['error']:
                                    yield f"data: {json.dumps({'event': 'error', 'data': {'message': result_container['error']}})}\n\n"
                                else:
                                    result = result_container['result']
                                    yield f"data: {json.dumps({'event': 'complete', 'data': result})}\n\n"
                            break

            except Exception as e:
                yield f"data: {json.dumps({'event': 'error', 'data': {'message': str(e)}})}\\n\\n"

        # Return Server-Sent Events stream
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def status():
    """Check application status and configuration"""
    try:
        # Check configuration
        config_status = {
            'anthropic': bool(Config.ANTHROPIC_API_KEY),
            'gemini': bool(Config.GEMINI_API_KEY),
            'supabase': bool(Config.SUPABASE_URL and Config.SUPABASE_SERVICE_KEY),
            'brand_context': Config.BRAND_CONTEXT_FILE.exists()
        }

        return jsonify({
            'status': 'ok',
            'config': config_status,
            'directories': {
                'images': str(Config.IMAGES_DIR),
                'blogs': str(Config.BLOGS_DIR)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/brand-context', methods=['GET'])
def get_brand_context():
    """Get the current brand context"""
    try:
        brand_context = Config.load_brand_context()
        return jsonify({
            'status': 'success',
            'content': brand_context,
            'file_path': str(Config.BRAND_CONTEXT_FILE)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/brand-context', methods=['POST'])
def update_brand_context():
    """Update the brand context"""
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({'error': 'Content is required'}), 400

        new_content = data['content']

        # Save the brand context to file
        Config.BRAND_CONTEXT_FILE.write_text(new_content)

        print(f"Brand context updated successfully")

        return jsonify({
            'status': 'success',
            'message': 'Brand context updated successfully',
            'file_path': str(Config.BRAND_CONTEXT_FILE)
        })
    except Exception as e:
        print(f"Error updating brand context: {str(e)}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500


def main():
    """
    Main entry point for the Flask application

    This function:
    1. Validates configuration
    2. Ensures required directories exist
    3. Starts the Flask development server
    """

    print("\\n" + "="*60)
    print("AI-POWERED SEO BLOG GENERATOR")
    print("="*60)

    # Validate configuration
    try:
        Config.validate()
        print("✅ Configuration validated successfully")
    except ValueError as e:
        print(f"\\n❌ Error: {e}")
        print("Please check your .env file at the project root")
        return

    # Ensure required directories exist
    Config.ensure_directories()
    print("✅ Required directories created")

    # Check optional configurations
    if not Config.GEMINI_API_KEY:
        print("⚠️  Warning: GEMINI_API_KEY not configured - image generation will not work")
    if not Config.SUPABASE_URL:
        print("⚠️  Warning: Supabase not configured - blogs will be saved locally only")

    # Print startup message
    print(f"\\n🚀 Starting Flask server...")
    print(f"📝 Open your browser and navigate to:")
    print(f"   http://localhost:{Config.PORT}")
    print("\\n" + "="*60 + "\\n")

    # Run the Flask app
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )


if __name__ == "__main__":
    main()