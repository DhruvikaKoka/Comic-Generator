"""
Flask Web UI for Comic Generator
Run this file to start the web interface
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import os
import json
from datetime import datetime
import threading
import sys

# Import the ComicGenerator from your existing code
# Make sure claude_code.py (or whatever you named it) is in the same directory
from claude_code import ComicGenerator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'generated_comics'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Global variables
generator = None
generation_status = {
    'is_generating': False,
    'progress': 0,
    'message': '',
    'error': None,
    'output_folder': None
}

def initialize_model():
    """Initialize the model on startup"""
    global generator
    try:
        print("Initializing Comic Generator model...")
        generator = ComicGenerator()
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def generate_comic_async(story_text, output_folder):
    """Generate comic in background thread"""
    global generation_status
    
    try:
        generation_status['is_generating'] = True
        generation_status['progress'] = 10
        generation_status['message'] = 'Analyzing story...'
        generation_status['error'] = None
        
        # Generate comic
        panel_paths = generator.generate_comic(story_text, output_path=output_folder)
        
        generation_status['progress'] = 100
        generation_status['message'] = 'Comic generated successfully!'
        generation_status['output_folder'] = output_folder
        generation_status['is_generating'] = False
        
    except Exception as e:
        generation_status['error'] = str(e)
        generation_status['is_generating'] = False
        generation_status['message'] = f'Error: {str(e)}'
        print(f"Error generating comic: {e}")
        import traceback
        traceback.print_exc()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    """API endpoint to generate comic"""
    global generation_status
    
    if generation_status['is_generating']:
        return jsonify({'error': 'Another comic is being generated. Please wait.'}), 429
    
    data = request.json
    story_text = data.get('story', '')
    
    if not story_text or len(story_text.strip()) < 10:
        return jsonify({'error': 'Please provide a story (at least 10 characters)'}), 400
    
    # Create output folder with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = os.path.join(app.config['UPLOAD_FOLDER'], f'comic_{timestamp}')
    os.makedirs(output_folder, exist_ok=True)
    
    # Reset status
    generation_status = {
        'is_generating': True,
        'progress': 0,
        'message': 'Starting generation...',
        'error': None,
        'output_folder': output_folder
    }
    
    # Start generation in background thread
    thread = threading.Thread(target=generate_comic_async, args=(story_text, output_folder))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Comic generation started',
        'output_folder': output_folder
    })

@app.route('/api/status')
def status():
    """Get generation status"""
    return jsonify(generation_status)

@app.route('/api/result/<path:folder>')
def get_result(folder):
    """Get generated comic results"""
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder.split('/')[-1])
    
    if not os.path.exists(folder_path):
        return jsonify({'error': 'Comic not found'}), 404
    
    # Get all panel images
    panels = []
    for file in sorted(os.listdir(folder_path)):
        if file.startswith('panel_') and file.endswith('.png'):
            panels.append(f'/api/image/{folder.split("/")[-1]}/{file}')
    
    # Check for comic strip
    comic_strip = None
    if os.path.exists(os.path.join(folder_path, 'comic_strip.png')):
        comic_strip = f'/api/image/{folder.split("/")[-1]}/comic_strip.png'
    
    return jsonify({
        'panels': panels,
        'comic_strip': comic_strip,
        'total_panels': len(panels)
    })

@app.route('/api/image/<folder>/<filename>')
def get_image(folder, filename):
    """Serve generated images"""
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
    return send_from_directory(folder_path, filename)

@app.route('/api/download/<folder>')
def download_comic(folder):
    """Download the complete comic strip"""
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
    comic_path = os.path.join(folder_path, 'comic_strip.png')
    
    if not os.path.exists(comic_path):
        return jsonify({'error': 'Comic strip not found'}), 404
    
    return send_file(comic_path, as_attachment=True, download_name=f'{folder}_comic.png')

@app.route('/api/examples')
def get_examples():
    """Get example stories"""
    examples = {
        'fantasy': """A brave knight named Sir Roland rode through the enchanted forest on his white horse. 
In a forest clearing, the knight met an elderly wizard named Merlin.
"Beware the dragon ahead!" warned the wizard urgently.
"I fear no beast," replied Sir Roland with confidence.
The knight and wizard journeyed together toward the distant mountains.
Near a dark cave, the massive dragon appeared and roared fiercely.
"Prepare for battle!" shouted the wizard to the knight.
The knight confronted the dragon with his sword raised high.
After an intense battle, the knight defeated the dragon victoriously.
"We make a great team," said the wizard with a wise smile.""",
        
        'scifi': """The astronaut and robot explored the alien planet together.
"Look at those strange plants!" said the astronaut excitedly.
The robot scanned the environment with its sensors.
Suddenly, a friendly alien appeared from behind a rock.
"Welcome to our planet," said the alien with a smile.
The astronaut and alien shook hands in friendship.""",
        
        'detective': """The detective investigated the mysterious mansion carefully.
"I found a clue!" shouted the detective, holding a magnifying glass.
A thief appeared suddenly from the shadows.
The detective chased the thief through the city streets.
After a long chase, the detective caught the thief.
"Case solved!" said the detective triumphantly.""",
        
        'adventure': """The pirate captain explored the treasure island with his crew.
"There must be treasure here!" said the pirate excitedly.
An explorer appeared from the jungle with a map.
"I can help you find it," said the explorer.
The pirate and explorer worked together to dig for treasure.
They discovered a chest full of golden coins and jewels."""
    }
    
    return jsonify(examples)

if __name__ == '__main__':
    # Create upload folder
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("=" * 70)
    print("COMIC GENERATOR WEB UI")
    print("=" * 70)
    
    # Initialize model
    if initialize_model():
        print("\n✅ Model loaded successfully!")
        print("🚀 Starting web server...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("=" * 70)
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    else:
        print("\n❌ Failed to load model. Please check your installation.")
        sys.exit(1)