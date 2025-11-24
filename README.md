🎨 AI Comic Generator
Transform any story into a multi-panel comic using NLP + Stable Diffusion v1.5
📌 Project Overview

The AI Comic Generator is a complete end-to-end system that converts a user-written story into a coherent, multi-panel digital comic.
It integrates Natural Language Processing, Stable Diffusion v1.5, and a Flask web application to generate consistent characters, meaningful scenes, and readable dialogue bubbles across panels.

This project was developed as part of an academic research/mini-project.

✨ Key Features

🧠 NLP-based story analysis

Detects characters, actions, backgrounds, and dialogues

Splits story into meaningful panels

🎨 High-quality Comic Generation

Stable Diffusion v1.5 with structured prompts

Consistent characters using deterministic seeds

Clean comic-style rendering

💬 Dialogue/Narration Rendering

Bottom caption boxes

Automatic text wrapping

🖼️ Final Comic Strip Assembly

All panels stitched into a single/combined output

🌐 Web Application (Flask)

Story input

Real-time status updates

Panel previews

Downloadable final comic

📊 Evaluation Module

Image quality

Story alignment

Character consistency

Technical quality metrics

🛠 Tech Stack

Backend & AI

Python 3.10

Stable Diffusion v1.5 (HuggingFace Diffusers)

PyTorch

Flask

Frontend

HTML

CSS

JavaScript

Supporting Tools

PIL (Pillow)

Regex

Threading

NumPy

OpenCV (optional, for evaluation)

🚀 Project Flow
User Story → Story Processing → Prompt Generation 
→ Stable Diffusion Image Generation → Dialogue Rendering 
→ Comic Strip Assembly → Web UI Output

📁 Project Structure
📦 AI-Comic-Generator
 ┣ 📂 static/                → CSS, JS, images (if any)
 ┣ 📂 templates/             → index.html (frontend UI)
 ┣ 📂 outputs/               → Generated comic panels & final strip
 ┣ 📜 app.py                 → Flask backend server
 ┣ 📜 claude_code.py         → NLP + SD generation logic
 ┣ 📜 comic_eval.py          → Evaluation script
 ┣ 📜 README.md              → Project documentation
 ┗ 📜 requirements.txt       → Python dependencies

🧩 Modules
1. Story Processing Module

Splits story into scenes

Extracts characters, actions, and backgrounds

Generates metadata for each panel

2. Stable Diffusion Generation Module

Builds prompts

Generates consistent panels

Applies negative prompts

Adds text captions

3. Flask Backend

Runs comic generation in a background thread

Provides status updates

Serves results to the UI

4. Evaluation Module

Computes image quality

Checks alignment

Measures consistency

⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2. Create Python environment
python -m venv venv
source venv/bin/activate  # for Linux/Mac
venv\Scripts\activate     # for Windows

3. Install dependencies
pip install -r requirements.txt

4. Run the Flask server
python app.py

5. Open in browser
http://127.0.0.1:5000/

🖥️ Usage

Enter a story in the text box

Click Generate Comic

Watch the progress bar update in real-time

View the generated panels

Download the final comic strip

📊 Evaluation Scores (Sample)
Metric	Score
Overall Score	86.2 / 100
Image Quality	99.0
Consistency	75.0
Story Alignment	100
Dialogue Presence	100
Panel Variety	26.3
📷 Screenshots (Add your own)
/screenshots
  │── panel1.png
  │── panel2.png
  │── comic_strip.png

🎯 Applications

Automatic comic creation

Visual storytelling

AI-assisted content creation

Educational storytelling tools

Children’s book illustrations

📝 Future Enhancements

Support for multiple characters with controlled identity

Speech bubble styling

Multiple comic layout templates

Multi-language story input

Fine-tuned model for comic-specific artistic styles

👩‍💻 Contributors

Banda Vandhana

Bandari Amulya

Dhruvika Koka
