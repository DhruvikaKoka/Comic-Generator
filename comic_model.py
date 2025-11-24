"""
comic_model.py
FULL comic generator:
✔ Saves Stable Diffusion locally (HF → folder)
✔ Saves config as .pkl
✔ Multi-genre support (adventure, sci-fi, fantasy, horror, mystery, romance)
✔ Auto genre detection or user-selected genre
✔ Works with ANY user story (auto split into panels)
✔ Returns final comic path (UI-ready)
"""

import os
import pickle
import torch
import random
from PIL import Image, ImageDraw, ImageFont, ImageOps
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

# ====================================================================
# PATHS & CONSTANTS
# ====================================================================
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
SAVE_DIR = "saved_sd_model"
CONFIG_PATH = "comic_config.pkl"
OUT_DIR = "comic_outputs"

os.makedirs(OUT_DIR, exist_ok=True)

WIDTH = 640
HEIGHT = 640
GUIDANCE = 7.5
STEPS = 25
TB_HEIGHT = 110

try:
    FONT = ImageFont.truetype("arial.ttf", 20)
except:
    FONT = ImageFont.load_default()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====================================================================
# MULTI-GENRE CONFIG
# ====================================================================

DEFAULT_CONFIG = {
    "genres": {
        "adventure": {
            "character_desc": "Two rugged explorers with hats and gear",
            "scenes": [
                "wide shot, desert ruins, digging site",
                "villain ambush behind rocks",
                "mountain chase, dust flying",
                "sunset victory with treasure"
            ]
        },
        "sci-fi": {
            "character_desc": "Two futuristic space explorers in armored suits",
            "scenes": [
                "spaceship bridge, holograms glowing",
                "alien creature attack, neon lighting",
                "zero-gravity pursuit in space station",
                "final reactor showdown"
            ]
        },
        "fantasy": {
            "character_desc": "A wizard and a knight, medieval armor & runes",
            "scenes": [
                "enchanted forest, glowing lights",
                "dragon swoops down breathing fire",
                "horse chase through battlefield ruins",
                "champions holding magical artifact"
            ]
        },
        "mystery": {
            "character_desc": "Detective in trench coat with assistant",
            "scenes": [
                "rainy alley, fog, lamppost",
                "suspect fleeing crime scene",
                "detective chase between buildings",
                "suspect caught and interrogated"
            ]
        },
        "romance": {
            "character_desc": "Young couple in warm cinematic lighting",
            "scenes": [
                "sunset walk through city street",
                "jealous rival confrontation",
                "dramatic rain scene chase",
                "reconciliation hug under umbrella"
            ]
        },
        "horror": {
            "character_desc": "Two survivors with flashlights, terrified expressions",
            "scenes": [
                "abandoned house, dark hallways",
                "monster charge from shadows",
                "sprinting down basement stairs",
                "escaping into sunrise"
            ]
        }
    },
    "default_panels": 4,
    "model_path": SAVE_DIR
}


# ====================================================================
# CONFIG SAVE / LOAD
# ====================================================================

def save_config():
    with open(CONFIG_PATH, "wb") as f:
        pickle.dump(DEFAULT_CONFIG, f)
    print("Saved config:", CONFIG_PATH)


def load_config():
    with open(CONFIG_PATH, "rb") as f:
        return pickle.load(f)


# ====================================================================
# DOWNLOAD + SAVE MODEL
# ====================================================================

def save_stable_diffusion_model():
    print("Downloading Stable Diffusion model…")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
    )
    pipe.save_pretrained(SAVE_DIR)
    print("Model saved at:", SAVE_DIR)


# ====================================================================
# LOAD SD MODEL FROM FOLDER
# ====================================================================

def load_sd_pipelines():
    print("Loading Stable Diffusion pipelines…")

    txt2img = StableDiffusionPipeline.from_pretrained(
        SAVE_DIR,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        SAVE_DIR,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    return txt2img, img2img


# ====================================================================
# GENRE DETECTION (if user does NOT choose)
# ====================================================================

def detect_genre(story):
    s = story.lower()

    keywords = {
        "sci-fi": ["spaceship", "alien", "robot", "galaxy", "laser"],
        "fantasy": ["dragon", "wizard", "magic", "castle"],
        "horror": ["blood", "monster", "haunted", "scream"],
        "romance": ["kiss", "love", "date", "romantic"],
        "mystery": ["detective", "murder", "crime", "investigation"],
        "adventure": ["treasure", "jungle", "ruins", "map", "explorer"]
    }

    for genre, words in keywords.items():
        if any(w in s for w in words):
            return genre

    return "adventure"   # fallback


# ====================================================================
# TEXT BUBBLE
# ====================================================================

def add_text(img, text):
    W, H = img.size
    base = img.convert("RGBA")
    draw = ImageDraw.Draw(base)

    draw.rectangle([0, H-TB_HEIGHT, W, H], fill=(255,255,255,230))
    y = H - TB_HEIGHT + 10

    # wrap text
    words = text.split()
    line = ""
    for w in words:
        if draw.textbbox((0,0), line + " " + w, font=FONT)[2] < W - 20:
            line += " " + w
        else:
            draw.text((10, y), line.strip(), fill="black", font=FONT)
            y += 28
            line = w
    draw.text((10, y), line.strip(), fill="black", font=FONT)

    return base.convert("RGB")


# ====================================================================
# GENERATION FUNCTIONS
# ====================================================================

def generate_txt2img(pipe, prompt, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                generator=gen).images[0]


def generate_img2img(pipe, prompt, ref, seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    return pipe(prompt=prompt,
                image=ref,
                strength=0.65,
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                generator=gen).images[0]


# ====================================================================
# MAIN COMIC GENERATOR
# ====================================================================

def generate_comic_from_story(story_text, genre=None):
    cfg = load_config()

    if genre is None:
        genre = detect_genre(story_text)

    genre_data = cfg["genres"][genre]
    panels = cfg["default_panels"]

    char_desc = genre_data["character_desc"]
    scene_templates = genre_data["scenes"]

    sentences = [s.strip() for s in story_text.replace("\n"," ").split(".") if s.strip()]
    if len(sentences) < panels:
        sentences += [sentences[-1]] * (panels - len(sentences))
    sentences = sentences[:panels]

    txt2img, img2img = load_sd_pipelines()

    images = []
    prev = None
    seed_base = 1234

    for i in range(panels):
        prompt = f"{char_desc}. {scene_templates[i]}. {sentences[i]}. comic style, detailed lineart."

        if i == 0:
            out = generate_txt2img(txt2img, prompt, seed_base+i)
        else:
            out = generate_img2img(img2img, prompt, prev, seed_base+i)

        prev = out
        final = add_text(out, sentences[i])
        images.append(final)

        final.save(f"{OUT_DIR}/panel_{i+1}.png")

    # combine horizontally
    W, H = images[0].size
    combined = Image.new("RGB", (W * panels, H))
    x = 0
    for img in images:
        combined.paste(img, (x, 0))
        x += W

    final_path = f"{OUT_DIR}/final_comic.png"
    combined.save(final_path)
    return final_path, genre


# ====================================================================
# SELF TEST
# ====================================================================
if __name__ == "__main__":
    print("---- First-time Setup ----")

    # Download + save SD model locally (runs only once)
    save_stable_diffusion_model()
    save_config()

    TEST_STORY = """
    Two men dig up old buried treasure.
    A villain tries to steal it.
    They chase him through the mountains.
    Finally they recover the treasure.
    """

    print("Generating test comic...")
    final_path, used_genre = generate_comic_from_story(TEST_STORY, genre="adventure")

    print("Saved comic:", final_path)
    print("Used genre:", used_genre)

