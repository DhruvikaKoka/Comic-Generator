"""
comic_auto_anchor.py

Goal:
- Generate a comic from text only (no user image required)
- Improve character consistency by auto-generating anchor images
  (pseudo-reference) from the textual character description and using
  them with img2img for subsequent panels.
- Works with genre templates, prints prompts for debugging.
- Low-VRAM friendly: single pipeline, attention-slicing, optional CPU offload.

Limitations:
- Cannot guarantee perfect, photo-level identity across panels without
  fine-tuning (DreamBooth/LoRA) or IP-Adapter. This approach is a practical
  compromise that improves consistency for text-only workflows.
"""

import os
import random
import math
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

# --------------------------- CONFIG ---------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "runwayml/stable-diffusion-v1-5"   # swap to SDXL if you have resources & API
LOCAL_MODEL_DIR = "saved_sd_model"             # prefer local cache
OUT_DIR = "comic_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Panel size (512 recommended for low VRAM)
WIDTH, HEIGHT = 512, 512
STEPS = 25
GUIDANCE = 7.5

# anchor generation
N_ANCHORS = 3      # number of auto-generated reference views of the character
ANCHOR_VARIATION = ["close-up", "mid shot", "three-quarter view"]

# img2img strength when using an anchor (0.5-0.7 usually works)
IMG2IMG_STRENGTH = 0.62

# text bubble and font
TB_HEIGHT = 96
try:
    FONT = ImageFont.truetype("arial.ttf", 16)
except Exception:
    FONT = ImageFont.load_default()

# genre templates
GENRE_TEMPLATES = {
    "sci-fi": {
        "scene": "high-tech laboratory with glowing consoles, cables, and holograms",
        "style": "futuristic, neon rim light, digital painting, cinematic"
    },
    "fantasy": {
        "scene": "enchanted forest with glowing runes and moss-covered stones",
        "style": "painterly, warm magical glow, dramatic lighting"
    },
    "horror": {
        "scene": "abandoned lab with flickering lights and wet floors",
        "style": "moody, high contrast, grainy, eerie"
    },
    "romance": {
        "scene": "city street at sunset, umbrellas and warm reflections",
        "style": "soft, warm cinematic, pastel tones"
    },
    "adventure": {
        "scene": "rocky mountain pass with dust and dynamic sky",
        "style": "epic, dynamic lighting, high detail"
    },
}

# --------------------------- HELPERS ---------------------------
def load_pipeline(low_vram=True):
    """
    Load a single pipeline. Prefer local cache if present.
    Enable attention slicing and cpu offload (if available) for low VRAM.
    """
    if os.path.isdir(LOCAL_MODEL_DIR):
        print("Loading model from local directory:", LOCAL_MODEL_DIR)
        pipe = StableDiffusionPipeline.from_pretrained(LOCAL_MODEL_DIR, torch_dtype=torch.float16)
    else:
        print("Downloading model from HF:", MODEL_NAME)
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
        try:
            pipe.save_pretrained(LOCAL_MODEL_DIR)
        except Exception:
            pass

    pipe.enable_attention_slicing()
    # optional CPU-offload (helps small GPUs but may add overhead)
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pass

    try:
        pipe.vae.enable_tiling()
    except Exception:
        pass

    # If CPU offload is enabled, do not call .to(DEVICE) — offload manages movement
    if hasattr(pipe, "model_cpu_offload"):
        return pipe
    else:
        return pipe.to(DEVICE)

def wrap_text(draw, text, font, max_w):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        tw = draw.textbbox((0,0), t, font=font)[2]
        if tw <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def add_dialogue(img, text, tb_height=TB_HEIGHT):
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    draw.rectangle([0, H - tb_height, W, H], fill=(255,255,255,230))
    lines = wrap_text(draw, text, FONT, W - 20)
    y = H - tb_height + 8
    for line in lines[:5]:
        draw.text((12, y), line, fill=(10,10,10,255), font=FONT)
        tb = draw.textbbox((12, y), line, font=FONT)
        y += (tb[3] - tb[1]) + 6
    return img.convert("RGB")

def framed_panel(img, border=6, border_color=(20,20,20)):
    w,h = img.size
    out = Image.new("RGB", (w + border*2, h + border*2), (240,240,240))
    out.paste(img, (border, border))
    d = ImageDraw.Draw(out)
    d.rectangle([0,0,out.size[0]-1,out.size[1]-1], outline=border_color, width=border)
    return out

def transform_reference(img, target_w=WIDTH, target_h=HEIGHT):
    """Random crop/rotate/flip to change composition but preserve identity cues."""
    w,h = img.size
    crop_scale = random.uniform(0.6, 0.95)
    cw, ch = int(w * crop_scale), int(h * crop_scale)
    left = random.randint(0, max(0, w - cw))
    top = random.randint(0, max(0, h - ch))
    crop = img.crop((left, top, left + cw, top + ch))
    angle = random.uniform(-12, 12)
    crop = crop.rotate(angle, expand=True, fillcolor=(230,230,230))
    if random.random() < 0.35:
        crop = ImageOps.mirror(crop)
    bg = Image.new("RGB", (max(crop.width, target_w), max(crop.height, target_h)), (230,230,230))
    bx = (bg.width - crop.width) // 2
    by = (bg.height - crop.height) // 2
    bg.paste(crop, (bx, by))
    return bg.resize((target_w, target_h), Image.LANCZOS)

def extract_entities(story, top_k=6):
    COMMON_STOP = set(["the","and","with","that","this","from","for","a","an","to","in","on"])
    words = [w.lower().strip() for w in story.replace("\n"," ").replace(",", " ").replace(".", " ").split() if len(w)>3]
    words = [w for w in words if w not in COMMON_STOP]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    top = [w for w,_ in sorted(freq.items(), key=lambda x:-x[1])[:top_k]]
    return top

# --------------------------- PROMPT BUILDERS ---------------------------
def build_character_prompt(base_desc, identity_lock_phrase):
    """
    base_desc: short textual description from user (e.g., "A scientist in a lab, wearing a blue coat")
    identity_lock_phrase: sentence that will be added to every prompt to enforce consistency
    """
    return f"{base_desc}. {identity_lock_phrase}. detailed face, clear clothing, photorealistic-comic mix, sharp facial features"

def build_panel_prompt(character_desc, scene_desc, sentence, entities, camera, action, style):
    include = ", ".join(entities) if entities else ""
    prompt = (
        f"Scene: {scene_desc}. Include: {include}. "
        f"Action: {action}. Camera: {camera}. "
        f"Character: {character_desc}. {sentence}. Style: {style}. "
        "High detail, crisp composition, no watermark, no text overlay."
    )
    return prompt

# --------------------------- CORE PIPELINE ---------------------------
def generate_anchors_from_text(pipe, char_text, anchors=N_ANCHORS, seed_base=1000):
    """
    Create a small set of generated images (anchors) representing the character
    in different camera poses/angles. These are used as pseudo-references for img2img later.
    """
    anchors_out = []
    identity_lock = "same face every panel, same hairstyle and clothing, consistent facial features"
    char_prompt = build_character_prompt(char_text, identity_lock)
    for i in range(anchors):
        view = ANCHOR_VARIATION[i % len(ANCHOR_VARIATION)]
        prompt = f"{char_prompt}. Camera: {view}. Close-up on character face and upper body. Cinematic lighting."
        seed = seed_base + i * 11
        print(f"[ANCHOR PROMPT {i+1}]\n{prompt}\n")
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        img = pipe(prompt, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS, guidance_scale=GUIDANCE, generator=gen).images[0]
        anchors_out.append(img)
    return anchors_out, identity_lock

def create_comic_from_text(story_text, user_character_text=None, genre="sci-fi", panels=4, use_anchors=True, seed_base=2000):
    """
    - story_text: user's story text (any length). We'll split into 'panels' pieces.
    - user_character_text: optional short textual character description. If None, character is inferred.
    - genre: one of keys in GENRE_TEMPLATES
    - use_anchors: if True, auto-generate anchors and use them for img2img references.
    """
    # 1) Prepare
    genre_cfg = GENRE_TEMPLATES.get(genre, GENRE_TEMPLATES["adventure"])
    scene_template = genre_cfg["scene"]
    style_template = genre_cfg["style"]

    # split story into sentences, ensure panels length
    sentences = [s.strip() for s in story_text.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        raise ValueError("Empty story text.")
    # simple splitting: if fewer sentences than panels, replicate last; if more, keep first N
    while len(sentences) < panels:
        sentences.append(sentences[-1])
    sentences = sentences[:panels]

    # 2) character description and anchors
    if user_character_text is None:
        # build a default inferred character prompt from story: (heuristic)
        inferred = "main protagonist, human, expressive face, mid-30s"
    else:
        inferred = user_character_text

    pipe = load_pipeline()

    anchors = []
    identity_lock_phrase = "same face every panel, same hairstyle, consistent clothing color, identifiable features"
    if use_anchors:
        anchors, identity_lock_phrase = generate_anchors_from_text(pipe, inferred, anchors=N_ANCHORS, seed_base=seed_base)

    # 3) extract core entities to force presence
    entities = extract_entities(story_text, top_k=6)
    print("Entities extracted:", entities)

    # 4) generate panels
    panels_out = []
    prev_raw = None
    for i in range(panels):
        camera = ["wide shot", "mid shot", "close-up", "three-quarter view"][i % 4]
        # pick an action heuristic from sentence
        action = "illustrate the described scene"
        s = sentences[i].lower()
        if "chase" in s or "run" in s:
            action = "dynamic running, motion blur"
        if "portal" in s or "opens" in s:
            action = "portal opening with swirling energy"
        if "fight" in s or "struggle" in s:
            action = "physical struggle, dynamic poses"

        # build prompt with locked identity phrase
        character_desc = f"{inferred}. {identity_lock_phrase}"
        prompt = build_panel_prompt(character_desc, scene_template, sentences[i], entities, camera, action, style_template + ", cinematic")
        print(f"\n[PANEL {i+1} PROMPT]\n{prompt}\n")

        gen = torch.Generator(device=DEVICE).manual_seed(seed_base + i * 31)

        # if we have anchors, use img2img with a transformed anchor/reference to preserve appearance
        if use_anchors and anchors:
            # choose an anchor (rotate through them) or prefer prev_raw if available
            ref_src = prev_raw if prev_raw is not None and random.random() < 0.6 else anchors[i % len(anchors)]
            ref_for_img2img = transform_reference(ref_src)
            # some diffusers versions support passing `image` to pipeline; handle gracefully
            try:
                panel_raw = pipe(prompt, image=ref_for_img2img, strength=IMG2IMG_STRENGTH,
                                 num_inference_steps=STEPS, guidance_scale=GUIDANCE, generator=gen).images[0]
            except TypeError:
                # fallback to plain txt2img (less ideal)
                panel_raw = pipe(prompt, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS,
                                 guidance_scale=GUIDANCE, generator=gen).images[0]
        else:
            panel_raw = pipe(prompt, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS,
                             guidance_scale=GUIDANCE, generator=gen).images[0]

        prev_raw = panel_raw
        panel_with_text = add_dialogue(panel_raw, sentences[i])
        framed = framed_panel(panel_with_text)
        save_p = os.path.join(OUT_DIR, f"panel_{i+1}.png")
        framed.save(save_p)
        print("Saved panel:", save_p)
        panels_out.append(framed)

    # combine horizontally into strip
    pw, ph = panels_out[0].size
    gutter = 14
    final_w = pw * panels + gutter * (panels - 1)
    final = Image.new("RGB", (final_w, ph), (240,240,240))
    x = 0
    for p in panels_out:
        final.paste(p, (x, 0))
        x += pw + gutter
    final_path = os.path.join(OUT_DIR, "final_comic.png")
    final.save(final_path)
    print("Final comic saved:", final_path)
    return final_path

# --------------------------- USAGE / DEMO ---------------------------
if __name__ == "__main__":
    DEMO_STORY = """
    A scientist opens a portal in a secret laboratory.
    A creature emerges from swirling energy and snarls.
    The scientist runs through glowing metallic corridors to escape.
    He slams the emergency switch and the portal collapses, trapping the creature.
    """
    # Optionally let user supply a short textual character description (None = inferred)
    character_text = None  # e.g., "male scientist with messy brown hair, blue lab coat"

    # generate, no image upload required
    out = create_comic_from_text(DEMO_STORY, user_character_text=character_text, genre="sci-fi", panels=4, use_anchors=True)
    print("Output:", out)
