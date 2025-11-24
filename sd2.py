"""
final_consistent_comic.py
-----------------------------------------
✔ ONE image per panel (no grids, no collage)
✔ Character consistency across panels
✔ Bigger dialogue text
✔ Pure txt2img (4GB GPU safe)
-----------------------------------------
"""

import os
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
LOCAL_DIR = "saved_sd_model"

WIDTH = 512
HEIGHT = 512
STEPS = 25
GUIDANCE = 7.5
OUT_DIR = "comic_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# BIGGER TEXT SETTINGS
TB_HEIGHT = 140
try:
    FONT = ImageFont.truetype("arial.ttf", 26)
except:
    FONT = ImageFont.load_default()

# Strong negative prompt to stop replacing characters
NEGATIVE = (
    "no multiple people, no different characters, no collage, no grid, "
    "no duplicate frames, no speech bubbles, no text, no watermark, "
    "no character changes, no different faces"
)


# ----------------------------- LOAD PIPE -----------------------------
def load_pipe():
    if os.path.exists(LOCAL_DIR):
        pipe = StableDiffusionPipeline.from_pretrained(
            LOCAL_DIR, torch_dtype=torch.float16
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
        )
        pipe.save_pretrained(LOCAL_DIR)

    pipe.enable_attention_slicing()
    try: pipe.enable_model_cpu_offload()
    except: pass

    return pipe.to(DEVICE)


# ----------------------------- DRAW CAPTION -----------------------------
def draw_caption(img, text):
    img = img.convert("RGBA")
    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")

    d.rectangle([0, H - TB_HEIGHT, W, H], fill=(255,255,255,235))

    words = text.split()
    lines = []
    cur = ""

    for w in words:
        t = (cur + " " + w).strip()
        if d.textbbox((0,0), t, font=FONT)[2] <= W - 40:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    y = H - TB_HEIGHT + 15
    for line in lines[:3]:
        d.text((15, y), line, fill="black", font=FONT)
        y += 32

    return img.convert("RGB")


# ----------------------------- MAIN GENERATOR -----------------------------
def generate_comic(story, panels=4):
    pipe = load_pipe()

    sentences = [s.strip() for s in story.split(".") if s.strip()]
    while len(sentences) < panels:
        sentences.append(sentences[-1])
    sentences = sentences[:panels]

    # STRONG CHARACTER IDENTITY (Modify this or auto-detect)
    character_identity = (
        "A young female scientist, short black hair, glasses, slim build, "
        "wearing red sweater and jeans, SAME FACE every panel, SAME clothing, "
        "consistent appearance, realistic proportions"
    )

    results = []
    seed_base = 9000

    for i in range(panels):

        prompt = (
            f"{sentences[i]}. "
            f"{character_identity}. "
            "realistic lighting, detailed environment, cinematic atmosphere, high quality"
        )

        print(f"\nPANEL {i+1} PROMPT:\n{prompt}\n")

        gen = torch.Generator(device=DEVICE).manual_seed(seed_base + i)

        img = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=gen
        ).images[0]

        img = draw_caption(img, sentences[i])
        path = os.path.join(OUT_DIR, f"panel_{i+1}.png")
        img.save(path)
        print("Saved:", path)

        results.append(img)

    W, H = results[0].size
    final = Image.new("RGB", (W * panels + 10*(panels-1), H), (240,240,240))
    x = 0
    for p in results:
        final.paste(p, (x,0))
        x += W + 10

    final_path = os.path.join(OUT_DIR, "comic_final.png")
    final.save(final_path)
    print("\nFINAL COMIC:", final_path)

    return final_path


# ----------------------------- DEMO -----------------------------
if __name__ == "__main__":
    DEMO = """
    Kai did homework.
    But a dinosaur showed up.
    It ate his homework.
    Kai got scolded by his teacher.
    """

    generate_comic(DEMO)
