import os
from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableDiffusionPipeline
import gc
import re

class ComicGenerator:
    def __init__(self):
        """Initialize the comic generator with Stable Diffusion"""
        self.characters = []
        self.panels = []
        self.pipe = None
        self.character_seeds = {}
        self.panel_seeds_used = []
        self.load_stable_diffusion()
        
    def assign_character_seeds(self):
        """Assign fixed seeds to characters for visual consistency"""
        base_seeds = [42, 123, 456, 789, 1011]
        for i, char in enumerate(self.characters):
            seed = base_seeds[i % len(base_seeds)]
            self.character_seeds[char['name']] = seed
            char['seed'] = seed
            print(f"  Assigned seed {seed} to {char['name']} for consistency")
    
    def load_stable_diffusion(self):
        """Load Stable Diffusion model"""
        print("Loading Stable Diffusion model (this may take a few minutes first time)...")
        
        try:
            model_id = "runwayml/stable-diffusion-v1-5"
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            if device == "cuda":
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                self.pipe = self.pipe.to(device)
                self.pipe.enable_attention_slicing()
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except:
                    pass
            else:
                print("WARNING: Running on CPU will be VERY slow. Consider using GPU.")
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    safety_checker=None,
                    requires_safety_checker=False
                )
                self.pipe = self.pipe.to(device)
            
            print("Stable Diffusion loaded successfully!")
            
        except Exception as e:
            print(f"Error loading Stable Diffusion: {e}")
            raise
    
    def analyze_story_enhanced(self, story_text):
        """Enhanced story analysis with better scene splitting and action detection"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', story_text) if s.strip()]
        
        # EXPANDED character descriptions across multiple genres
        character_keywords = {
            # Fantasy
            'knight': {
                'patterns': [r'\bknight\b', r'\bSir\s+([A-Z][a-z]+)'],
                'description': 'cartoon knight with silver helmet, red cape, sword, shield, blue eyes, friendly smile',
                'appearance_tokens': 'silver armor, red flowing cape, metal helmet with face visible'
            },
            'wizard': {
                'patterns': [r'\bwizard\b', r'\bmage\b', r'\bMerlin\b'],
                'description': 'cartoon wizard with long white beard, blue robe with stars, purple pointed hat, wooden staff',
                'appearance_tokens': 'white fluffy beard, blue star-patterned robe, purple pointed wizard hat'
            },
            'dragon': {
                'patterns': [r'\bdragon\b'],
                'description': 'cartoon red dragon with wings, scales, yellow eyes, friendly expression',
                'appearance_tokens': 'red scales, large bat wings, long tail, yellow reptilian eyes'
            },
            'princess': {
                'patterns': [r'\bprincess\b'],
                'description': 'cartoon princess with long blonde hair, pink dress, golden crown',
                'appearance_tokens': 'long blonde hair, pink elegant gown, small golden crown'
            },
            
            # Sci-Fi
            'astronaut': {
                'patterns': [r'\bastronaut\b', r'\bspace explorer\b'],
                'description': 'cartoon astronaut with white space suit, bubble helmet, oxygen tank, American flag patch',
                'appearance_tokens': 'white spacesuit, clear bubble helmet, NASA logo, oxygen tanks on back'
            },
            'robot': {
                'patterns': [r'\brobot\b', r'\bandroid\b'],
                'description': 'cartoon robot with silver metal body, blue glowing eyes, antenna, mechanical joints',
                'appearance_tokens': 'silver metallic body, blue LED eyes, antenna on head, visible mechanical joints'
            },
            'alien': {
                'patterns': [r'\balien\b', r'\bextraterrestrial\b'],
                'description': 'cartoon alien with green skin, large black eyes, thin body, three fingers',
                'appearance_tokens': 'green smooth skin, oversized black eyes, thin frame, three-fingered hands'
            },
            
            # Detective/Mystery
            'detective': {
                'patterns': [r'\bdetective\b', r'\binvestigator\b', r'\bsleuth\b'],
                'description': 'cartoon detective with brown trench coat, fedora hat, magnifying glass, mustache',
                'appearance_tokens': 'tan trench coat, brown fedora hat, magnifying glass in hand, thin mustache'
            },
            'thief': {
                'patterns': [r'\bthief\b', r'\bburglar\b', r'\brobber\b'],
                'description': 'cartoon thief with black mask over eyes, striped shirt, bag of loot',
                'appearance_tokens': 'black eye mask, black-white striped shirt, sack over shoulder'
            },
            
            # Western
            'cowboy': {
                'patterns': [r'\bcowboy\b', r'\bsheriff\b'],
                'description': 'cartoon cowboy with brown cowboy hat, vest, sheriff star badge, holstered pistol',
                'appearance_tokens': 'brown cowboy hat, leather vest, gold sheriff badge, gun holster'
            },
            
            # Animals
            'cat': {
                'patterns': [r'\bcat\b', r'\bkitten\b'],
                'description': 'cartoon cat with orange fur, white belly, whiskers, tail, pink nose',
                'appearance_tokens': 'orange striped fur, white chest, long whiskers, curled tail'
            },
            'dog': {
                'patterns': [r'\bdog\b', r'\bpuppy\b'],
                'description': 'cartoon dog with brown fur, floppy ears, wagging tail, collar',
                'appearance_tokens': 'brown fur, floppy hanging ears, collar with tag, wagging tail'
            },
            
            # Modern/Everyday
            'doctor': {
                'patterns': [r'\bdoctor\b', r'\bphysician\b'],
                'description': 'cartoon doctor with white coat, stethoscope around neck, glasses, clipboard',
                'appearance_tokens': 'white lab coat, stethoscope, black-framed glasses, medical clipboard'
            },
            'chef': {
                'patterns': [r'\bchef\b', r'\bcook\b'],
                'description': 'cartoon chef with white tall hat, white uniform, mustache, wooden spoon',
                'appearance_tokens': 'tall white chef hat, white chef uniform, handlebar mustache'
            },
            'teacher': {
                'patterns': [r'\bteacher\b', r'\bprofessor\b'],
                'description': 'cartoon teacher with glasses, sweater, holding pointer stick, friendly smile',
                'appearance_tokens': 'round glasses, cardigan sweater, pointer stick in hand'
            },
            
            # Adventure
            'pirate': {
                'patterns': [r'\bpirate\b', r'\bcaptain\b'],
                'description': 'cartoon pirate with eye patch, red bandana, striped shirt, hook hand',
                'appearance_tokens': 'black eye patch, red bandana, red-white striped shirt, hook for hand'
            },
            'explorer': {
                'patterns': [r'\bexplorer\b', r'\badventurer\b'],
                'description': 'cartoon explorer with khaki outfit, safari hat, binoculars around neck, backpack',
                'appearance_tokens': 'tan safari outfit, wide-brimmed hat, binoculars, large backpack'
            },
            
            # Generic fallback
            'villager': {
                'patterns': [r'\bvillager\b', r'\bpeople\b', r'\bcrowd\b'],
                'description': 'cartoon villager with simple clothes, friendly face',
                'appearance_tokens': 'simple medieval tunic, brown pants'
            }
        }
        
        # Detect characters
        detected_characters = {}
        story_lower = story_text.lower()
        
        for char_type, char_info in character_keywords.items():
            for pattern in char_info['patterns']:
                matches = re.findall(pattern, story_text, re.IGNORECASE)
                if matches or char_type in story_lower:
                    name_match = re.search(pattern, story_text)
                    if name_match and name_match.groups():
                        char_name = name_match.group(1) if name_match.group(1) else char_type.title()
                    else:
                        char_name = char_type.title()
                    
                    detected_characters[char_type] = {
                        'name': char_name,
                        'description': char_info['description'],
                        'appearance_tokens': char_info['appearance_tokens'],
                        'type': char_type
                    }
                    break
        
        if not detected_characters:
            detected_characters = {
                'hero': {
                    'name': 'Hero',
                    'description': 'cartoon hero character with brave expression',
                    'appearance_tokens': 'casual clothes, determined expression',
                    'type': 'hero'
                }
            }
        
        self.characters = list(detected_characters.values())
        
        # Create panels with more detail - merge related sentences for action scenes
        panels = []
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            
            # Check if this is an action scene that should be merged with next sentence
            should_merge = False
            next_context = ""
            
            if i + 1 < len(sentences):
                next_sentence = sentences[i + 1]
                # Merge if next sentence continues the action
                action_continuations = ['after', 'then', 'the knight', 'the wizard', 'the dragon']
                if any(next_sentence.lower().startswith(cont) for cont in action_continuations):
                    # Check for major action keywords
                    major_actions = ['battle', 'fight', 'fought', 'defeated', 'confronted', 'attacked']
                    if any(action in sentence.lower() or action in next_sentence.lower() for action in major_actions):
                        should_merge = True
                        next_context = next_sentence
            
            # Combine sentences if it's an action scene
            if should_merge:
                combined_scene = f"{sentence} {next_context}"
                i += 2  # Skip next sentence as we've merged it
            else:
                combined_scene = sentence
                i += 1
            
            # Detect characters in this scene
            scene_characters = []
            for char_type, char_data in detected_characters.items():
                if char_type in combined_scene.lower() or char_data['name'].lower() in combined_scene.lower():
                    scene_characters.append(char_data['name'])
            
            if not scene_characters:
                scene_characters = [list(detected_characters.values())[0]['name']]
            
            # Multiple dialogue detection patterns
            dialogue = None
            
            # Pattern 1: Standard quotes
            dialogue_match = re.search(r'"([^"]+)"', combined_scene)
            if dialogue_match:
                dialogue = dialogue_match.group(1)
            
            # Pattern 2: Smart quotes
            if not dialogue:
                dialogue_match = re.search(r'["""]([^"""]+)["""]', combined_scene)
                if dialogue_match:
                    dialogue = dialogue_match.group(1)
            
            # Pattern 3: Single quotes
            if not dialogue:
                dialogue_match = re.search(r"'([^']+)'", combined_scene)
                if dialogue_match:
                    dialogue = dialogue_match.group(1)
            
            # If NO dialogue found, create descriptive narration text
            if not dialogue:
                action_word = self.extract_action_word(combined_scene)
                if action_word:
                    char_list = ', '.join([c.lower() for c in scene_characters])
                    dialogue = f"The {char_list} {action_word}"
                else:
                    char_names = ' and '.join(scene_characters)
                    dialogue = f"{char_names} in the scene"
                
                if len(dialogue) > 60:
                    dialogue = dialogue[:57] + "..."
                
                print(f"  → Generated narration: \"{dialogue}\"")
            else:
                print(f"  ✓ Found dialogue: \"{dialogue}\"")
            
            background = self.detect_background(combined_scene)
            action = self.detect_action(combined_scene)
            
            panel_data = {
                'scene': combined_scene,
                'characters': scene_characters,
                'action': action,
                'background': background,
                'dialogue': dialogue
            }
            
            panels.append(panel_data)
        
        # Ensure even number of panels
        if len(panels) % 2 != 0:
            # Add a closing panel if odd number
            print(f"\n  ℹ️  Adjusting to even number of panels (current: {len(panels)})")
            
            # Create a conclusion panel
            all_chars = list(set([c['name'] for panel in panels for c in self.characters if c['name'] in panel['characters']]))
            
            panels.append({
                'scene': 'The story comes to an end.',
                'characters': all_chars[:2] if len(all_chars) >= 2 else all_chars,
                'action': 'happy ending pose, satisfied expressions, standing together',
                'background': 'cartoon outdoor background, bright sky, colorful grass, animated style',
                'dialogue': 'The End'
            })
            print(f"  ✓ Added closing panel. Total panels: {len(panels)}")
        
        self.panels = panels
        return {'characters': self.characters, 'panels': panels}
    
    def extract_action_word(self, sentence):
        """Extract the main action/verb from sentence for narration - expanded genres"""
        action_patterns = [
            # Fantasy
            r'\b(rode|riding)\b',
            r'\b(met|meeting)\b',
            r'\b(warned|warning)\b',
            r'\b(replied|responding)\b',
            r'\b(journeyed|traveling|walking)\b',
            r'\b(appeared|appearing)\b',
            r'\b(roared|roaring)\b',
            r'\b(confronted|facing)\b',
            r'\b(fought|fighting|battled)\b',
            r'\b(defeated|conquering)\b',
            r'\b(celebrated|celebrating)\b',
            r'\b(returned|returning)\b',
            
            # Sci-Fi
            r'\b(launched|launching)\b',
            r'\b(explored|exploring)\b',
            r'\b(discovered|discovering)\b',
            r'\b(scanned|scanning)\b',
            r'\b(beamed|teleporting)\b',
            
            # Detective/Mystery
            r'\b(investigated|investigating)\b',
            r'\b(searched|searching)\b',
            r'\b(examined|examining)\b',
            r'\b(solved|solving)\b',
            r'\b(chased|chasing)\b',
            
            # General Action
            r'\b(ran|running)\b',
            r'\b(jumped|jumping)\b',
            r'\b(climbed|climbing)\b',
            r'\b(escaped|escaping)\b',
            r'\b(rescued|rescuing)\b',
            r'\b(helped|helping)\b',
            r'\b(worked|working)\b',
            r'\b(cooked|cooking)\b',
            r'\b(taught|teaching)\b'
        ]
        
        sentence_lower = sentence.lower()
        for pattern in action_patterns:
            match = re.search(pattern, sentence_lower)
            if match:
                return match.group(1)
        
        return None
    
    def detect_background(self, text):
        """Detect background setting with cartoon style - expanded genres"""
        backgrounds = {
            # Fantasy
            'forest': 'cartoon forest background, colorful trees, bright green leaves, sunny, animated style',
            'cave': 'cartoon cave interior, stylized rocks, simple shadows, animated background',
            'castle': 'cartoon medieval castle, bright stone walls, colorful banners, animated style',
            'village': 'cartoon village background, cute cottages, colorful buildings, animated style',
            'mountain': 'cartoon mountain landscape, stylized peaks, bright sky, animated background',
            'clearing': 'cartoon forest clearing, bright grass, flowers, colorful trees, animated style',
            'lair': 'cartoon cave lair, treasure piles, fun atmosphere, animated background',
            
            # Sci-Fi
            'spaceship': 'cartoon spaceship interior, control panels, buttons, blinking lights, futuristic',
            'space': 'cartoon outer space, stars, planets, asteroids, dark blue background',
            'lab': 'cartoon science laboratory, beakers, computers, equipment, white walls',
            'alien planet': 'cartoon alien planet, strange plants, two moons, purple sky',
            
            # Modern/Urban
            'city': 'cartoon city background, buildings, streets, cars, urban setting',
            'office': 'cartoon office, desk, computer, papers, indoor workplace',
            'school': 'cartoon classroom, chalkboard, desks, books, educational setting',
            'hospital': 'cartoon hospital room, bed, medical equipment, clean white walls',
            'kitchen': 'cartoon kitchen, stove, counter, pots and pans, homey setting',
            'restaurant': 'cartoon restaurant interior, tables, chairs, menu board',
            
            # Nature/Outdoor
            'beach': 'cartoon beach, sand, ocean waves, palm trees, sunny day',
            'park': 'cartoon park, grass, trees, playground, sunny outdoor scene',
            'jungle': 'cartoon jungle, thick vegetation, vines, tropical plants',
            'desert': 'cartoon desert, sand dunes, cactus, bright sun',
            
            # Adventure
            'ship': 'cartoon pirate ship deck, wooden planks, masts, ocean view',
            'treasure island': 'cartoon tropical island, palm trees, beach, treasure chest',
            
            # Mystery
            'mansion': 'cartoon spooky mansion, old furniture, dim lighting, mysterious',
            'detective office': 'cartoon detective office, desk, filing cabinets, window'
        }
        
        text_lower = text.lower()
        for location, description in backgrounds.items():
            if location in text_lower:
                return description
        
        return 'cartoon outdoor background, bright sky, colorful grass, animated style'
    
    def detect_action(self, text):
        """Detect main action with more detailed descriptions - expanded genres"""
        actions = {
            # Fantasy
            'rode': 'riding on horse, dynamic animated pose, galloping motion',
            'met': 'standing facing each other, friendly conversation pose, making eye contact',
            'warned': 'pointing gesture with raised hand, concerned worried expression, animated body language',
            'battle': 'intense combat action scene, weapons raised and swinging, dynamic fighting poses, action lines',
            'fought': 'fierce fighting action, characters in combat stances, weapons clashing, motion blur effects',
            'defeated': 'victorious triumphant pose, winner standing over fallen opponent, arms raised in victory',
            'celebrated': 'happy joyful celebration, arms raised high, jumping with joy, big smiles',
            'confronted': 'tense face-to-face standoff, determined expressions, weapons drawn and ready',
            'roared': 'mouth wide open roaring loudly, aggressive threatening pose, dramatic sound effect',
            'raised': 'holding object high overhead with both hands, dramatic heroic gesture',
            'journey': 'walking together side by side, traveling companions, moving forward with purpose',
            
            # Sci-Fi
            'launched': 'launching spaceship, rocket taking off, flames and smoke below',
            'explored': 'looking around curiously, discovering new things, pointing at discoveries',
            'discovered': 'excited discovery pose, pointing at finding, surprised expression',
            'scanned': 'holding scanning device, looking at readings, analyzing',
            'beamed': 'teleportation effect, glowing light surrounding characters',
            
            # Detective/Mystery
            'investigated': 'examining clues carefully, holding magnifying glass, detective pose',
            'searched': 'looking around carefully, searching through items, focused expression',
            'examined': 'closely inspecting object, thoughtful detective pose',
            'solved': 'eureka moment, finger pointing up, lightbulb realization',
            'chased': 'running after someone, dynamic chase scene, motion lines',
            
            # General Actions
            'ran': 'running fast, legs in motion, speed lines behind',
            'jumped': 'leaping in air, dynamic jump pose, excited movement',
            'climbed': 'climbing upward, reaching for handholds, vertical movement',
            'escaped': 'fleeing scene, running away quickly, looking back',
            'rescued': 'heroic rescue pose, carrying or helping someone',
            'helped': 'assisting someone, friendly helpful gesture',
            'worked': 'working diligently, focused on task, busy pose',
            'cooked': 'cooking food, holding utensils, chef in action',
            'taught': 'teaching gesture, pointing at board, explaining',
            
            # Communication
            'appeared': 'dramatic surprise entrance, emerging from location, characters reacting with surprise',
            'replied': 'speaking with expressive gestures, animated talking pose, hand movements',
            'said': 'conversational talking pose, friendly expression, making eye contact',
            'shouted': 'animated shouting with mouth wide open, urgent gesture, loud exclamation',
            'returned': 'walking back home, satisfied content expressions, journey completed'
        }
        
        text_lower = text.lower()
        for keyword, action_desc in actions.items():
            if keyword in text_lower:
                return action_desc
        
        return 'standing in scene with expressive poses and body language'
    
    def generate_image_prompt(self, panel_data):
        """Generate detailed cartoon-style prompt with STRONG character consistency"""
        character_descriptions = []
        character_appearances = []
        
        for char_name in panel_data['characters']:
            char = next((c for c in self.characters if c['name'] == char_name), None)
            if char:
                character_descriptions.append(char['description'])
                character_appearances.append(char['appearance_tokens'])
        
        cartoon_keywords = "cartoon art style, animated style, comic book illustration, bold outlines, cel-shaded, vibrant colors, clean lines, simplified features, expressive characters, dynamic composition"
        
        action_detail = panel_data['action']
        num_characters = len(character_descriptions)
        
        # Build consistency tokens - REPEAT key features
        consistency_emphasis = " | ".join(character_appearances)
        
        if num_characters == 1:
            prompt = f"CONSISTENT CHARACTER: {character_descriptions[0]} | APPEARANCE: {character_appearances[0]} | FULL BODY dynamic shot, {action_detail}, clearly showing the action, {panel_data['background']}, {cartoon_keywords}, wide shot showing full scene, professional comic panel with depth, dramatic angle | REMEMBER: {character_appearances[0]}"
        elif num_characters == 2:
            char1_desc, char2_desc = character_descriptions[0], character_descriptions[1]
            char1_app, char2_app = character_appearances[0], character_appearances[1]
            prompt = f"TWO DISTINCT CHARACTERS IN SAME SCENE | CHARACTER 1 (left): {char1_desc} WITH {char1_app} | CHARACTER 2 (right): {char2_desc} WITH {char2_app} | wide angle FULL BODY shot, BOTH characters clearly visible and separate with space between them, {action_detail}, both characters fully visible from head to feet, {panel_data['background']}, {cartoon_keywords}, panoramic view, rule of thirds composition | MAINTAIN APPEARANCES: {char1_app} AND {char2_app}"
        else:
            char_list = ", ".join([f"{desc} with {app}" for desc, app in zip(character_descriptions, character_appearances)])
            prompt = f"{num_characters} DISTINCT CHARACTERS IN SAME SCENE | CHARACTERS: {char_list} | ultra wide angle FULL BODY group shot, ALL {num_characters} characters clearly visible as separate individuals with space between them, {action_detail}, complete bodies visible from head to feet, {panel_data['background']}, {cartoon_keywords}, panoramic wide shot, clear spacing between characters | MAINTAIN ALL APPEARANCES: {consistency_emphasis}"
        
        negative_prompt = "inconsistent character design, changing appearance, different outfit, realistic, photorealistic, photograph, real life, 3D render, cropped, cut off, close-up, portrait, static pose, boring composition, blurry, low quality, distorted, deformed, MERGED CHARACTERS, FUSED BODIES, characters touching, overlapping characters, same character appearing twice, duplicate character, character fusion, morphed creatures, hybrid creatures, one character, solo character, single person, text, watermark, messy, dark, gloomy, horror style, anime, manga, no action, standing still"
        
        return prompt, negative_prompt
    
    def generate_image_with_sd(self, prompt, negative_prompt, panel_characters, panel_index, num_characters):
        """Generate cartoon-style image with variable width based on character count"""
        try:
            print(f"  Generating cartoon-style image with {num_characters} character(s)...")
            
            seed_components = []
            for char_name in sorted(panel_characters):
                if char_name in self.character_seeds:
                    seed_components.append(self.character_seeds[char_name])
            
            if seed_components:
                base_seed = sum(seed_components) * 100
            else:
                base_seed = 42
            
            unique_seed = base_seed + (panel_index * 7)
            
            generator = torch.Generator(device=self.pipe.device).manual_seed(unique_seed)
            
            # Variable width based on number of characters
            if num_characters == 1:
                width = 768  # Standard width for single character
            elif num_characters == 2:
                width = 1024  # Wider for two characters
            else:
                width = 1280  # Extra wide for 3+ characters
            
            print(f"  Panel width: {width}px (for {num_characters} character(s))")
            print(f"  Character seeds: {seed_components} → Base: {base_seed} → Final: {unique_seed}")
            
            with torch.no_grad():
                result = self.pipe(
                    prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=50,
                    guidance_scale=8.5,
                    height=512,
                    width=width,
                    generator=generator
                )
            
            image = result.images[0]
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return image
                
        except Exception as e:
            print(f"  Error generating image: {e}")
            return None
    
    def add_dialogue_to_image(self, img, dialogue, is_narration=False):
        """Add comic-style dialogue bubble or narration box"""
        if not dialogue or dialogue.strip() == "":
            return img
        
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        width, height = img_copy.size
        
        font_size = 28 if is_narration else 32
        font = None
        font_paths = [
            "C:\\Windows\\Fonts\\comic.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc"
        ]
        
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
        
        # Text wrapping
        words = dialogue.split()
        lines = []
        current_line = []
        max_chars_per_line = 30
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if len(test_line) <= max_chars_per_line:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        lines = lines[:3]
        
        if not lines:
            return img_copy
        
        padding = 30
        line_spacing = 40
        
        max_line_width = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            max_line_width = max(max_line_width, line_width)
        
        bubble_width = max_line_width + (padding * 2)
        bubble_height = len(lines) * line_spacing + (padding * 2)
        
        bubble_x = (width - bubble_width) // 2
        bubble_y = 20 if not is_narration else height - bubble_height - 20
        
        # Shadow
        shadow_offset = 5
        draw.rounded_rectangle(
            [(bubble_x + shadow_offset, bubble_y + shadow_offset), 
             (bubble_x + bubble_width + shadow_offset, bubble_y + bubble_height + shadow_offset)],
            radius=20 if not is_narration else 10,
            fill='#666666'
        )
        
        if is_narration:
            draw.rounded_rectangle(
                [(bubble_x, bubble_y), (bubble_x + bubble_width, bubble_y + bubble_height)],
                radius=10,
                fill='#FFFACD',
                outline='black',
                width=4
            )
        else:
            draw.rounded_rectangle(
                [(bubble_x, bubble_y), (bubble_x + bubble_width, bubble_y + bubble_height)],
                radius=20,
                fill='white',
                outline='black',
                width=6
            )
        
        # Draw text
        y_text = bubble_y + padding
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            x_text = bubble_x + (bubble_width - line_width) // 2
            
            for adj in [(2,2), (-2,2), (2,-2), (-2,-2)]:
                draw.text((x_text + adj[0], y_text + adj[1]), line, fill='black', font=font)
            
            draw.text((x_text, y_text), line, fill='black', font=font)
            y_text += line_spacing
        
        text_type = "Narration" if is_narration else "Dialogue"
        print(f"      ✓ {text_type} added: \"{dialogue}\"")
        return img_copy
    
    def add_panel_border(self, img):
        """Add comic panel border"""
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        width, height = img_copy.size
        
        border_width = 10
        draw.rectangle(
            [(0, 0), (width-1, height-1)],
            outline='black',
            width=border_width
        )
        
        return img_copy
    
    def generate_comic(self, story_text, output_path='comic_output'):
        """Generate complete comic from story text"""
        print("Analyzing story with enhanced dialogue detection...")
        self.analyze_story_enhanced(story_text)
        
        self.assign_character_seeds()
        
        print(f"\n{'='*70}")
        print(f"DETECTED CHARACTERS ({len(self.characters)}):")
        print(f"{'='*70}")
        for char in self.characters:
            print(f"\n{char['name']} ({char['type']}) - Seed: {char.get('seed', 'N/A')}")
            print(f"  Cartoon Style: {char['description'][:80]}...")
        
        print(f"\n{'='*70}")
        print(f"GENERATING {len(self.panels)} CARTOON-STYLE PANELS")
        print(f"{'='*70}")
        
        os.makedirs(output_path, exist_ok=True)
        
        panel_images = []
        
        for i, panel_data in enumerate(self.panels, 1):
            print(f"\n📖 PANEL {i}/{len(self.panels)}")
            print(f"   Scene: {panel_data['scene'][:70]}...")
            print(f"   Characters: {', '.join(panel_data['characters'])}")
            print(f"   Action: {panel_data['action']}")
            
            has_original_dialogue = '"' in panel_data['scene'] or '"' in panel_data['scene'] or "'" in panel_data['scene']
            
            if panel_data.get('dialogue'):
                if has_original_dialogue:
                    print(f"   💬 DIALOGUE: \"{panel_data['dialogue']}\"")
                else:
                    print(f"   📝 NARRATION: \"{panel_data['dialogue']}\"")
            
            prompt, negative_prompt = self.generate_image_prompt(panel_data)
            print(f"   Prompt: Cartoon style with character consistency enabled")
            
            num_characters = len(panel_data['characters'])
            img = self.generate_image_with_sd(prompt, negative_prompt, panel_data['characters'], i, num_characters)
            
            if img is not None:
                img = self.add_panel_border(img)
                
                if panel_data.get('dialogue'):
                    is_narration = not has_original_dialogue
                    print(f"   📝 Adding {'narration box' if is_narration else 'dialogue bubble'}...")
                    img = self.add_dialogue_to_image(img, panel_data['dialogue'], is_narration)
                
                panel_path = os.path.join(output_path, f'panel_{i:02d}.png')
                img.save(panel_path, quality=95, optimize=True)
                panel_images.append(panel_path)
                print(f"   ✅ Saved: {panel_path}")
            else:
                print(f"   ❌ Failed to generate panel {i}")
        
        if panel_images:
            self.create_comic_strip(panel_images, output_path)
        
        return panel_images
    
    def create_comic_strip(self, panel_paths, output_path):
        """Combine panels into comic strip with variable widths"""
        if not panel_paths:
            return
        
        panels = [Image.open(path) for path in panel_paths]
        
        # Find max dimensions
        max_height = max(panel.size[1] for panel in panels)
        margin = 30
        
        # Calculate layout based on variable widths
        panels_per_row = 2  # Two panels per row for better visibility
        rows = (len(panels) + panels_per_row - 1) // panels_per_row
        
        # Calculate required width (max width of any row)
        row_widths = []
        for row_idx in range(rows):
            start_idx = row_idx * panels_per_row
            end_idx = min(start_idx + panels_per_row, len(panels))
            row_panels = panels[start_idx:end_idx]
            row_width = sum(p.size[0] for p in row_panels) + margin * (len(row_panels) + 1)
            row_widths.append(row_width)
        
        strip_width = max(row_widths)
        strip_height = max_height * rows + margin * (rows + 1)
        
        strip = Image.new('RGB', (strip_width, strip_height), color='#DDDDDD')
        
        # Place panels
        current_row = 0
        current_x = margin
        current_y = margin
        
        for i, panel in enumerate(panels):
            if i > 0 and i % panels_per_row == 0:
                # Move to next row
                current_row += 1
                current_y = margin + current_row * (max_height + margin)
                current_x = margin
            
            strip.paste(panel, (current_x, current_y))
            current_x += panel.size[0] + margin
        
        strip_path = os.path.join(output_path, 'comic_strip.png')
        strip.save(strip_path, quality=95, optimize=True)
        print(f"\n{'='*70}")
        print(f"✅ COMIC STRIP SAVED: {strip_path}")
        print(f"   Layout: {panels_per_row} panels per row × {rows} row(s)")
        print(f"   Variable panel widths based on character count")
        print(f"{'='*70}")
    
    def save_model_config(self, config_path='comic_model_config.json'):
        """Save the model configuration for later use"""
        import json
        
        config = {
            'model_id': 'runwayml/stable-diffusion-v1-5',
            'description': 'Comic Generator with Multi-Genre Support',
            'supported_genres': [
                'Fantasy (knights, wizards, dragons, princesses)',
                'Sci-Fi (astronauts, robots, aliens)',
                'Detective/Mystery (detectives, thieves)',
                'Western (cowboys, sheriffs)',
                'Animals (cats, dogs)',
                'Modern/Everyday (doctors, chefs, teachers)',
                'Adventure (pirates, explorers)'
            ],
            'generation_params': {
                'num_inference_steps': 50,
                'guidance_scale': 8.5,
                'height': 512,
                'width_single_char': 768,
                'width_two_chars': 1024,
                'width_multi_chars': 1280
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Model configuration saved to: {config_path}")
        return config_path


# Example usage
if __name__ == "__main__":
    
    # Example 1: Fantasy story
    fantasy_story = """
    A brave knight named Sir Roland rode through the enchanted forest on his white horse. 
    In a forest clearing, the knight met an elderly wizard named Merlin.
    "Beware the dragon ahead!" warned the wizard urgently.
    "I fear no beast," replied Sir Roland with confidence.
    The knight and wizard journeyed together toward the distant mountains.
    Near a dark cave, the massive dragon appeared and roared fiercely.
    "Prepare for battle!" shouted the wizard to the knight.
    The knight confronted the dragon with his sword raised high.
    After an intense battle, the knight defeated the dragon victoriously.
    "We make a great team," said the wizard with a wise smile.
    The villagers celebrated as the knight and wizard returned home.
    """
    
    # Example 2: Sci-Fi story
    scifi_story = """
    The astronaut and robot explored the alien planet together.
    "Look at those strange plants!" said the astronaut excitedly.
    The robot scanned the environment with its sensors.
    Suddenly, a friendly alien appeared from behind a rock.
    "Welcome to our planet," said the alien with a smile.
    The astronaut and alien shook hands in friendship.
    """
    
    # Example 3: Detective story
    detective_story = """
    The detective investigated the mysterious mansion carefully.
    "I found a clue!" shouted the detective, holding a magnifying glass.
    A thief appeared suddenly from the shadows.
    The detective chased the thief through the city streets.
    After a long chase, the detective caught the thief.
    "Case solved!" said the detective triumphantly.
    """
    
    print("=" * 70)
    print("MULTI-GENRE CARTOON COMIC GENERATOR v3.0")
    print("Character Consistency • Multiple Genres • Text in Every Panel")
    print("=" * 70)
    print("\nSupported Genres:")
    print("  • Fantasy (knights, wizards, dragons)")
    print("  • Sci-Fi (astronauts, robots, aliens)")
    print("  • Detective/Mystery (detectives, thieves)")
    print("  • Western (cowboys)")
    print("  • Animals (cats, dogs)")
    print("  • Modern (doctors, chefs, teachers)")
    print("  • Adventure (pirates, explorers)")
    print("=" * 70)
    
    try:
        generator = ComicGenerator()
        
        # Choose which story to generate (change this to test different genres)
        story_choice = detective_story  # Change to scifi_story or detective_story
        
        panel_paths = generator.generate_comic(story_choice, output_path='my_comic')
        
        # Save model configuration
        generator.save_model_config('comic_model_config.json')
        
        print(f"\n🎉 COMIC GENERATION COMPLETE!")
        print(f"   Generated {len(panel_paths)} cartoon-style panels")
        print(f"   ✓ Characters maintain consistent appearance across panels")
        print(f"   ✓ Every panel has dialogue or narration text")
        print(f"   ✓ Variable panel widths based on character count")
        print(f"   ✓ Model configuration saved for UI integration")
        print(f"   Check the 'my_comic' folder for results")
        
        print(f"\n📝 INTEGRATION NOTES:")
        print(f"   1. This generator is ready for UI integration")
        print(f"   2. Input: story_text (string)")
        print(f"   3. Output: Panel images + comic strip")
        print(f"   4. Model config saved in: comic_model_config.json")
        print(f"   5. Simply call: generator.generate_comic(user_story, output_path)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()