"""
Evaluation Metrics for Comic Generator
Measures quality, consistency, and accuracy of generated comics
"""

import os
import json
from PIL import Image
import numpy as np
from datetime import datetime
import torch
from torchvision import transforms
from torchvision.models import inception_v3
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re


class ComicEvaluator:
    def __init__(self):
        """Initialize evaluation metrics"""
        self.metrics = {}
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def evaluate_comic(self, comic_folder, story_text, save_results=True):
        """
        Complete evaluation of a generated comic
        
        Args:
            comic_folder: Path to folder containing generated panels
            story_text: Original input story
            save_results: Whether to save results to JSON
        
        Returns:
            Dictionary of all metrics
        """
        print("=" * 70)
        print("COMIC EVALUATION METRICS")
        print("=" * 70)
        
        # Get all panel images
        panel_paths = self.get_panel_paths(comic_folder)
        
        if not panel_paths:
            print("❌ No panels found in folder!")
            return None
        
        print(f"\n📊 Evaluating {len(panel_paths)} panels...")
        
        # 1. Image Quality Metrics
        print("\n1️⃣ Evaluating Image Quality...")
        quality_metrics = self.evaluate_image_quality(panel_paths)
        
        # 2. Character Consistency Metrics
        print("\n2️⃣ Evaluating Character Consistency...")
        consistency_metrics = self.evaluate_character_consistency(panel_paths)
        
        # 3. Story-Image Alignment
        print("\n3️⃣ Evaluating Story-Image Alignment...")
        alignment_metrics = self.evaluate_story_alignment(story_text, comic_folder)
        
        # 4. Dialogue Detection
        print("\n4️⃣ Evaluating Dialogue Presence...")
        dialogue_metrics = self.evaluate_dialogue_presence(panel_paths)
        
        # 5. Panel Variety
        print("\n5️⃣ Evaluating Panel Variety...")
        variety_metrics = self.evaluate_panel_variety(panel_paths)
        
        # 6. Technical Metrics
        print("\n6️⃣ Evaluating Technical Quality...")
        technical_metrics = self.evaluate_technical_quality(panel_paths)
        
        # Compile all metrics
        all_metrics = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'comic_folder': comic_folder,
            'num_panels': len(panel_paths),
            'image_quality': quality_metrics,
            'character_consistency': consistency_metrics,
            'story_alignment': alignment_metrics,
            'dialogue_presence': dialogue_metrics,
            'panel_variety': variety_metrics,
            'technical_quality': technical_metrics,
            'overall_score': self.calculate_overall_score(
                quality_metrics, consistency_metrics, alignment_metrics,
                dialogue_metrics, variety_metrics, technical_metrics
            )
        }
        
        # Print results
        self.print_results(all_metrics)
        
        # Save to JSON
        if save_results:
            results_path = os.path.join(comic_folder, 'evaluation_metrics.json')
            with open(results_path, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            print(f"\n💾 Results saved to: {results_path}")
        
        return all_metrics
    
    def get_panel_paths(self, folder):
        """Get sorted list of panel image paths"""
        panels = []
        for file in sorted(os.listdir(folder)):
            if file.startswith('panel_') and file.endswith('.png'):
                panels.append(os.path.join(folder, file))
        return panels
    
    def evaluate_image_quality(self, panel_paths):
        """Evaluate image quality metrics"""
        metrics = {
            'sharpness': [],
            'brightness': [],
            'contrast': [],
            'colorfulness': []
        }
        
        for path in panel_paths:
            img = cv2.imread(path)
            
            # Sharpness (Laplacian variance)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics['sharpness'].append(sharpness)
            
            # Brightness
            brightness = np.mean(img)
            metrics['brightness'].append(brightness)
            
            # Contrast
            contrast = np.std(img)
            metrics['contrast'].append(contrast)
            
            # Colorfulness (RMS of RGB standard deviations)
            r, g, b = cv2.split(img)
            colorfulness = np.sqrt(np.std(r)**2 + np.std(g)**2 + np.std(b)**2)
            metrics['colorfulness'].append(colorfulness)
        
        # Calculate averages
        return {
            'avg_sharpness': float(np.mean(metrics['sharpness'])),
            'avg_brightness': float(np.mean(metrics['brightness'])),
            'avg_contrast': float(np.mean(metrics['contrast'])),
            'avg_colorfulness': float(np.mean(metrics['colorfulness'])),
            'sharpness_consistency': float(np.std(metrics['sharpness'])),
            'quality_score': self.normalize_quality_score(metrics)
        }
    
    def normalize_quality_score(self, metrics):
        """Normalize quality metrics to 0-100 score"""
        # Ideal ranges (based on experimentation)
        ideal_sharpness = 1000  # Higher is sharper
        ideal_brightness = 127  # Middle brightness
        ideal_contrast = 60     # Good contrast
        ideal_colorfulness = 80 # Vibrant colors
        
        sharpness_score = min(np.mean(metrics['sharpness']) / ideal_sharpness * 100, 100)
        brightness_score = 100 - abs(np.mean(metrics['brightness']) - ideal_brightness) / 127 * 100
        contrast_score = min(np.mean(metrics['contrast']) / ideal_contrast * 100, 100)
        color_score = min(np.mean(metrics['colorfulness']) / ideal_colorfulness * 100, 100)
        
        return float((sharpness_score + brightness_score + contrast_score + color_score) / 4)
    
    def evaluate_character_consistency(self, panel_paths):
        """Evaluate visual consistency across panels"""
        if len(panel_paths) < 2:
            return {'consistency_score': 100.0, 'note': 'Single panel, perfect consistency'}
        
        # Extract image features
        features = []
        for path in panel_paths:
            img = Image.open(path).convert('RGB')
            # Simple color histogram as feature
            img_array = np.array(img)
            hist_r = np.histogram(img_array[:,:,0], bins=32, range=(0, 256))[0]
            hist_g = np.histogram(img_array[:,:,1], bins=32, range=(0, 256))[0]
            hist_b = np.histogram(img_array[:,:,2], bins=32, range=(0, 256))[0]
            features.append(np.concatenate([hist_r, hist_g, hist_b]))
        
        # Calculate pairwise similarities
        features = np.array(features)
        similarities = []
        for i in range(len(features) - 1):
            sim = cosine_similarity([features[i]], [features[i+1]])[0][0]
            similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 1.0
        consistency_score = float(avg_similarity * 100)
        
        return {
            'avg_similarity': float(avg_similarity),
            'consistency_score': consistency_score,
            'pairwise_similarities': [float(s) for s in similarities],
            'interpretation': self.interpret_consistency(consistency_score)
        }
    
    def interpret_consistency(self, score):
        """Interpret consistency score"""
        if score >= 80:
            return "Excellent - Characters very consistent"
        elif score >= 60:
            return "Good - Characters mostly consistent"
        elif score >= 40:
            return "Fair - Some consistency issues"
        else:
            return "Poor - Characters vary significantly"
    
    def evaluate_story_alignment(self, story_text, comic_folder):
        """Evaluate how well the comic matches the story"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', story_text) if s.strip()]
        panel_paths = self.get_panel_paths(comic_folder)
        
        # Check if number of panels matches story complexity
        expected_panels = len(sentences)
        actual_panels = len(panel_paths)
        panel_ratio = min(actual_panels / expected_panels, 1.0) if expected_panels > 0 else 1.0
        
        # Detect key story elements
        story_elements = self.extract_story_elements(story_text)
        
        # Check dialogue coverage
        dialogue_count = len(re.findall(r'["""]([^"""]+)["""]', story_text))
        dialogue_count += len(re.findall(r'"([^"]+)"', story_text))
        
        alignment_score = float(
            (panel_ratio * 40 +  # Panel count appropriateness
             min(actual_panels / max(dialogue_count, 1), 1.0) * 30 +  # Dialogue coverage
             30)  # Base score for attempting alignment
        )
        
        return {
            'story_sentences': expected_panels,
            'generated_panels': actual_panels,
            'panel_story_ratio': float(panel_ratio),
            'detected_dialogues': dialogue_count,
            'story_elements': story_elements,
            'alignment_score': alignment_score,
            'interpretation': self.interpret_alignment(alignment_score)
        }
    
    def extract_story_elements(self, story_text):
        """Extract key elements from story"""
        story_lower = story_text.lower()
        
        elements = {
            'characters': [],
            'actions': [],
            'locations': []
        }
        
        # Common character types
        char_types = ['knight', 'wizard', 'dragon', 'astronaut', 'robot', 'detective', 
                     'pirate', 'hero', 'villain', 'princess', 'alien']
        for char in char_types:
            if char in story_lower:
                elements['characters'].append(char)
        
        # Common actions
        actions = ['fought', 'met', 'discovered', 'chased', 'defeated', 'explored', 
                  'found', 'rescued', 'battled']
        for action in actions:
            if action in story_lower:
                elements['actions'].append(action)
        
        # Common locations
        locations = ['forest', 'castle', 'space', 'city', 'mountain', 'cave', 
                    'spaceship', 'island']
        for loc in locations:
            if loc in story_lower:
                elements['locations'].append(loc)
        
        return elements
    
    def interpret_alignment(self, score):
        """Interpret alignment score"""
        if score >= 80:
            return "Excellent - Comic closely follows story"
        elif score >= 60:
            return "Good - Comic represents story well"
        elif score >= 40:
            return "Fair - Some story elements missing"
        else:
            return "Poor - Comic deviates from story"
    
    def evaluate_dialogue_presence(self, panel_paths):
        """Evaluate presence and quality of dialogue boxes"""
        panels_with_text = 0
        text_areas = []
        
        for path in panel_paths:
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect white/light regions (dialogue boxes)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for rectangular regions (dialogue boxes)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5000:  # Minimum size for dialogue box
                    panels_with_text += 1
                    text_areas.append(area)
                    break
        
        dialogue_coverage = panels_with_text / len(panel_paths) * 100 if panel_paths else 0
        
        return {
            'panels_with_dialogue': panels_with_text,
            'total_panels': len(panel_paths),
            'dialogue_coverage': float(dialogue_coverage),
            'avg_dialogue_box_size': float(np.mean(text_areas)) if text_areas else 0,
            'dialogue_score': float(dialogue_coverage),
            'interpretation': self.interpret_dialogue(dialogue_coverage)
        }
    
    def interpret_dialogue(self, coverage):
        """Interpret dialogue coverage"""
        if coverage >= 90:
            return "Excellent - All/most panels have text"
        elif coverage >= 70:
            return "Good - Most panels have text"
        elif coverage >= 50:
            return "Fair - Some panels missing text"
        else:
            return "Poor - Many panels without text"
    
    def evaluate_panel_variety(self, panel_paths):
        """Evaluate visual variety across panels"""
        color_entropies = []
        edge_densities = []
        
        for path in panel_paths:
            img = cv2.imread(path)
            
            # Color entropy (variety of colors)
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = hist.flatten()
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            color_entropies.append(entropy)
            
            # Edge density (complexity)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
            edge_densities.append(edge_density)
        
        variety_score = float(
            (np.std(color_entropies) / np.mean(color_entropies) * 50 +
             np.std(edge_densities) / np.mean(edge_densities) * 50)
            if len(panel_paths) > 1 else 50
        )
        variety_score = min(max(variety_score, 0), 100)
        
        return {
            'avg_color_entropy': float(np.mean(color_entropies)),
            'avg_edge_density': float(np.mean(edge_densities)),
            'color_variance': float(np.std(color_entropies)),
            'edge_variance': float(np.std(edge_densities)),
            'variety_score': variety_score,
            'interpretation': self.interpret_variety(variety_score)
        }
    
    def interpret_variety(self, score):
        """Interpret variety score"""
        if score >= 40:
            return "Good - Panels show visual variety"
        elif score >= 20:
            return "Fair - Some visual variety"
        else:
            return "Poor - Panels too similar"
    
    def evaluate_technical_quality(self, panel_paths):
        """Evaluate technical aspects"""
        resolutions = []
        file_sizes = []
        aspect_ratios = []
        
        for path in panel_paths:
            img = Image.open(path)
            width, height = img.size
            
            resolutions.append(width * height)
            file_sizes.append(os.path.getsize(path) / 1024)  # KB
            aspect_ratios.append(width / height)
        
        return {
            'avg_resolution': float(np.mean(resolutions)),
            'avg_file_size_kb': float(np.mean(file_sizes)),
            'avg_aspect_ratio': float(np.mean(aspect_ratios)),
            'resolution_consistency': float(np.std(resolutions)),
            'aspect_ratio_consistency': float(np.std(aspect_ratios)),
            'all_resolutions': [f"{Image.open(p).size[0]}x{Image.open(p).size[1]}" for p in panel_paths]
        }
    
    def calculate_overall_score(self, quality, consistency, alignment, 
                                dialogue, variety, technical):
        """Calculate weighted overall score"""
        weights = {
            'quality': 0.20,
            'consistency': 0.25,
            'alignment': 0.25,
            'dialogue': 0.15,
            'variety': 0.10,
            'technical': 0.05
        }
        
        overall = (
            quality['quality_score'] * weights['quality'] +
            consistency['consistency_score'] * weights['consistency'] +
            alignment['alignment_score'] * weights['alignment'] +
            dialogue['dialogue_score'] * weights['dialogue'] +
            variety['variety_score'] * weights['variety'] +
            100 * weights['technical']  # Technical always counts as 100
        )
        
        return {
            'score': float(overall),
            'grade': self.score_to_grade(overall),
            'interpretation': self.interpret_overall(overall)
        }
    
    def score_to_grade(self, score):
        """Convert score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def interpret_overall(self, score):
        """Interpret overall score"""
        if score >= 85:
            return "Excellent - High quality comic generation"
        elif score >= 70:
            return "Good - Satisfactory comic generation"
        elif score >= 55:
            return "Fair - Acceptable with room for improvement"
        else:
            return "Poor - Significant improvements needed"
    
    def print_results(self, metrics):
        """Print formatted results"""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        
        print(f"\n📊 OVERALL SCORE: {metrics['overall_score']['score']:.1f}/100")
        print(f"   Grade: {metrics['overall_score']['grade']}")
        print(f"   {metrics['overall_score']['interpretation']}")
        
        print("\n" + "-" * 70)
        print("DETAILED METRICS:")
        print("-" * 70)
        
        print(f"\n1️⃣  IMAGE QUALITY: {metrics['image_quality']['quality_score']:.1f}/100")
        print(f"   Sharpness: {metrics['image_quality']['avg_sharpness']:.1f}")
        print(f"   Brightness: {metrics['image_quality']['avg_brightness']:.1f}")
        print(f"   Contrast: {metrics['image_quality']['avg_contrast']:.1f}")
        print(f"   Colorfulness: {metrics['image_quality']['avg_colorfulness']:.1f}")
        
        print(f"\n2️⃣  CHARACTER CONSISTENCY: {metrics['character_consistency']['consistency_score']:.1f}/100")
        print(f"   {metrics['character_consistency']['interpretation']}")
        
        print(f"\n3️⃣  STORY ALIGNMENT: {metrics['story_alignment']['alignment_score']:.1f}/100")
        print(f"   Panels: {metrics['story_alignment']['generated_panels']}/{metrics['story_alignment']['story_sentences']}")
        print(f"   {metrics['story_alignment']['interpretation']}")
        
        print(f"\n4️⃣  DIALOGUE PRESENCE: {metrics['dialogue_presence']['dialogue_score']:.1f}/100")
        print(f"   Coverage: {metrics['dialogue_presence']['panels_with_dialogue']}/{metrics['dialogue_presence']['total_panels']} panels")
        print(f"   {metrics['dialogue_presence']['interpretation']}")
        
        print(f"\n5️⃣  PANEL VARIETY: {metrics['panel_variety']['variety_score']:.1f}/100")
        print(f"   {metrics['panel_variety']['interpretation']}")
        
        print(f"\n6️⃣  TECHNICAL QUALITY:")
        print(f"   Avg Resolution: {int(metrics['technical_quality']['avg_resolution']):,} pixels")
        print(f"   Avg File Size: {metrics['technical_quality']['avg_file_size_kb']:.1f} KB")


# Example usage
if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("COMIC GENERATOR EVALUATION SYSTEM")
    print("=" * 70)
    
    if len(sys.argv) < 3:
        print("\nUsage: python comic_evaluation.py <comic_folder> <story_file>")
        print("\nExample:")
        print("  python comic_evaluation.py my_comic story.txt")
        sys.exit(1)
    
    comic_folder = sys.argv[1]
    story_file = sys.argv[2]
    
    if not os.path.exists(comic_folder):
        print(f"\n❌ Error: Folder '{comic_folder}' not found!")
        sys.exit(1)
    
    if not os.path.exists(story_file):
        print(f"\n❌ Error: Story file '{story_file}' not found!")
        sys.exit(1)
    
    # Read story
    with open(story_file, 'r') as f:
        story_text = f.read()
    
    # Run evaluation
    evaluator = ComicEvaluator()
    metrics = evaluator.evaluate_comic(comic_folder, story_text, save_results=True)
    
    print("\n✅ Evaluation complete!")
    print(f"📊 Results saved to: {os.path.join(comic_folder, 'evaluation_metrics.json')}")