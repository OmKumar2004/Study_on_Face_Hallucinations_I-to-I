# prompt_comparator_cpu.py
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

class CPUPromptComparator:
    def __init__(self):
        # Initialize with CPU-only components
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.nlp = spacy.load("en_core_web_sm")
        
    def calculate_scores(self, original_prompt, generated_prompts):
        """
        Compare original prompt against generated variants
        Returns list of dicts with scores and differences
        """
        # Encode all prompts
        original_embed = self.embedder.encode(original_prompt)
        generated_embeds = self.embedder.encode(generated_prompts)
        
        # Calculate similarities
        similarities = cosine_similarity([original_embed], generated_embeds)[0]
        
        # Extract features
        orig_features = self._extract_features(original_prompt)
        results = []
        
        for idx, (prompt, score) in enumerate(zip(generated_prompts, similarities)):
            gen_features = self._extract_features(prompt)
            differences = self._find_differences(orig_features, gen_features)
            
            results.append({
                "index": idx,
                "score": round(float(score), 3),
                "differences": differences
            })
            
        return sorted(results, key=lambda x: x['score'], reverse=True)

    def _extract_features(self, text):
        """Extract facial features using lightweight NLP"""
        doc = self.nlp(text)
        features = {
            'colors': [],
            'shapes': [],
            'adjectives': [],
            'facial_features': []
        }
        
        for ent in doc.ents:
            if ent.label_ in ['COLOR', 'SHAPE']:
                features['colors' if ent.label_ == 'COLOR' else 'shapes'].append(ent.text)
        
        for token in doc:
            if token.pos_ == 'ADJ':
                features['adjectives'].append(token.text)
            if token.dep_ in ['amod', 'nsubj'] and 'face' in token.head.text.lower():
                features['facial_features'].append(token.text)
                
        return features

    def _find_differences(self, orig, gen):
        """Identify differences between feature sets"""
        diffs = {}
        for key in orig:
            orig_vals = set(orig[key])
            gen_vals = set(gen.get(key, []))
            
            added = list(gen_vals - orig_vals)
            removed = list(orig_vals - gen_vals)
            
            if added or removed:
                diffs[key] = {
                    'added': added,
                    'removed': removed
                }
        return diffs

# Usage Example
if __name__ == "__main__":
    comparator = CPUPromptComparator()
    
    original = "1.	Face Shape: Appears somewhat angular/rectangular, with strong bone structure. Features seem balanced and symmetrical.\
2.	Jaw & Cheekbones: Jawline is well-defined and angular. Cheekbones are prominent, contributing to a strong facial contour, particularly visible under the eyes when smiling.\
3.	Forehead: Appears average to high size. Several horizontal forehead lines and vertical frown lines between the eyebrows are visible. Skin texture shows natural pores and some subtle variations.\
4.	Eyes & Brows: Eyes are moderately sized, set well within the sockets. Brown irises are visible. Crow's feet wrinkles radiate from the outer corners. Eyelids have visible creases. Eyebrows are dark, moderately thick, and have a slight natural arch.\
5.	Nose: Bridge appears relatively straight. Tip is rounded. Nostrils are visible and naturally shaped.\
6.	Lips & Mouth: Lips are relatively thin, especially the upper lip. Mouth is wide open in a broad smile, revealing both upper and lower teeth. Teeth appear slightly yellowed and show individual shapes and minor irregularities.\
7.	Skin & Wrinkles: Skin texture appears natural with visible pores, particularly on the nose and cheeks. Prominent wrinkles include crow's feet around the eyes, forehead lines, and nasolabial folds (smile lines). No significant blemishes or acne.\
8.	Tone & Lighting: Skin tone is fair, with areas of natural color variation and some redness, possibly from sun exposure. Strong directional lighting (likely sunlight) creates bright highlights on the forehead, nose bridge, cheeks, and chin, and distinct shadows under the cheekbones and jawline.\
9.	Hairline & Hair on Face: Hairline is visible, slightly receding at the temples. Dark hair, showing some grey, is swept back from the forehead, covering parts of the hairline.\
10.	Ears & Expression: Ears are partially visible on both sides. Small portions of the lobes and upper structure are seen.\
11.	Expression: A broad, genuine, and happy smile, conveying warmth and enjoyment. Eyes are slightly squinted due to the smile\
"
    generated = [
  "1.	Face Shape: Angular shape is suggested, but overall features are smoothed and lack sharp definition compared to the original. Symmetry is generally preserved.\
2.	Jaw & Cheekbones: Jawline is present but appears rounded and less defined. Cheekbones are smoothed, lacking the prominence seen in the original.\
3.	Forehead: Very smooth, almost entirely lacking texture or visible lines. Horizontal and vertical wrinkles are absent or heavily smoothed.\
4.	Eyes & Brows: Eye shape is roughly preserved, but iris detail is simplified and lacks sharpness. Crow's feet wrinkles are mostly absent. Eyebrows are dark but lack individual hair texture, appearing blocky or airbrushed.\
5.	Nose: Bridge and tip are present but lack fine detail. Nostrils are less defined.\
6.	Lips & Mouth: Lips are thin. Smile reveals teeth, but they appear extremely smooth and fused together, lacking individual tooth shapes and definition.\
7.	Skin & Wrinkles: Skin texture is highly smoothed and artificial-looking, lacking natural pores or blemishes. Wrinkles are predominantly removed.\
8.	Tone & Lighting: Skin tone is fair, appearing very smooth and uniform, lacking natural variations. Lighting effects are present but simplified, lacking the subtle transitions and sharp definition of the original.\
9.	Hairline & Hair on Face: Hairline is suggested. Hair on forehead is smoothed and lacks individual strands.\
10.	Ears & Expression: Ears are partially visible, heavily smoothed.\
11.	Expression: Smile is present, but the lack of fine detail and unnatural teeth diminish its genuineness, making it appear somewhat artificial.\
",
"1.	Face Shape: Angular shape is better represented than in the 32x32 version, closer to the original structure. Symmetry is maintained.\
2.	Jaw & Cheekbones: Jawline is better defined than 32x32, showing more angularity. Cheekbones are more prominent than 32x32 but still appear slightly smoothed compared to the original.\
3.	Forehead: Smooth, but faint horizontal lines are subtly hinted at. Still lacks fine skin texture.\
4.	Eyes & Brows: Eye shape is maintained. Iris detail is improved over 32x32 but still somewhat simplified. Crow's feet are present but softened. Eyebrows are dark, texture is smoothed but better defined than 32x32.\
5.	Nose: Bridge, tip, and nostrils are better defined than 32x32, appearing closer to the original.\
6.	Lips & Mouth: Lips are thin. Smile reveals teeth. Teeth are more distinct than 32x32 but still appear very smooth and uniform, lacking natural texture.\
7.	Skin & Wrinkles: Skin texture is smoother than the original, lacking detailed pores, but less aggressively smoothed than the 32x32 version. Wrinkles around eyes and mouth are present but softened.\
8.	Tone & Lighting: Fair skin tone, with slightly more variation than 32x32. Lighting effects are better preserved, creating more realistic highlights and shadows than the 32x32 image.\
9.	Hairline & Hair on Face: Hairline is visible. Hair on forehead is better defined than 32x32, showing some strands but still quite smoothed.\
10.	Ears & Expression: Ears are partially visible, slightly smoothed.\
11.	Expression: Smile is present and appears more natural than the 32x32 version, though still slightly affected by the smoothing and simplified teeth.\
",
"1.	Face Shape: Angular/rectangular shape is well preserved and looks natural. Symmetry is good.\
2.	Jaw & Cheekbones: Jawline is clearly defined and angular, similar to the original. Cheekbones are prominent and well-defined, contributing effectively to the facial structure.\
3.	Forehead: Appears relatively smooth, but faint horizontal lines are visible. Still lacks fine pores or blemishes seen in the original.\
4.	Eyes & Brows: Eye shape and size are accurate. Crow's feet wrinkles are visible and appear quite similar to the original. Iris detail is improved and more natural-looking. Eyebrows are dark, with better defined texture, closer to individual hairs.\
5.	Nose: Bridge, rounded tip, and nostrils are well-defined and natural-looking, closely matching the original.\
6.	Lips & Mouth: Lips are thin, natural shape. Smile reveals teeth. Teeth look more natural, with individual shapes visible, though still quite smooth compared to the original.\
7.	Skin & Wrinkles: Skin texture is relatively smooth but retains more natural variation than lower-resolution GPEN results. Some subtle texture is present. Wrinkles are well-preserved, including crow's feet and smile lines.\
8.	Tone & Lighting: Fair skin tone, with good variation and natural-looking lighting and shadows that define the contours effectively.\
9.	Hairline & Hair on Face: Hairline is visible. Hair on forehead shows good definition of strands, similar to the original.\
10.	Ears & Expression: Ears are partially visible, well-defined.\
11.	Expression: The wide, happy smile is well-preserved and looks very natural, closely resembling the original expression.\
",
"1.	Face Shape: Angular/rectangular shape, looks symmetrical and balanced, very close to the original.\
2.	Jaw & Cheekbones: Jawline is well-defined and angular. Cheekbones are prominent and contribute to the facial structure, closely matching the original.\
3.	Forehead: Appears mostly covered by hair, visible areas retain natural texture. Horizontal lines are visible.\
4.	Eyes & Brows: Eye shape and size are accurate. Crow's feet are well-defined and present. Iris detail is clear and natural. Eyebrows are dark, with good individual hair texture.\
5.	Nose: Bridge, rounded tip, and nostrils look natural and well-defined, closely matching the original.\
6.	Lips & Mouth: Lips are thin, natural shape. Smile reveals teeth. Teeth appear natural, with visible individual shapes and slight variations, very similar to the original.\
7.	Skin & Wrinkles: Skin texture looks natural, with visible pores and subtle variations, although potentially slightly smoother overall than the original. Wrinkles are well-preserved, including crow's feet and smile lines.\
8.	Tone & Lighting: Fair skin tone, with natural variations. Lighting and shadows are consistent with the original, creating good depth.\
9.	Hairline & Hair on Face: Hairline is visible. Hair on forehead has good strand definition and appears natural.\
10.	Ears & Expression: Ears are partially visible, well-defined.\
11.	Expression: The wide, happy smile is well-preserved, looking natural and closely matching the original expression\
",
"1.	Face Shape: Angular shape is somewhat suggested, but the face appears distorted or stretched, particularly around the cheeks and jaw. Symmetry seems slightly off.\
2.	Jaw & Cheekbones: Jawline is present but appears somewhat soft and less defined. Cheekbones are present but lack natural prominence, appearing smoothed or distorted. A noticeable pinkish/reddish discoloration artifact is visible on the right cheek/jaw area.\
3.	Forehead: Appears smooth and lacking natural texture. Horizontal lines are mostly absent.\
4.	Eyes & Brows: Eye shape is roughly preserved, but detail is simplified. Iris detail is blurry. Crow's feet wrinkles are largely absent. Eyebrows are dark, thick, but smoothed, lacking hair texture.\
5.	Nose: Bridge and tip lack fine detail. Nostrils appear somewhat blurred or simplified.\
6.	Lips & Mouth: Lips are thin, shape is roughly correct. Smile reveals teeth. Teeth appear very smooth and uniform, lacking individual definition.\
7.	Skin & Wrinkles: Skin texture is very smooth and lacks natural pores or blemishes. The pink discoloration on the cheek/jaw is a significant artifact. Most wrinkles are removed.\
8.	Tone & Lighting: Skin tone is fair but uneven due to the discoloration artifact. Lighting pattern is partially preserved but appears less sharp and natural.\
9.	Hairline & Hair on Face: Hairline seems distorted or fuzzy. Hair on forehead is smoothed and lacks definition.\
10.	Ears & Expression: Ears are partially visible but appear distorted or blurred.\
11.	Expression: Smile is present, but the overall distortion, artifacts, and lack of fine detail make the expression appear less natural or even slightly unsettling\
",
"1.	Face Shape: Angular shape is better preserved than 32x32, but still shows subtle artifacts or distortions, particularly around the right side of the face. Symmetry is better but not perfect.\
2.	Jaw & Cheekbones: Jawline is defined, but perhaps slightly less sharp than the original. Cheekbones are more prominent than 32x32 but still slightly smoothed. The discoloration artifact is still somewhat visible on the right cheek/jaw, but less prominent than 32x32.\
3.	Forehead: Smooth texture, lacking fine natural details. Faint horizontal lines might be present but are softened.\
4.	Eyes & Brows: Eye shape and size are maintained. Iris detail is improved over 32x32. Crow's feet are present but appear softened. Eyebrows are dark, showing some texture but still smoother than the original.\
5.	Nose: Bridge, rounded tip, and nostrils are better defined than 32x32, appearing closer to the original but possibly slightly smoothed.\
6.	Lips & Mouth: Lips are thin, natural shape. Smile reveals teeth. Teeth are more distinct than 32x32, but still appear very smooth and uniform.\
7.	Skin & Wrinkles: Skin texture is smooth, lacking fine pores. Wrinkles are present but smoothed compared to the original. The pink discoloration artifact is less noticeable but still present.\
8.	Tone & Lighting: Fair skin tone, still affected by the subtle discoloration artifact. Lighting is better preserved than 32x32, creating more natural highlights and shadows.\
9.	Hairline & Hair on Face: Hairline is visible. Hair on forehead shows slightly better definition than 32x32 but still appears smoothed.\
10.	Ears & Expression: Ears are partially visible, slightly smoothed.\
11.	Expression: Smile is present, looks more natural than 32x32, but the residual artifact and smoothing still affect the overall realism.\
",
"1.	Face Shape: Angular/rectangular shape is well preserved and looks natural. Symmetry is good.\
2.	Jaw & Cheekbones: Jawline is clearly defined and angular, closely matching the original. Cheekbones are prominent and well-defined. No significant artifacts or discoloration are visible.\
3.	Forehead: Appears smooth but retains some subtle, natural-looking texture. Horizontal lines are visible but perhaps slightly smoothed compared to the original.\
4.	Eyes & Brows: Eye shape and size are accurate. Iris detail is clearer and appears natural. Crow's feet wrinkles are well-defined and look natural. Eyebrows are dark, showing good hair texture.\
5.	Nose: Bridge, rounded tip, and nostrils are well-defined and look natural, closely matching the original.\
6.	Lips & Mouth: Lips are thin, natural shape. Smile reveals teeth. Teeth look natural, with individual shapes visible, although potentially slightly sharpened or smoothed compared to the original.\
7.	Skin & Wrinkles: Skin texture appears fairly natural, with visible pores and subtle variations. Wrinkles (crow's feet, smile lines) are well-preserved and look natural.\
8.	Tone & Lighting: Fair skin tone, with good natural variation. Lighting and shadows are well-rendered, creating good depth and definition.\
9.	Hairline & Hair on Face: Hairline is visible. Hair on forehead shows good definition of strands, similar to the original.\
10.	Ears & Expression: Ears are partially visible, well-defined.\
11.	Expression: The wide, happy smile is well-preserved and looks very natural, closely resembling the original expression\
",
"1.	Face Shape: Angular/rectangular shape, looks symmetrical and balanced, almost identical to the original.\
2.	Jaw & Cheekbones: Jawline is sharply defined and angular. Cheekbones are prominent and well-defined, very close to the original.\
3.	Forehead: Visible skin retains natural texture. Horizontal lines are visible and clearly defined.\
4.	Eyes & Brows: Eye shape and size are accurate. Iris detail is sharp and clear. Crow's feet are sharply defined. Eyebrows are dark, with excellent individual hair texture, perhaps slightly enhanced sharpness compared to original.\
5.	Nose: Bridge, rounded tip, and nostrils look very natural and well-defined, closely matching the original.\
6.	Lips & Mouth: Lips are thin, natural shape. Smile reveals teeth. Teeth look natural, with visible individual shapes and slight variations, possibly slightly sharper/crisper than the original.\
7.	Skin & Wrinkles: Skin texture looks very natural and detailed, perhaps slightly sharper or with more pronounced micro-texture than the original. Visible pores and subtle variations are present. Wrinkles are sharply defined and look natural.\
8.	Tone & Lighting: Fair skin tone with natural variations. Lighting and shadows are consistent with the original and well-defined.\
9.	Hairline & Hair on Face: Hairline is visible. Hair on forehead has sharp strand definition and appears natural.\
10.	Ears & Expression: Ears are partially visible, well-defined and detailed.\
11.	Expression: The wide, happy smile is sharply defined and looks very natural, closely matching the original expression with potentially enhanced detail\
",
"1.	Face Shape: Angular/rectangular shape, looks symmetrical and natural. Appears very similar to the original.\
2.	Jaw & Cheekbones: Jawline is well-defined and angular. Cheekbones are prominent and look natural.\
3.	Forehead: Visible skin appears relatively smooth, but retains some natural texture. Horizontal lines are visible but might be slightly smoothed compared to the original or CodeFormer 256x256.\
4.	Eyes & Brows: Eye shape and size are accurate. Iris detail is clear and appears natural, possibly slightly softer than CodeFormer 256x256. Crow's feet are present but potentially slightly smoothed. Eyebrows are dark, with good definition but perhaps slightly softer hair texture than CodeFormer.\
5.	Nose: Bridge, rounded tip, and nostrils look natural and well-defined.\
6.	Lips & Mouth: Lips are thin, natural shape. Smile reveals teeth. Teeth look natural, with visible individual shapes, possibly slightly smoother or more uniform than the original or CodeFormer 256x256.\
7.	Skin & Wrinkles: Skin texture looks natural and smooth, less sharply defined than the CodeFormer 256x256 result, possibly slightly idealized but still realistic. Visible pores and variations are present but subtle. Wrinkles are present but appear slightly smoothed.\
8.	Tone & Lighting: Fair skin tone with natural variations. Lighting and shadows are consistent with the original, creating good depth.\
9.	Hairline & Hair on Face: Hairline is visible. Hair on forehead has good definition and appears natural, possibly slightly softer than CodeFormer 256x256.\
10.	Ears & Expression: Ears are partially visible, well-defined.\
11.	Expression: The wide, happy smile is well-preserved, looking natural and closely matching the original expression, with a focus on naturalness and coherence rather than extreme sharpness.\
"
     ]
    
    
    results = comparator.calculate_scores(original, generated)

    # Write results to txt file in structured format
    with open("comparsion_results_22610.txt", "w", encoding="utf-8") as f:
        for result in results:
            f.write(f"--- Result {result['index'] + 1} ---\n")
            f.write(f"Score: {result['score']}\n")
            f.write("Differences:\n")
            if result['differences']:
                for feature, change in result['differences'].items():
                    added = ', '.join(change['added']) if change['added'] else 'None'
                    removed = ', '.join(change['removed']) if change['removed'] else 'None'
                    f.write(f"  {feature}:\n")
                    f.write(f"    Added: {added}\n")
                    f.write(f"    Removed: {removed}\n")
            else:
                f.write("  None\n")
            f.write("\n")

    print("Comparison results written to comparison_results.txt")
