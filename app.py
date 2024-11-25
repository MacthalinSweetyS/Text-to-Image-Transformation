import torch
from diffusers import StableDiffusionPipeline
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Initialize the Stable Diffusion pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)

# Ensure the directory to save images exists
os.makedirs('generated_images', exist_ok=True)

def apply_colors_to_prompt(prompt, colored_words):
    color_prompt = prompt
    for word, color in colored_words.items():
        color_prompt += f", {word} in {color}"
    return color_prompt

@app.route('/generate_image', methods=['POST'])
def generate_image_endpoint():
    data = request.json
    prompt = data.get('text')
    colored_words = data.get('colored_words', {})

    if not prompt:
        return jsonify({'error': 'No text provided'}), 400

    # Modify the prompt with colors
    color_prompt = apply_colors_to_prompt(prompt, colored_words)

    try:
        # Generate the image using the modified prompt
        image = pipe(color_prompt, num_inference_steps=50).images[0]

        # Save the image to the specified folder
        image_path = f'generated_images/{prompt[:10].replace(" ", "_")}.png'
        image.save(image_path)

        # Convert image to base64
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

        return jsonify({'image': img_base64})
    except Exception as e:
        app.logger.error(f'Error generating image: {e}')
        return jsonify({'error': 'Error generating image'}), 500

if __name__ == '__main__':
    app.run(debug=True)
