from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import tempfile
import ffmpeg
import subprocess

app = Flask(__name__)

# Titan effektlari parametrlari
TITAN_PARAMS = {
    'edge_thickness': 1.5,
    'color_palette': 24,
    'darkness_level': 0.65,
    'contrast': 2.0,
    'blood_color': (150, 30, 30),
    'eye_glow': 3.0,
    'muscle_definition': 1.7,
    'smoke_intensity': 0.4,
    'damage_texture': 0.3,
    'titan_scale': 1.1,
}

def add_titan_effects(frame):
    """Rasmga Titan effektlarini qo'llash"""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 1. Konturlarni yaratish
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
    _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY_INV)
    edges_pil = Image.fromarray(edges).convert('L')
    
    # 2. Rang kvantizatsiyasi
    quantized = pil_img.quantize(colors=TITAN_PARAMS['color_palette'], method=2).convert('RGB')
    enhancer = ImageEnhance.Contrast(quantized)
    contrast_img = enhancer.enhance(TITAN_PARAMS['contrast'])
    
    # 3. Muskul detallari
    embossed = contrast_img.filter(ImageFilter.EMBOSS)
    muscle_enhanced = Image.blend(contrast_img, embossed, TITAN_PARAMS['muscle_definition']/5)
    
    # 4. Qon effektlari
    blood_overlay = Image.new('RGB', pil_img.size, TITAN_PARAMS['blood_color'])
    blood_mask = edges_pil.point(lambda x: 0 if x < 100 else 150)
    wounded_img = Image.composite(muscle_enhanced, blood_overlay, blood_mask)
    
    # 5. Ko'z yorqinligi
    eyes_enhanced = ImageEnhance.Brightness(wounded_img).enhance(TITAN_PARAMS['eye_glow'])
    
    # 6. Titan o'lchami
    scaled_img = eyes_enhanced.resize(
        (int(eyes_enhanced.width * TITAN_PARAMS['titan_scale']), 
        int(eyes_enhanced.height * TITAN_PARAMS['titan_scale'])), 
        Image.LANCZOS
    )
    position = ((pil_img.width - scaled_img.width) // 2, (pil_img.height - scaled_img.height) // 2)
    final_img = Image.new('RGB', pil_img.size, (0, 0, 0))
    final_img.paste(scaled_img, position)
    
    return cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2BGR)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({"error": "Video fayl topilmadi!"}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "Fayl tanlanmagan!"}), 400

    # Saqlash uchun vaqtinchalik fayl
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, "input.mp4")
    output_path = os.path.join(temp_dir, "output.mp4")
    video_file.save(input_path)

    # Videoni qayta ishlash
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = add_titan_effects(frame)
        out.write(processed_frame)

    cap.release()
    out.release()

    # Foydalanuvchiga yuborish
    return send_file(output_path, as_attachment=True, download_name="titan_effect.mp4")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)