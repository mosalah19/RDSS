from flask import Flask, request, jsonify, send_file
import numpy as np
from PIL import Image
import io
from intialized_models import *
from io import BytesIO

app = Flask(__name__)


@app.route('/segmentation', methods=['POST'])
def segmentation():
    # Receive image data from the request
    image_file = request.files['image']
    image_bytes = image_file.read()
    image_pil = Image.open(io.BytesIO(image_bytes))

    # Convert the PIL Image to a NumPy array
    image_np = np.array(image_pil)
    image_array = model_1(image_np,
                          r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\models_pth\Segmentation_model.pth')
    image_pil = Image.fromarray(image_array)

    # Create an in-memory byte stream buffer to hold the image data
    image_buffer = BytesIO()

    # Save the PIL Image to the byte stream buffer as PNG format
    image_pil.save(image_buffer, format='PNG')

    # Seek to the beginning of the stream
    image_buffer.seek(0)

    # Return the image data as a file using Flask's send_file function
    return send_file(image_buffer, mimetype='image/png')


@app.route('/classification', methods=['POST'])
def classification():
    # Receive image path from the request
    image_file = request.files['image']
    image_bytes = image_file.read()
    image_pil = Image.open(io.BytesIO(image_bytes))

    # Convert the PIL Image to a NumPy array
    image_np = np.array(image_pil)
    C = prediction(image_pil,
                   model_path=r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\models_pth\Cataract.pth', dignosis="C")
    G = prediction(image_pil,
                   model_path=r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\models_pth\model_for_G.pth', dignosis="G")
    Binary_D = prediction(image_pil,
                          model_path=r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\models_pth\Binary_DR.pth', dignosis="D&B")
    MultiClass_D = prediction(image_np,
                              model_path=r'D:\Graduated Project\Retinal_Diseases_Diagnosis_Support_System\models_pth\MultiDR.pth', dignosis="D")

    return jsonify({'Catarect': C.item(), 'Galucoma': G.item(), 'Binary_D': Binary_D.item(), 'MultiClass_D': MultiClass_D.item()})


if __name__ == '__main__':
    app.run(debug=True)
