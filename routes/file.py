from flask import request, render_template, Blueprint
from controllers.segmentation import run_segmentation
import os

file = Blueprint('file', __name__)  # Define the Blueprint

@file.route('/file')
def file_upload_page():
    return render_template('file.html')

@file.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return "No file selected"
    
    path = os.path.join('uploads', uploaded_file.filename)
    uploaded_file.save(path)
    
    result_df = run_segmentation(path)
    return render_template('dashboard.html', tables=[result_df.to_html(classes='data')])
