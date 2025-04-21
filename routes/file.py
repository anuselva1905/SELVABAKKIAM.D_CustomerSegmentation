from flask import request, render_template
from app import app
from controllers.segmentation import run_segmentation
import os

@app.route('/file')
def file_upload_page():
    return render_template('file.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file.filename == '':
        return "No file selected"
    
    path = os.path.join('uploads', file.filename)
    file.save(path)
    
    result_df = run_segmentation(path)
    return render_template('dashboard.html', tables=[result_df.to_html(classes='data')])
