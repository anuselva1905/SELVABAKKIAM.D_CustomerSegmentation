import os
from flask import app, request, jsonify
from controllers.segmentation import run_segmentation

@app.route('/segment', methods=['POST'])
def segment():
    file = request.files['file']
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    result_df = run_segmentation(file_path)
    return result_df.to_json(orient='records')
