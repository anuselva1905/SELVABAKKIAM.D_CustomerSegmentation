import os
from flask import request, Blueprint
from flask.json import jsonify
from controllers.segmentation import run_segmentation

# Create the Blueprint
file = Blueprint('dashboard', __name__)

# Define the route for segmentation
@file.route('/segment', methods=['POST'])
def segment():
    uploaded_file = request.files['file']  # Get the file from the request
    if uploaded_file.filename == '':
        return jsonify({"error": "No file selected"}), 400  # Return error if no file selected
    
    # Save the uploaded file
    file_path = os.path.join('uploads', uploaded_file.filename)
    uploaded_file.save(file_path)
    
    # Run segmentation
    result_df = run_segmentation(file_path)

    # Return the result in JSON format
    return result_df.to_json(orient='records')

