from flask import Flask
import secrets

# Initialize Flask app with templates folder
application = app = Flask(__name__, template_folder='../templates')


app.secret_key = 'b5a3b0b4658f82b2616b0e655e2b776b'

