from flask import render_template,Blueprint


home = Blueprint('home', __name__)

@home.route('/')
def index():
    from app import app  # Lazy import inside the function
    return render_template('home.html')