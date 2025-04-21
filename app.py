from flask import Flask
from routes.file import file  # Import the file blueprint

def create_app():
    app = Flask(__name__)

    # Register the blueprint
    app.register_blueprint(file)

    return app

# Create and run the app
app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
