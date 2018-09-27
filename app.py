from flask import Flask
import os

secret_key = os.environ.get('SECRET_KEY')

app = Flask(__name__)
app.secret_key = secret_key


@app.route('/')
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    port = int(os.environ.get('PORT', 5000))
    app.run()