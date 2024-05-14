from flask import Flash

app = Flask(__name__)

@app.route('/')
def home():
    return "hello world"

@app.route('/about')
def about():
    return "nice"