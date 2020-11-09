import os
from flask import Flask, request, render_template, current_app, send_from_directory
from m import customer_seg

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        customer_seg(os.path.join(current_app.root_path, f.filename))
        return render_template("success.html", name=f.filename)


@app.route('/get_result', methods=['GET'])
def get_result():
    return send_from_directory(directory=current_app.root_path, filename="output/result.csv")


if __name__ == "__main__":
    app.run(debug=True)