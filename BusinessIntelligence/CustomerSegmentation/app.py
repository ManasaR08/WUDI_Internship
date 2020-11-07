import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from model import customer_seg


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload',methods = ['POST'])  
def upload():
    if request.method =='POST':
        f=request.files['file']
        f.save(f.filename) 
        path = f"D:\WUDI_Internship\BusinessIntelligence\CustomerSegmentation\\"
        customer_seg(path +  f.filename)
        return render_template("success.html",name=f.filename, value = "result.csv")



if __name__ == "__main__":
    app.run(debug=True)        