from LaTeXOCR import LaTeXOCR
import io
import json                    
import base64                  
import logging

from flask import Flask, request, jsonify, abort, render_template

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

import os
wd = os.getcwd()

image_resizer_path = wd + '\\models\\image_resizer.onnx'
encoder_path = wd + '\\models\\encoder.onnx'
decoder_path = wd + '\\models\\decoder.onnx'
tokenizer_json = wd + '\\models\\tokenizer.json'
model = LaTeXOCR(image_resizer_path=image_resizer_path,
                encoder_path=encoder_path,
                decoder_path=decoder_path,
                tokenizer_json=tokenizer_json)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/math_ocr/post", methods=['POST'])
def input_method():
    print(request)
    if not request.json or 'im_b64' not in request.json: 
        abort(400)
             
    # get the base64 encoded string
    im_b64 = request.json['im_b64']
    # print(im_b64)
    # convert it into bytes  
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))
    res, elapse = model(img_bytes)
    print(res)
    print(elapse)
    result_dict = {'result': res,'time_taken':elapse}
    return jsonify(result_dict)
  
  
def run_server_api():
    app.run(debug=True)
  
  
if __name__ == "__main__":     
    run_server_api()