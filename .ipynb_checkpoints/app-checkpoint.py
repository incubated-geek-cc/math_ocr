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
img_path = wd + '\\formulae\\test\\0000013.png'
print(img_path)

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

@app.route("/test", methods=['POST'])
def test_method():       
    if not request.json or 'image' not in request.json: 
        abort(400)
        
    print(request.json)   
    im_b64 = request.json['image']

    # convert it into bytes  
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))

    # PIL image object to numpy array
    img_arr = np.asarray(img)      
    print('img shape', img_arr.shape)
    # process your img_arr here   
    # access other keys of json
    # print(request.json['other_key'])
    result_dict = {'output': 'output_key'}
    return result_dict
    
@app.route("/input_math_ocr", methods=['POST'])
def input_method():         
    # print(request.json)      
    if not request.json or 'image' not in request.json: 
        abort(400)
             
    # get the base64 encoded string
    im_b64 = request.json['image']
    
    # convert it into bytes  
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))

    # convert bytes data to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))

    # PIL image object to numpy array
    img_arr = np.asarray(img)      
    print('img shape', img_arr.shape)

    # process your img_arr here    
    
    # access other keys of json
    # print(request.json['other_key'])

    result_dict = {'output': 'output_key'}
    return result_dict
    
    if not request.json or 'image' not in request.json: 
        abort(400)
             
    # get the base64 encoded string
    im_b64 = request.json['image']
    print(im_b64)
    # convert it into bytes  
    img_bytes = base64.b64decode(im_b64.encode('utf-8'))
    res, elapse = model(img_bytes)
    print(res)
    print(elapse)
    result_dict = {'result': res,'time_taken':elapse}
    return result_dict
  
  
def run_server_api():
    app.run(host='0.0.0.0', port=8000, debug=True)
  
  
if __name__ == "__main__":     
    run_server_api()