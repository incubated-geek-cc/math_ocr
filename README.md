# Math OCR

## Motivation
<p>Enable more intuitive rendering of complex mathematical notations into LaTeX typesettings without having in-depth knowledge of LaTeX syntax.</p>

## Features & Functionalities
- Open Neural Network Exchange (Onnx) OCR models at <a href='https://github.com/incubated-geek-cc/math_ocr/tree/main/models' target='_blank'>📁 models</a> were originally retrieved from <a href='https://github.com/RapidAI/RapidLaTeXOCR' target='_blank'>RapidLaTeXOCR</a> and deployed on Python Flask App

- Onnx OCR models reads input graphical data of math equations and outputs:
	- Math Inline notations
	- LaTeX typesetting using <a href='https://katex.org/' target='_blank'>KaTeX</a> JavaScript library
	- HTML markup of formatted output

## Preview

### Upload image of math formualae
<p><img src='https://cdn-images-1.medium.com/v2/resize:fit:800/1*GbFMaFHxwK_l5NZdtWmxUQ.gif' alt='Upload image' width='800'></p>

### Draw math equation on HTML canvas
<p><img src='https://cdn-images-1.medium.com/v2/resize:fit:800/1*Uy7LqUVZ4SctZMyTGG7Lhg.gif' alt='Draw on canvas' width='800'></p>

## Demo
<p>Python Flask Web App deployed on <a href='https://math-ocr.onrender.com' target='_blank'>Render</a>.</p>

### Purpose of .bat files
| Filename  | Description  |
| :-------- | :----------- |
| requirements.txt | Contains list of Python package dependencies for application |
| pip_freeze.bat | Extract used python packages into`requirements.txt` file |
| pip_install_requirements.bat | Install all python packages in`requirements.txt` file  |
| activate_env.bat | Activate virtual environment named `.env` |
| run_app.bat | Run web app in dev environment on port 5000 |


## Credits and Acnowledgement
* <a href='https://github.com/RapidAI/RapidLaTeXOCR' target='_blank'>RapidLaTeXOCR</a> (for Onnx Math OCR models)
* <a href='https://katex.org/' target='_blank'>KaTeX</a> JavaScript library, an open-sourced library to display LaTeX-formatted math formulae and equations on digital print

## Related Readings
### Published in <a href='https://geek-cc.medium.com/deploy-math-ocr-onnx-model-in-python-flask-web-app-fd2aab576eb0'>Deploy Math OCR ONNX Model In Python Flask Web App</a>

<p>— <b>Join me on 📝 <b>Medium</b> at <a href='https://medium.com/@geek-cc' target='_blank'>~ ξ(🎀˶❛◡❛) @geek-cc</a></b></p>

---

#### 🌮 Please buy me a <a href='https://www.buymeacoffee.com/geekcc' target='_blank'>Taco</a>! 😋