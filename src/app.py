from flask import Flask, render_template, request, url_for
from base64 import b64encode
from PIL import Image
import io
import glob
import os
from model import Model
from files import upload_image, save_sparse_matrix, save
from vectorize import vectorize_all

app = Flask(__name__, static_folder='/usr/src/app')

#########################################################

def main(dir_name):

    if os.path.exists('./data/result.npz') == False :
        files = glob.glob(dir_name, recursive=True)
        save(files, "./data/files")
    else:
        files = open('./data/files', 'r').readline().split(',')

    model = Model.get_model()
   
    if os.path.exists('./data/result.npz') == False :
        vecs = vectorize_all(files, model, n_dims=4096)
        save_sparse_matrix("./data/result", vecs)

########################################################

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    
    img = Image.open(file.stream).convert('RGB')

    files = upload_image(img)

    image_io = io.BytesIO()
    img.save(image_io, "PNG")
    image_io.seek(0)

    dataurl = 'data:image/png;base64,' + b64encode(image_io.getvalue()).decode('ascii')

    return render_template('index.html', files=files, img=dataurl)

if __name__ == "__main__":
    main('./data/**/*.jpg')
    app.run(host='0.0.0.0', port=80)