 
import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
import numpy as np
import argparse

import torch
import torch.nn as nn 
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.models import mobilenet_v2

from utils.utils import to_original_index


ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
UPLOAD_FOLDER = 'data/images'


app = Flask(__name__)

parser = argparse.ArgumentParser(description = 'PyTorch inference of MobileNetV2 on Rakuten Catalog data')

parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--checkpoint',  default = 'checkpoints/final_model.pt', type = str, help = 'Checkpoint to use for inference')

args = parser.parse_args()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def predict(file):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mobilenet_v2()
    checkpoint = torch.load(args.checkpoint)
    model.classifier[1] = nn.Linear(1280, list(checkpoint['model_state_dict']['classifier.1.bias'].shape)[0])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    image = ToTensor()(Image.open(file).convert('RGB')).unsqueeze(0)
    image = image.to(device)

    preds = model(image)
    y_pred_softmax = torch.softmax(preds, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)


    output = y_pred_tags.item()
    return output

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('home.html', label='', imagesource='file://null')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = predict(file_path)
            output = 'Predicted Class : {}'.format(to_original_index(output))
    return render_template("home.html", label=output, imagesource=file_path)


@app.route('/data/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(debug=False, threaded=False)