from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import subprocess
import shutil

app = Flask(__name__, static_folder='static', template_folder='templates')

UPLOAD_FOLDER = 'static/images/uploads/'
RESULTS_FOLDER = 'static/images/results/'
ARCHIVE_FOLDER = 'static/images/archive/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(ARCHIVE_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            try:
                subprocess.run(
                    ['python', 'yolo/detect.py'], check=True
                )

                archive_file_path = os.path.join(ARCHIVE_FOLDER, filename)
                shutil.move(file_path, archive_file_path)

                image_path = f"images/results/{filename}"

                tree_count_file = f"{RESULTS_FOLDER}/{filename}_tree_count.txt"
                with open(tree_count_file, "r") as f:
                    tree_count = f.read().strip()

                return redirect(url_for('result', image_path=image_path, tree_count=tree_count))

            except Exception as e:
                return f"Error processing image: {str(e)}"
            
    return render_template('index.html')

@app.route('/result')
def result():
    image_path = request.args.get('image_path')
    tree_count = request.args.get('tree_count')
    return render_template('result.html', image_path=image_path, tree_count=tree_count)

@app.route('/cleanup', methods=['POST'])
def cleanup():
    try:
        results_folder = app.config['RESULTS_FOLDER']

        if os.path.exists(results_folder):
            for filename in os.listdir(results_folder):
                file_path = os.path.join(results_folder, filename)

                if os.path.isfile(file_path):
                    os.remove(file_path)

                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

        return "Cleanup complete", 200
    
    except Exception as e:
        return f"Error during cleanup: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)