import os
import sys
import json
import subprocess
from flask import Flask, render_template, request, flash, jsonify, Response
from vector_db_creator import create_vector_db, create_vector_db_stream

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from secret.secret_info import HUG_FACE_KEY
    HF_TOKEN = HUG_FACE_KEY
except ImportError:
    HF_TOKEN = os.environ.get('HF_TOKEN', '')

app = Flask(__name__)
app.secret_key = os.urandom(24)


@app.route('/')
def index():
    return render_template('index.html')


def resolve_output(input_folder, output_name=None, output_folder=None):
    if output_folder:
        return output_folder
    base = os.path.basename(input_folder.rstrip('/\\'))
    name = output_name or (base + '_vector_db')
    return os.path.join(os.path.dirname(input_folder.rstrip('/\\')), name)


@app.route('/create', methods=['POST'])
def create():
    input_folder = request.form.get('input_folder', '').strip()
    output_name = request.form.get('output_name', '').strip()
    output_folder_picked = request.form.get('output_folder', '').strip()

    if not input_folder:
        flash('Please provide the input folder path.', 'error')
        return render_template('index.html')

    output_folder = resolve_output(input_folder, output_name, output_folder_picked or None)
    result = create_vector_db(input_folder, output_folder, hf_token=HF_TOKEN)

    if result['success']:
        flash(f"Database created successfully! {result['total_vectors']} vectors from {result['total_processed']} files.", 'success')
    else:
        flash(f"Error: {result['error']}", 'error')

    return render_template('index.html', result=result)


@app.route('/create/stream')
def create_stream():
    input_folder = request.args.get('input_folder', '').strip()
    output_name = request.args.get('output_name', '').strip()
    output_folder_picked = request.args.get('output_folder', '').strip()

    if not input_folder:
        def no_folder():
            yield f"data: {json.dumps({'type': 'error', 'error': 'No input folder provided.'})}\n\n"
        return Response(no_folder(), mimetype='text/event-stream')

    output_folder = resolve_output(input_folder, output_name, output_folder_picked or None)

    def generate():
        for event in create_vector_db_stream(input_folder, output_folder, hf_token=HF_TOKEN):
            yield f"data: {json.dumps(event)}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/cancel', methods=['POST'])
def cancel():
    import shutil
    folder = request.form.get('folder', '').strip()
    if folder and os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)
        return jsonify({'cancelled': True, 'folder': folder})
    return jsonify({'cancelled': False})


@app.route('/browse', methods=['POST'])
def browse():
    folder = ''
    try:
        proc = subprocess.run(
            ['osascript', '-e', 'return POSIX path of (choose folder)'],
            capture_output=True, text=True, timeout=120
        )
        if proc.returncode == 0:
            folder = proc.stdout.strip()
    except Exception as e:
        print(f'Browse error: {e}', flush=True)
    return jsonify({'folder': folder})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
