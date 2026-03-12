from flask import Flask, render_template, request, redirect, url_for, session, flash
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image
import sqlite3

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# File upload settings
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        email TEXT,
        password TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Class and remedy info...

# # Disease classes and descriptions
classes = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
           'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

descriptions = {
    'Anthracnose': 'A fungal disease causing dark spots on leaves and fruit.',
    'Bacterial Canker': 'Bacterial infection causing sunken lesions on stems and leaves.',
    'Cutting Weevil': 'Insect damage appearing as cuts or notches on leaf edges.',
    'Die Back': 'Gradual death of branches from tips inward.',
    'Gall Midge': 'Tiny insects that form galls on leaves, distorting their shape.',
    'Healthy': 'No visible disease or insect damage.',
    'Powdery Mildew': 'Fungal growth appearing as white powder on surfaces.',
    'Sooty Mould': 'Black moldy covering usually due to insect honeydew.'
}

remedies = {
    'Anthracnose': {
        'remedy': 'Use copper-based fungicide and prune infected areas.',
        'steps': [
            'Remove infected plant parts.',
            'Apply copper-based fungicide regularly.',
            'Avoid overhead watering.',
            'Maintain good air circulation.'
        ]
    },
    'Bacterial Canker': {
        'remedy': 'Apply bactericides and ensure good drainage.',
        'steps': [
            'Remove affected branches.',
            'Disinfect pruning tools.',
            'Improve air flow around plants.',
            'Apply copper sprays in early season.'
        ]
    },
    'Cutting Weevil': {
        'remedy': 'Use insecticides and remove affected leaves.',
        'steps': [
            'Inspect plants regularly.',
            'Remove and destroy damaged leaves.',
            'Use recommended insecticides.',
            'Promote beneficial insects.'
        ]
    },
    'Die Back': {
        'remedy': 'Improve plant nutrition and prune affected parts.',
        'steps': [
            'Cut back dead branches.',
            'Fertilize adequately.',
            'Improve drainage.',
            'Apply protective fungicides.'
        ]
    },
    'Gall Midge': {
        'remedy': 'Apply systemic insecticides and remove galls.',
        'steps': [
            'Remove and destroy galled tissues.',
            'Spray systemic insecticides.',
            'Encourage natural predators.',
            'Avoid nitrogen over-fertilization.'
        ]
    },
    'Healthy': {
        'remedy': 'No treatment required.',
        'steps': [
            'Continue regular care and observation.',
            'Maintain plant hygiene.'
        ]
    },
    'Powdery Mildew': {
        'remedy': 'Use sulfur-based fungicides and prune affected areas.',
        'steps': [
            'Remove infected leaves.',
            'Apply sulfur or neem oil.',
            'Avoid overcrowding plants.',
            'Water in early morning.'
        ]
    },
    'Sooty Mould': {
        'remedy': 'Control insect pests like aphids and use mild soap solution.',
        'steps': [
            'Clean leaves with soap solution.',
            'Control aphids and whiteflies.',
            'Use neem oil regularly.',
            'Ensure sunlight exposure.'
        ]
    }
}


# Load model
img_height, img_width = 224, 224
model = Sequential()
model.add(ResNet50(include_top=False, pooling='max', weights='imagenet', input_shape=(224, 224, 3)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8))
model.layers[0].trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.load_weights('mango_leaf_classification_model_weights_omdena_resnet50.hdf5')


def process_image(file_path):
    img = Image.open(file_path).resize((img_width, img_height)).convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def edge_segment_image(file_path):
    img = cv2.imread(file_path)
    img_resized = cv2.resize(img, (224, 224))
    edges = cv2.Canny(img_resized, 100, 200)
    edge_path = os.path.join(app.config['UPLOAD_FOLDER'], 'segmented.png')
    cv2.imwrite(edge_path, edges)
    return 'uploads/segmented.png'  # relative to static folder

# 🟢 Welcome Page
@app.route('/')
def welcome():
    return render_template('welcome.html')

# 🟢 Register Page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email    = request.form['email']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, password))
            conn.commit()
            flash('Registration successful. Please log in.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists!')
        conn.close()
    return render_template('register.html')

# 🟢 Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('Invalid login credentials')
    return render_template('login.html')

# 🟢 Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))

# 🟢 Main Prediction Page (after login)

@app.route('/indexo', methods=['GET', 'POST'])
def index():
    prediction = None
    class_info = None
    uploaded_image = None
    segmented_image = None
    remedy_info = None
    remedy_steps = []

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(full_path)

            uploaded_image = f'uploads/{filename}'

            img = process_image(full_path)
            pred = model.predict(img)
            class_index = np.argmax(pred)
            prediction = classes[class_index]
            class_info = descriptions[prediction]
            remedy_info = remedies[prediction]['remedy']
            remedy_steps = remedies[prediction]['steps']

            segmented_image = edge_segment_image(full_path)

    return render_template('indexo.html',
                           prediction=prediction,
                           class_info=class_info,
                           uploaded_image=uploaded_image,
                           segmented_image=segmented_image,
                           remedy_info=remedy_info,
                           remedy_steps=remedy_steps)


if __name__ == '__main__':
    app.run(debug=True)