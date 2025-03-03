#!/usr/bin/env python3
"""
Magnificent Fractal Art Generator - Main Application
"""
import os
import io
import base64
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from fractal_generator import MandelbrotSet, JuliaSet, BurningShip, generate_fractal_image

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'development-key-12345')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///fractals.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    fractals = db.relationship('Fractal', backref='author', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Fractal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    fractal_type = db.Column(db.String(20), nullable=False)
    parameters = db.Column(db.Text, nullable=False)  # JSON string of parameters
    image_path = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    public = db.Column(db.Boolean, default=True)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    try:
        app.logger.info("Fractal generation started")
        data = request.json
        app.logger.info(f"Received parameters: {data}")
        
        fractal_type = data.get('type', 'mandelbrot')
        width = min(int(data.get('width', 800)), 1200)  # Limit max size
        height = min(int(data.get('height', 600)), 1000)  # Limit max size
        max_iter = min(int(data.get('max_iter', 100)), 500)  # Limit max iterations
        x_min = float(data.get('x_min', -2.5))
        x_max = float(data.get('x_max', 1.5))
        y_min = float(data.get('y_min', -1.5))
        y_max = float(data.get('y_max', 1.5))
        color_scheme = data.get('color_scheme', 'viridis')
        
        c_real = float(data.get('c_real', -0.7))
        c_imag = float(data.get('c_imag', 0.27015))
        
        # Calculate complexity score to predict if this will be a slow generation
        pixel_count = width * height
        complexity = pixel_count * max_iter
        
        app.logger.info(f"Complexity score: {complexity}")
        
        # Set a reasonable limit for complexity to prevent server overload
        if complexity > 50000000:  # 50 million operations is a good threshold
            app.logger.warning(f"Rejecting request due to high complexity: {complexity}")
            return jsonify({
                'error': 'Parameters too complex',
                'message': 'The combination of resolution and iterations is too high. Try reducing either value.'
            }), 400
        
        app.logger.info("Starting fractal calculation...")
        # Generate the fractal image based on type
        try:
            img_data = generate_fractal_image(
                fractal_type=fractal_type,
                width=width,
                height=height,
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                max_iter=max_iter,
                color_scheme=color_scheme,
                c_real=c_real,
                c_imag=c_imag
            )
            app.logger.info("Fractal calculation completed successfully")
        except Exception as calc_error:
            app.logger.error(f"Fractal calculation failed: {str(calc_error)}")
            # If calculation fails, generate a simple gradient as fallback
            app.logger.info("Generating fallback image")
            img_data = create_fallback_image(width, height, color_scheme)
            
        # Convert to base64 for sending to frontend
        buffer = io.BytesIO()
        img_data.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        app.logger.info("Returning image data to client")
        return jsonify({
            'image': f'data:image/png;base64,{img_str}',
            'parameters': data,
            'fallback': img_data is None
        })
    except Exception as e:
        app.logger.error(f"Error in generate route: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Failed to generate fractal. Try reducing resolution or iterations.'
        }), 500


def create_fallback_image(width, height, color_scheme='viridis'):
    """Create a simple gradient fallback image when fractal generation fails"""
    # Create a simple gradient as a fallback
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    gradient = x[:, np.newaxis] * y[np.newaxis, :]
    
    # Apply colormap
    cmap = plt.get_cmap(color_scheme)
    colored_data = cmap(gradient)
    
    # Convert to 8-bit RGB
    colored_data = (colored_data[:, :, :3] * 255).astype(np.uint8)
    
    # Create PIL image
    img = Image.fromarray(colored_data)
    
    # Add text to indicate this is a fallback image
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Add text explaining the fallback
    message = "Fractal generation failed. Try simpler parameters."
    text_width = draw.textlength(message, font=font)
    position = ((width - text_width) // 2, height // 2)
    
    # Add a background for the text to ensure readability
    text_bg = ((position[0] - 10, position[1] - 10), 
              (position[0] + text_width + 10, position[1] + 30))
    draw.rectangle(text_bg, fill=(0, 0, 0, 128))
    
    # Draw the text
    draw.text(position, message, fill=(255, 255, 255), font=font)
    
    return img


@app.route('/save', methods=['POST'])
@login_required
def save_fractal():
    data = request.json
    title = data.get('title', f'Fractal {datetime.utcnow().strftime("%Y%m%d%H%M%S")}')
    fractal_type = data.get('type')
    parameters = data.get('parameters')
    image_data = data.get('image').split(',')[1]  # Remove the data:image/png;base64, part
    
    # Save image to disk
    filename = f'fractal_{uuid.uuid4()}.png'
    filepath = os.path.join('static', 'saved_fractals', filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        f.write(base64.b64decode(image_data))
    
    # Save to database
    fractal = Fractal(
        title=title,
        fractal_type=fractal_type,
        parameters=str(parameters),
        image_path=filepath,
        user_id=current_user.id
    )
    
    db.session.add(fractal)
    db.session.commit()
    
    return jsonify({'success': True, 'id': fractal.id})


@app.route('/gallery')
def gallery():
    if current_user.is_authenticated:
        # Show user's fractals and public fractals
        fractals = Fractal.query.filter(
            (Fractal.public == True) | (Fractal.user_id == current_user.id)
        ).order_by(Fractal.created_at.desc()).all()
    else:
        # Only show public fractals
        fractals = Fractal.query.filter_by(public=True).order_by(Fractal.created_at.desc()).all()
    
    return render_template('gallery.html', fractals=fractals)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        
        flash('Invalid username or password')
    
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return render_template('register.html')
        
        user = User(username=username, email=email)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/profile')
@login_required
def profile():
    user_fractals = Fractal.query.filter_by(user_id=current_user.id).order_by(Fractal.created_at.desc()).all()
    return render_template('profile.html', fractals=user_fractals)


@app.route('/profile/update', methods=['POST'])
@login_required
def update_profile():
    data = request.json
    user = User.query.get(current_user.id)
    
    if not user:
        return jsonify({'success': False, 'message': 'User not found'})
    
    # Update email if provided
    if data.get('email'):
        # Check if email is already taken by another user
        existing_user = User.query.filter(User.email == data['email'], User.id != current_user.id).first()
        if existing_user:
            return jsonify({'success': False, 'message': 'Email already in use'})
        
        user.email = data['email']
    
    # Update password if provided
    if data.get('password'):
        user.set_password(data['password'])
    
    db.session.commit()
    return jsonify({'success': True})


@app.route('/fractal/<int:fractal_id>/delete', methods=['POST'])
@login_required
def delete_fractal(fractal_id):
    fractal = Fractal.query.get(fractal_id)
    
    if not fractal:
        return jsonify({'success': False, 'message': 'Fractal not found'})
    
    # Check if the current user owns the fractal
    if fractal.user_id != current_user.id:
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    # Delete the image file
    if fractal.image_path and os.path.exists(fractal.image_path):
        os.remove(fractal.image_path)
    
    # Delete the database record
    db.session.delete(fractal)
    db.session.commit()
    
    return jsonify({'success': True})


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 