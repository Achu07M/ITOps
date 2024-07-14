from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, login_required, logout_user, current_user
from .models import User
from . import db, bcrypt, login_manager
from .forms import RegistrationForm, LoginForm
from .network_maintenance import return_prediction
import json
import os
from flask import jsonify

main = Blueprint('main', __name__)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@main.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('main.login'))
    return render_template('register.html', form=form)

@main.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('main.home'))
        else:
            flash('Login unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', form=form)

@main.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.login'))

@main.route('/')
@login_required
def home():
    return render_template('home.html')

@main.route('/admin', methods=['GET', 'POST'])
@login_required
def admin():
    if current_user.role != 'Admin':
        flash('You do not have access to this page.', 'danger')
        return redirect(url_for('main.home'))
    users = User.query.all()
    if request.method == 'POST':
        user_id = request.form['user_id']
        role = request.form['role']
        user = User.query.get(user_id)
        if user:
            user.role = role
            db.session.commit()
            flash('User role updated successfully.', 'success')
        else:
            flash('User not found.', 'danger')
    return render_template('admin.html', users=users, roles=["IT Asset Manager", "Network Manager", "Admin", "Other Role"])


@main.route('/asset_management')
@login_required
def asset_management():
    return render_template('asset_management.html')

@main.route('/hardware_assets')
@login_required
def hardware_assets():
    return render_template('hardware_assets.html')

@main.route('/software_assets')
@login_required
def software_assets():
    return render_template('software_assets.html')

@main.route('/network_management')
@login_required
def network_management():
    return render_template('network_management.html')

@main.route('/network_predict', methods=['POST'])
@login_required
def network_predict():
    prediction = return_prediction('../ITAM_project/models')
    predictions_file = "prediction.json"
    if os.path.exists(predictions_file):
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
    else:
        predictions = []

    # Append the new prediction to the list
    predictions.append(prediction)

    # Write the updated predictions list back to the JSON file
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f)
        
    return jsonify(prediction)

@main.route('/network_logs')
def network_logs():
    # Read logs from JSON file
    with open('prediction.json','r') as f:
        logs = json.load(f)
    return render_template('network_logs.html', logs=logs)