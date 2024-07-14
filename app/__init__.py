from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from chatbot.chatbot import chatbot_bp

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = 'main.login'
bcrypt = Bcrypt()

def create_app():
    app = Flask(__name__,static_folder='static')
    app.config['SECRET_KEY'] = '371629ac10b92dc16b9f907c50cc77f0'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)

    from .routes import main
    app.register_blueprint(main)
    app.register_blueprint(chatbot_bp, url_prefix='/chatbot')

    with app.app_context():
        db.create_all()


    return app

