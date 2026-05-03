from flask import Flask
import os

def create_app(config_name=None):
    app = Flask(__name__, instance_relative_config=True)
    
    # Load environment-based config
    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "development")
    
    if config_name == "development":
        from app.config import DevelopmentConfig
        app.config.from_object(DevelopmentConfig)
    elif config_name == "testing":
        from config import TestingConfig
        app.config.from_object(TestingConfig)
    elif config_name == "production":
        from config import ProductionConfig
        app.config.from_object(ProductionConfig)
    else:
        from config import Config
        app.config.from_object(Config)
    
    # Register blueprints
    from app.routes.main import main
    from app.routes.api import api
    from app.routes.dashboard import dashboard
    from app.routes.bogosort import bogosort_demo
    
    app.register_blueprint(main)
    app.register_blueprint(api, url_prefix='/api')
    app.register_blueprint(dashboard, url_prefix='/dashboard')
    app.register_blueprint(bogosort_demo, url_prefix='/bogosort') 
    
    # Initialize DB
    #from app.db.queries import init_db
    #init_db(app)
    
    return app