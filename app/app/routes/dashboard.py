from flask import Blueprint, render_template

dashboard = Blueprint('dashboard', __name__,url_prefix="/dashboard")

@dashboard.route('/')
def dashboard_page():
    return render_template('dashboard.html')

@dashboard.route("/nerdy/")
def nerdy_dashboard():
    return render_template("dashboard_nerdy.html")