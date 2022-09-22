import os, json, random, io, base64

import matplotlib

from datetime import datetime

from flask import Flask, render_template, redirect, url_for, request, flash
from flask_pymongo import PyMongo
from flask_login import current_user, LoginManager, login_required, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

from ztfnuclear.sample import NuclearSample, Transient

from ztfnuclear.database import SampleInfo
from ztfnuclear.utils import is_ztf_name
from ztfnuclear.forms import LoginForm, UserForm

matplotlib.pyplot.switch_backend("Agg")

# Class-based application configuration
class ConfigClass(object):
    """Flask application config"""

    # Flask settings
    SECRET_KEY = "aotvm8vFJIELTORETU8VFJDK453JKjfdkoo"

    # Flask-MongoEngine settings
    MONGO_URI = "mongodb://localhost:27017/ztfnuclear_viewer"

    USER_APP_NAME = "ZTFNuclear Sample"
    USER_ENABLE_EMAIL = False
    USER_ENABLE_USERNAME = True
    USER_REQUIRE_RETYPE_PASSWORD = False


app = Flask(__name__)
app.config.from_object(__name__ + ".ConfigClass")
app.jinja_env.auto_reload = True

mongo = PyMongo(app)

login_manager = LoginManager(app)
login_manager.init_app(app)
login_manager.login_view = "login"

if __name__ == "__main__":
    app.run(debug=True)


class User:
    def __init__(self, username, password_hash):
        self.username = username
        self.password_hash = password_hash

    @staticmethod
    def is_authenticated():
        return True

    @staticmethod
    def is_active():
        return True

    @staticmethod
    def is_anonymous():
        return False

    def get_id(self):
        return self.username

    @staticmethod
    def check_password(password_hash, password):
        return check_password_hash(password_hash, password)

    @login_manager.user_loader
    def load_user(username):
        u = mongo.db.ztfnuclear_viewer.find_one({"Name": username})
        print("-----")
        print(u)
        print("------")
        if not u:
            return None
        return User(username=u["Name"])

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("index"))
        form = LoginForm()
        if form.validate_on_submit():
            user = mongo.db.ztfnuclear_viewer.find_one({"Name": form.name.data})
            if user and User.check_password(user["Password"], form.password.data):
                user_obj = User(username=user["Name"])
                login_user(user_obj)
                next_page = request.args.get("next")
                if not next_page or url_parse(next_page).netloc != "":
                    next_page = url_for("index")
                return redirect(next_page)
            else:
                flash("Invalid username or password")
        return render_template("login.html", title="Sign In", form=form)

    @app.route("/logout")
    def logout():
        logout_user()
        return redirect(url_for("login"))


sample_ztfids = NuclearSample().ztfids
info_db = SampleInfo()
flaring_ztfids = info_db.read()["flaring"]["ztfids"]


@app.route("/")
def home():
    """
    This is the default page
    """
    return render_template(
        "home.html",
    )


@app.route("/transients/<string:ztfid>")
def transient_page(ztfid):
    """
    Show the transient page
    """
    if not is_ztf_name(ztfid):
        return render_template("bad_query.html", bad_id=ztfid, rej_reason="not_valid")

    if ztfid not in sample_ztfids:
        return render_template("bad_query.html", bad_id=ztfid)

    t = Transient(ztfid=ztfid)

    t.plot(plot_png=True, wide=True)

    base_dir = os.path.join(str(os.getenv("ZTFDATA")), "nuclear_sample")
    plot_file = os.path.join(base_dir, "plots", "lightcurves", "flux", f"{ztfid}.png")

    plot_data = open(plot_file, "rb")

    base64_string = base64.b64encode(plot_data.read()).decode("ascii")

    s = NuclearSample()

    previous_transient = s.previous_transient(ztfid)
    next_transient = s.next_transient(ztfid)

    return render_template(
        "transient.html",
        transient=t,
        lcplot=base64_string,
        previous_transient=previous_transient,
        next_transient=next_transient,
        flaring=False,
    )


@app.route("/flaring/<string:ztfid>")
def flaring_page(ztfid):
    """
    Show the page of a flaring transient
    """
    if not is_ztf_name(ztfid):
        return render_template("bad_query.html", bad_id=ztfid, rej_reason="not_valid")

    if ztfid not in flaring_ztfids:
        return render_template("bad_query.html", bad_id=ztfid)

    t = Transient(ztfid=ztfid)

    t.plot(plot_png=True, wide=True)

    base_dir = os.path.join(str(os.getenv("ZTFDATA")), "nuclear_sample")
    plot_file = os.path.join(base_dir, "plots", "lightcurves", "flux", f"{ztfid}.png")

    plot_data = open(plot_file, "rb")

    base64_string = base64.b64encode(plot_data.read()).decode("ascii")

    s = NuclearSample()

    previous_transient = s.previous_transient(ztfid, flaring=True)
    next_transient = s.next_transient(ztfid, flaring=True)

    return render_template(
        "transient.html",
        transient=t,
        lcplot=base64_string,
        previous_transient=previous_transient,
        next_transient=next_transient,
        flaring=True,
    )


@app.route("/rate/<string:ztfid>", methods=["GET", "POST"])
def rate_transient(ztfid):
    """ """
    t = Transient(ztfid)

    input_key = list(request.form.keys())[0]

    if request.method == "POST":

        if input_key == "rating":
            raw_value = request.form["rating"]
            split = raw_value.split("&")
            origin = split[0]
            rating = int(split[1].split("=")[1])

            t.set_rating(rating)

        elif input_key == "pinned":
            origin = request.form["pinned"]
            # do stuff

        return redirect(origin)


@app.route("/sample")
def transient_list():
    """
    Show a list of all the transients
    """
    s = NuclearSample()
    transients = s.get_transients(n=100)
    return render_template("transient_list.html", transients=transients)


@app.route("/flaringsample")
def flaring_transient_list():
    """
    Show a list of all the transients
    """
    s = NuclearSample()

    flaring_transients = s.get_flaring_transients()
    return render_template("transient_list.html", transients=flaring_transients)


@app.route("/random")
def transient_random():
    """
    Show a random transient
    """
    s = NuclearSample()
    random_ztfid = random.choice(s.ztfids)
    # return transient_page(ztfid=random_ztfid)
    return redirect(url_for("transient_page", ztfid=random_ztfid))


@app.route("/flaringrandom")
def flaring_transient_random():
    """
    Show a random transient
    """
    random_flaring_ztfid = random.choice(flaring_ztfids)
    # return transient_page(ztfid=random_ztfid)
    return redirect(url_for("flaring_page", ztfid=random_flaring_ztfid))


@app.route("/search", methods=["GET", "POST"])
def search():
    """
    Search for a transient
    """
    if request.method == "POST":
        ztfid = request.form["name"]
        if not is_ztf_name(ztfid):
            return render_template(
                "bad_query.html", bad_id=ztfid, rej_reason="not_valid"
            )

        if ztfid not in sample_ztfids:
            return render_template("bad_query.html", bad_id=ztfid)

        return redirect(url_for(f"transient_page", ztfid=ztfid))
    else:
        return redirect(url_for("home"))


@app.route("/register", methods=["GET", "POST"])
def add_user():
    """ """
    form = UserForm()
    if form.validate_on_submit():  # If you submit, this happens

        # query the Users-Database that have the inout user email and return the first one
        # This should return None if it is indeed unique
        # user = User.query.filter_by(username=form.name.data).first()
        user = mongo.db.ztfnuclear_viewer.find_one({"Name": form.name.data})
        print(form.name.data)
        print(user)
        if user is None:
            # create a new db user entry
            hashed_pwd = generate_password_hash(form.password_hash.data, "sha256")

            user = User(
                username=form.name.data,
                password_hash=hashed_pwd,
            )
            mongo.db.ztfnuclear_viewer.update_one(
                {"Name": form.name.data},
                {"$set": {"Name": form.name.data, "Password": hashed_pwd}},
                upsert=False,
            )

            # add it to the actual db
            # mongo.db.ztfnuclear_viewer.add(user)
            # and commit it
            # mongo.db.ztfnuclear_viewer.commit()
            flash("User added successfully", category="success")
        else:
            flash(
                "Username already used. User not added to the database",
                category="error",
            )

        # Clearing this out
        name = form.name.data
        form.name.data = ""
        form.password_hash.data = ""

    return render_template("register.html", form=form)


if __name__ == "__main__":
    app.run(host="127.0.0.1")
