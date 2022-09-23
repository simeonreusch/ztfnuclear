import os, json, random, io, base64

import matplotlib

from datetime import datetime

from flask_wtf import FlaskForm
from wtforms import (
    StringField,
    SubmitField,
    PasswordField,
    BooleanField,
    ValidationError,
)
from wtforms.validators import DataRequired, EqualTo, Length

from flask import Flask, render_template, redirect, url_for, request, flash
from flask_mongoengine import MongoEngine, Document
from flask_login import (
    current_user,
    LoginManager,
    logout_user,
    login_required,
    UserMixin,
    login_user,
)
from werkzeug.security import generate_password_hash, check_password_hash

from werkzeug.urls import url_parse

from ztfnuclear.sample import NuclearSample, Transient

from ztfnuclear.database import SampleInfo
from ztfnuclear.utils import is_ztf_name


matplotlib.pyplot.switch_backend("Agg")


class LoginForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    remember_me = BooleanField("Remember Me")
    submit = SubmitField("Sign In")


class RegistrationForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    password2 = PasswordField(
        "Repeat Password", validators=[DataRequired(), EqualTo("password")]
    )
    submit = SubmitField("Register")

    def validate_username(self, username):
        user = User.objects(username=username.data).first()
        if user is not None:
            raise ValidationError("Please use a different username.")


app = Flask(__name__)
app.config["MONGODB_SETTINGS"] = [
    {
        "db": "ztfnuclear_viewer",
        "host": "localhost",
        "port": 27017,
        "alias": "default",
    }
]
app.config["SECRET_KEY"] = "aotvm8vFJIELTORETU8VFJDK453JKjfdkoo"
app.jinja_env.auto_reload = True


login = LoginManager(app)
login.login_view = "login"

db = MongoEngine(app)

if __name__ == "__main__":
    app.run(debug=True)


@login.user_loader
def load_user(id):
    return User.objects.get(id=id)


class User(UserMixin, db.Document):
    meta = {"collection": "users"}
    username = db.StringField(default=True, unique=True)
    password_hash = db.StringField(default=True)
    timestamp = db.DateTimeField(default=datetime.now())

    def __repr__(self):
        return "<User {}>".format(self.username)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


sample_ztfids = NuclearSample().ztfids
info_db = SampleInfo()
flaring_ztfids = info_db.read()["flaring"]["ztfids"]


@app.route("/")
def home():
    """
    This is the default page
    """
    s = NuclearSample()
    if current_user.is_authenticated:
        ratings = s.get_ratings(username=current_user.username)
        user_rate_percentage = len(ratings) / len(flaring_ztfids) * 100
    else:
        user_rate_percentage = None

    return render_template(
        "home.html",
        user_rate_percentage=user_rate_percentage,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.objects(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash("Invalid username or password")
            return redirect(url_for("login"))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get("next")
        if not next_page or url_parse(next_page).netloc != "":
            next_page = url_for("home")
        return redirect(next_page)
    return render_template("login.html", title="Sign In", form=form)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data)
        user.set_password(form.password.data)
        user.save()
        flash("Congratulations, you have now registered!")
        return redirect(url_for("login"))
    return render_template("register.html", title="Register", form=form)


@app.route("/transients/<string:ztfid>")
@login_required
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
@login_required
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
@login_required
def rate_transient(ztfid):
    """ """
    t = Transient(ztfid)

    input_key = list(request.form.keys())[0]

    if request.method == "POST":

        if input_key == "rating":
            raw_value = request.form["rating"]
            print(raw_value)
            split = raw_value.split("&")
            origin = split[0]
            rating = int(split[1].split("=")[1])
            username = str(split[2].split("=")[1])

            t.set_rating(rating, username=username)

        elif input_key == "pinned":
            origin = request.form["pinned"]
            # do stuff

        if "transients" in origin:
            return redirect(url_for("transient_random"))
        elif "flaring" in origin:
            return redirect(url_for("flaring_transient_random"))
        else:
            return redirect(origin)


@app.route("/sample")
@login_required
def transient_list():
    """
    Show a list of all the transients
    """
    s = NuclearSample()
    transients = s.get_transients(n=100)
    return render_template("transient_list.html", transients=transients)


@app.route("/flaringsample")
@login_required
def flaring_transient_list():
    """
    Show a list of all the transients
    """
    s = NuclearSample()

    flaring_transients = s.get_flaring_transients()
    return render_template("transient_list.html", transients=flaring_transients)


@app.route("/random")
@login_required
def transient_random():
    """
    Show a random transient
    """
    s = NuclearSample()
    random_ztfid = random.choice(s.ztfids)
    # return transient_page(ztfid=random_ztfid)
    return redirect(url_for("transient_page", ztfid=random_ztfid))


@app.route("/flaringrandom")
@login_required
def flaring_transient_random():
    """
    Show a random transient
    """
    random_flaring_ztfid = random.choice(flaring_ztfids)
    # return transient_page(ztfid=random_ztfid)
    return redirect(url_for("flaring_page", ztfid=random_flaring_ztfid))


@app.route("/search", methods=["GET", "POST"])
@login_required
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
