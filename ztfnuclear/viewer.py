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

from ztfnuclear.database import SampleInfo, MetadataDB
from ztfnuclear.utils import is_ztf_name, is_tns_name


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
    This is the landing page
    """
    s = NuclearSample()
    if current_user.is_authenticated:

        ratings = s.get_ratings(username=current_user.username)
        ratings_ztfids = list(ratings.keys())
        flaring_rated = []
        for ztfid in ratings_ztfids:
            if ztfid in flaring_ztfids:
                flaring_rated.append(ztfid)
        user_rate_percentage = len(flaring_rated) / len(flaring_ztfids) * 100
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


@app.route("/transient/<string:ztfid>")
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

    base_dir = os.path.join(str(os.getenv("ZTFDATA")), "nuclear_sample")

    plot_file = os.path.join(base_dir, "plots", "lightcurves", "flux", f"{ztfid}.png")

    if not os.path.isfile(plot_file):
        axlims = t.plot(plot_png=True, wide=True)

    plot_data = open(plot_file, "rb")

    base64_string = base64.b64encode(plot_data.read()).decode("ascii")

    plot_file_irsa = os.path.join(
        base_dir, "plots", "lightcurves_irsa", "flux", f"{ztfid}.png"
    )

    if not os.path.isfile(plot_file_irsa):
        axlims = t.plot(plot_png=True, wide=True)
        t.plot_irsa(plot_png=True, wide=True, axlims=axlims)

    if os.path.isfile(plot_file_irsa):

        plot_data_irsa = open(plot_file_irsa, "rb")
        base64_string_irsa = base64.b64encode(plot_data_irsa.read()).decode("ascii")
        plot_irsa = True

    else:

        plot_irsa = False
        base64_string_irsa = None

    plot_file_tde_fit = os.path.join(
        base_dir, "plots", "lightcurves", "tde_fit", f"{ztfid}.png"
    )

    success = t.plot_tde()

    if success and os.path.isfile(plot_file_tde_fit):
        plot_data_tde_fit = open(plot_file_tde_fit, "rb")
        base64_string_tde_fit = base64.b64encode(plot_data_tde_fit.read()).decode(
            "ascii"
        )
        plot_tde_fit = True
        tde_fitres = t.meta["tde_fit_exp"]
    else:
        plot_tde_fit = False
        base64_string_tde_fit = None
        tde_fitres = None

    s = NuclearSample()

    previous_transient = s.previous_transient(ztfid)
    next_transient = s.next_transient(ztfid)

    comments = t.get_comments_generator()
    comment_count = t.get_comment_count()

    return render_template(
        "transient.html",
        transient=t,
        lcplot=base64_string,
        plot_irsa=plot_irsa,
        lcplot_irsa=base64_string_irsa,
        plot_tde_fit=plot_tde_fit,
        lcplot_tde_fit=base64_string_tde_fit,
        tde_fitres=tde_fitres,
        previous_transient=previous_transient,
        next_transient=next_transient,
        flaring=False,
        comments=comments,
        comment_count=comment_count,
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

    base_dir = os.path.join(str(os.getenv("ZTFDATA")), "nuclear_sample")
    plot_file = os.path.join(base_dir, "plots", "lightcurves", "flux", f"{ztfid}.png")

    if not os.path.isfile(plot_file):
        axlims = t.plot(plot_png=True, wide=True)

    plot_data = open(plot_file, "rb")

    base64_string = base64.b64encode(plot_data.read()).decode("ascii")

    plot_file_irsa = os.path.join(
        base_dir, "plots", "lightcurves_irsa", "flux", f"{ztfid}.png"
    )

    if not os.path.isfile(plot_file_irsa):
        axlims = t.plot(plot_png=True, wide=True)
        t.plot_irsa(plot_png=True, wide=True, axlims=axlims)

    if os.path.isfile(plot_file_irsa):

        plot_data_irsa = open(plot_file_irsa, "rb")
        base64_string_irsa = base64.b64encode(plot_data_irsa.read()).decode("ascii")
        plot_irsa = True

    else:

        plot_irsa = False
        base64_string_irsa = None

    plot_file_tde_fit = os.path.join(
        base_dir, "plots", "lightcurves", "tde_fit", f"{ztfid}.png"
    )

    # if not os.path.isfile(plot_file_tde_fit):
    t.plot_tde()

    if os.path.isfile(plot_file_tde_fit):
        plot_data_tde_fit = open(plot_file_tde_fit, "rb")
        base64_string_tde_fit = base64.b64encode(plot_data_tde_fit.read()).decode(
            "ascii"
        )
        plot_tde_fit = True
        tde_fitres = t.meta["tde_fit_exp"]
    else:
        plot_tde_fit = False
        base64_string_tde_fit = None
        tde_fitres = None

    s = NuclearSample()

    previous_transient = s.previous_transient(ztfid, flaring=True)
    next_transient = s.next_transient(ztfid, flaring=True)

    comments = t.get_comments_generator()
    comment_count = t.get_comment_count()

    return render_template(
        "transient.html",
        transient=t,
        lcplot=base64_string,
        plot_irsa=plot_irsa,
        lcplot_irsa=base64_string_irsa,
        plot_tde_fit=plot_tde_fit,
        lcplot_tde_fit=base64_string_tde_fit,
        tde_fitres=tde_fitres,
        previous_transient=previous_transient,
        next_transient=next_transient,
        flaring=True,
        comments=comments,
        comment_count=comment_count,
    )


@app.route("/rate/<string:ztfid>", methods=["GET", "POST"])
@login_required
def rate_transient(ztfid):
    """
    To deal with rating form data
    """
    t = Transient(ztfid)

    input_key = list(request.form.keys())[0]

    if request.method == "POST":

        if input_key == "rating":
            raw_value = request.form["rating"]

            split = raw_value.split("&")
            origin = split[0]
            rating = int(split[1].split("=")[1])
            username = str(split[2].split("=")[1])

            t.set_rating(rating, username=username)

        if "transient" in origin:
            return redirect(url_for("transient_random"))
        elif "flaring" in origin:
            return redirect(url_for("flaring_transient_random"))
        else:
            return redirect(origin)


@app.route("/comment/<string:ztfid>", methods=["GET", "POST"])
@login_required
def comment_on_transient(ztfid):
    """
    To deal with comment form data
    """
    t = Transient(ztfid)

    if request.method == "POST":

        for input_key in list(request.form.keys()):

            if input_key == "delete":
                raw = request.form["delete"]
                origin = raw.split("&")[0].split("=")[1]
                timestamp = raw.split("&")[1].split("=")[1]
                t.delete_comment(timestamp=timestamp)

                return redirect(origin)

            if input_key == "comment":
                comment_text = request.form["comment"]

            if input_key == "origin":
                origin = request.form["origin"]

    t.add_comment(username=current_user.username, comment=comment_text)

    return redirect(origin)


@app.route("/sample")
@login_required
def transient_list():
    """
    Show a list of all the transients
    """
    s = NuclearSample()
    transients = s.get_transients_pickled()

    return render_template("transient_list.html", transients=transients)


@app.route("/flaringsample")
@login_required
def flaring_transient_list():
    """
    Show a list of all the transients
    """
    s = NuclearSample()

    flaring_transients = s.get_transients_pickled(flaring_only=True)
    return render_template("transient_list.html", transients=flaring_transients)


@app.route("/interesting")
@login_required
def list_interesting():
    """
    List all the transients selected as 'interesting'
    """
    s = NuclearSample()
    interesting_ztfids = list(
        s.get_ratings(username=current_user.username, select="interesting").keys()
    )

    interesting_transients = s.get_transients(ztfids=interesting_ztfids)

    return render_template("transient_list.html", transients=interesting_transients)


@app.route("/maybe")
@login_required
def list_maybe():
    """
    List all the transients selected as 'interesting'
    """
    s = NuclearSample()
    maybe_interesting_ztfids = list(
        s.get_ratings(username=current_user.username, select="maybe").keys()
    )

    maybe_interesting_transients = s.get_transients(ztfids=maybe_interesting_ztfids)

    return render_template(
        "transient_list.html", transients=maybe_interesting_transients
    )


@app.route("/gold")
@login_required
def list_golden():
    """
    List all the transients selected as 'interesting' by at least 2 users
    """
    s = NuclearSample()
    golden_sample_ztfids = []

    # get all transients rated as interesting by at least one person
    interesting_ztfids = s.get_ratings(select="interesting")

    # at least two persons must have rated the transient as interesting
    for k, v in interesting_ztfids.items():
        if len(v) > 1:
            rated_interesting = 0
            for entry in v.keys():
                if v[entry] == 3:
                    rated_interesting += 1

            if rated_interesting >= 2:
                golden_sample_ztfids.append(k)

    golden_sample_transients = s.get_transients(ztfids=golden_sample_ztfids)

    return render_template("transient_list.html", transients=golden_sample_transients)


@app.route("/random")
@login_required
def transient_random():
    """
    Show a random transient that has NOT been rated yet
    """
    s = NuclearSample()

    rated_ztfids = set(list(s.get_ratings(username=current_user.username).keys()))

    all_ztfids = set(s.ztfids)

    non_rated_ztfids = list(all_ztfids.difference(rated_ztfids))

    if len(non_rated_ztfids) > 0:
        random_ztfid = random.choice(non_rated_ztfids)

    else:
        random_ztfid = random.choice(all_ztfids)

    return redirect(url_for("transient_page", ztfid=random_ztfid))


@app.route("/flaringrandom")
@login_required
def flaring_transient_random():
    """
    Show a random IR flaring transient that has NOT been rated yet
    """
    s = NuclearSample()

    rated_ztfids = set(list(s.get_ratings(username=current_user.username).keys()))

    non_rated_ztfids = list(set(flaring_ztfids).difference(rated_ztfids))

    if len(non_rated_ztfids) > 0:
        random_ztfid = random.choice(non_rated_ztfids)

    else:
        random_ztfid = random.choice(flaring_ztfids)

    return redirect(url_for("flaring_page", ztfid=random_ztfid))


@app.route("/search", methods=["GET", "POST"])
@login_required
def search():
    """
    Search for a transient
    """
    meta = MetadataDB()
    if request.method == "POST":
        object_id = request.form["name"]
        if not is_ztf_name(object_id) and not is_tns_name(object_id):
            return render_template(
                "bad_query.html", bad_id=object_id, rej_reason="not_valid"
            )

        if is_ztf_name(object_id) and object_id not in sample_ztfids:
            return render_template("bad_query.html", bad_id=object_id)

        if is_tns_name(object_id) and object_id not in info_db.read()["tns_names"]:
            return render_template("bad_query.html", bad_id=object_id)

        if is_tns_name(object_id):
            ztfid = meta.find_by_tns(tns_name=object_id)["_id"]

        if is_ztf_name(object_id):
            ztfid = object_id

        return redirect(url_for(f"transient_page", ztfid=ztfid))
    else:
        return redirect(url_for("home"))


@app.route("/register", methods=["GET", "POST"])
def add_user():
    """
    Add a user, hash his/her password and write that to the UserDB
    """
    form = UserForm()
    if form.validate_on_submit():

        user = mongo.db.ztfnuclear_viewer.find_one({"Name": form.name.data})

        if user is None:
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

            flash("User added successfully", category="success")
        else:
            flash(
                "Username already used. User not added to the database",
                category="error",
            )

        name = form.name.data
        form.name.data = ""
        form.password_hash.data = ""

    return render_template("register.html", form=form)


if __name__ == "__main__":
    app.run(host="127.0.0.1")
