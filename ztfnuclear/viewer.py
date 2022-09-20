import os, json, random, io, base64

import matplotlib

from flask import Flask, render_template, redirect, url_for, request
from flask_login import current_user, LoginManager
from ztfnuclear.sample import NuclearSample, Transient
from ztfnuclear.database import SampleInfo
from ztfnuclear.utils import is_ztf_name
from ztfnuclear.forms import LoginForm

app = Flask(__name__)
app.config["SECRET_KEY"] = "aotvm8vFJIELTORETU8VFJDK453JKjfdkoo"
app.jinja_env.auto_reload = True


# Flask Login Stuff
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"  # where to point to log if needed

sample_ids = NuclearSample().ztfids


@login_manager.user_loader
def load_user(user_id):
    """ """
    try:
        return Users.query.get(int(user_id))
    except:
        return None


class Users:

    id: int = 15
    username: str = "simeon"


@app.route("/")
def home():
    """
    This is the default page
    """
    return render_template(
        "home.html",
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    """ """
    form = LoginForm()
    # entry the if went hit submit from login.html
    if form.validate_on_submit():
        # grab the first user given the inputform username
        user = Users.query.filter_by(username=form.username.data).first()
        if user:
            # Check the hash
            if check_password_hash(user.password_hash, form.password.data):
                login_user(user)  # Flask login
                flash("Login Successfull", category="success")
                return redirect(url_for("dashboard"))
            else:
                flash("Wrong Password - Try again", category="error")
        else:  # no user
            flash("That user doesn't exist - Try again", category="warning")

    return render_template("login.html", form=form)


@app.route("/transients/<string:ztfid>")
def transient_page(ztfid):
    """
    Show the transient page
    """
    matplotlib.pyplot.switch_backend("Agg")

    if not is_ztf_name(ztfid):
        return render_template("bad_query.html", bad_id=ztfid, rej_reason="not_valid")

    if ztfid not in sample_ids:
        return render_template("bad_query.html", bad_id=ztfid)

    t = Transient(ztfid=ztfid)
    t.plot(plot_png=True, wide=True)
    base_dir = os.path.join(str(os.getenv("ZTFDATA")), "nuclear_sample")
    plot_file = os.path.join(
        base_dir, "plots", "lightcurves", "flux", f"{ztfid}_bl.png"
    )
    if os.path.isfile(plot_file):
        plot_data = open(plot_file, "rb")
    else:
        t.plot(plot_png=True, baseline_correction=False, wide=True)
        plot_file = os.path.join(
            base_dir, "plots", "lightcurves", "flux", f"{ztfid}.png"
        )
        plot_data = open(plot_file, "rb")
    base64_string = base64.b64encode(plot_data.read()).decode("ascii")

    return render_template("transient.html", transient=t, lcplot=base64_string)


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
    return redirect(url_for(f"transient_page", ztfid=random_ztfid))


@app.route("/flaringrandom")
def flaring_transient_random():
    """
    Show a random transient
    """
    info_db = SampleInfo()
    flaring_ztfids = info_db.read()["flaring"]["ztfids"]
    random_flaring_ztfid = random.choice(flaring_ztfids)
    # return transient_page(ztfid=random_ztfid)
    return redirect(url_for(f"transient_page", ztfid=random_flaring_ztfid))


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

        if ztfid not in sample_ids:
            return render_template("bad_query.html", bad_id=ztfid)

        return redirect(url_for(f"transient_page", ztfid=ztfid))
    else:
        return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(host="127.0.0.1")
