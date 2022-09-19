import os, json, random, io, base64

import matplotlib

from flask import Flask, render_template
from flask_login import current_user, LoginManager
from sample import NuclearSample, Transient

from forms import LoginForm

app = Flask(__name__)
app.config["SECRET_KEY"] = "aotvm8vFJIELTORETU8VFJDK453JKjfdkoo"
app.jinja_env.auto_reload = True


# Flask Login Stuff
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"  # where to point to log if needed


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


# @app.route("/transients/<string:ztfid>")
# def show_post(ztfid: str):
#     # show the post with the given id, the id is an integer
#     t = Transient(ztfid=ztfid)
#     json_string = json.dumps(t.meta)
#     return_dict = json.loads(json_string)
#     print(return_dict)
#     return return_dict


@app.route("/transients/<string:ztfid>")
def target_page(ztfid):
    """ """
    matplotlib.pyplot.switch_backend("Agg")
    # DB
    t = Transient(ztfid=ztfid)
    # print(t.meta)
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
    # output = io.BytesIO()
    # fig.savefig("test.png")
    # _ = fig.savefig(buffer, format="png", dpi=250)
    # lcplot = base64.b64encode(buffer.getbuffer()).decode("ascii")
    return render_template("transient.html", transient=t, lcplot=base64_string)


@app.route("/random")
def transient_random():
    """ """
    s = NuclearSample()
    random_ztfid = random.choice(s.ztfids)
    return target_page(ztfid=random_ztfid)
