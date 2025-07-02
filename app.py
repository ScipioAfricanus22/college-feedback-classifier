from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
    # Load the classified feedback CSV
    df = pd.read_csv("classified_feedback.csv")
    # Convert to list of dicts so we can show in HTML
    feedback_list = df.to_dict(orient='records')
    return render_template("index.html", feedbacks=feedback_list)

if __name__ == "__main__":
    app.run()
