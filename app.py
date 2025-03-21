from flask import Flask, render_template, request, jsonify
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/predict"

# Store results for visualization
history = []


# 📌 Route: Home Page
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        review = request.form["review"]
        response = requests.post(API_URL, json={"text": review})

        if response.status_code == 200:
            result = response.json()
            prediction = result["sentiment"]
            confidence = round(result["confidence"], 2)

            # Store results
            history.append({"review": review, "sentiment": prediction, "confidence": confidence})

            # Save updated history as CSV
            df = pd.DataFrame(history)
            df.to_csv("history.csv", index=False)

    return render_template("index.html", prediction=prediction, history=history)


# 📌 Route: Sentiment Graphs
@app.route("/graph")
def sentiment_graph():
    if not history:
        return "No data yet! Submit a review first."

    df = pd.DataFrame(history)

    # Create graph
    plt.figure(figsize=(5, 3))
    sns.countplot(x="sentiment", data=df, palette="coolwarm")
    plt.title("Sentiment Analysis Results")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")

    # Save graph
    graph_path = os.path.join("static", "sentiment_plot.png")
    plt.savefig(graph_path)

    return render_template("graph.html", graph_path=graph_path)


if __name__ == "__main__":
    app.run(debug=True)
