from flask import Flask, render_template, request
import pickle

tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")



@app.route("/predict", methods=["GET","POST"])
def predict():
    text=""
    if request.method == "POST" :
        text = request.form.get("email-content")
    tokenized_text = tokenizer.transform([text])
    predictions = model.predict(tokenized_text)
    pred = 1 if predictions == 1 else -1

    # return "Hello, World! Updated(since debug=True)"
    return render_template("index.html", predictions=pred, text=text)





# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)