
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained('app_model/epoch3_lrate2e_b48_s15000v3000')
model = AutoModelForSeq2SeqLM.from_pretrained('app_model/epoch3_lrate2e_b48_s15000v3000')
max_input = 1024

graphs_app = Flask(__name__)
graphs_app.secret_key = 'graph_flask_app'


@graphs_app.route("/")
def hello_world():
    return render_template("home.html")

def get_count(text):
    tokens = tokenizer.encode(text)
    return len(tokens)
@graphs_app.route("/", methods=['POST'])
def give_summary():
    model_input = ""

    if request.method == 'POST':
        model_input = request.form['input_text']
    total_tokens = get_count(model_input)
    summary = summarize(model_input)
    print(total_tokens)
    return summary


def summarize(summary):
    # Consistent preprocessing
    inputs = tokenizer(
        summary,
        truncation=True,
        padding="max_length",
        max_length=max_input,
        return_tensors='pt'
    )

    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=150,
        min_length=20,
        length_penalty=1.0,
        num_beams=4,
        early_stopping=True
    )

    # Decode and return summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

if __name__ == "__main__":
    graphs_app.run(debug=True)
