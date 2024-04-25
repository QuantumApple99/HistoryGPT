from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('/html/index.html')


@app.route("/process_data", methods=['POST'])
def process_data():
    data = request.get_json()
    prompt = data['prompt']

    os.environ['OPENAI_API_KEY'] = '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'

    client = OpenAI()

    fine_tuned_model = 'ft:gpt-3.5-turbo-0125:personal:historygpt:9HnX5Uv9'
    prompt = prompt
    messages1 = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model=fine_tuned_model,
        messages=messages1
    )

    answer = response.choices[0].message.content

    return jsonify({"message": answer})


if __name__ == '__main__':
    app.run(debug=True)
