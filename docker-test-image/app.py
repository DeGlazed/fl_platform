from flask import Flask, render_template_string
import requests

app = Flask(__name__)

@app.route('/')
def index():
    # Fetching data from JSONPlaceholder API
    url = "https://jsonplaceholder.typicode.com/posts/1"
    response = requests.get(url)
    data = response.json()

    # Extract the title and body from the response
    title = data.get('title')
    body = data.get('body')

    return render_template_string('''
        <html>
            <head><title>API Data</title></head>
            <body>
                <h1>Post from JSONPlaceholder</h1>
                <h2>Title:</h2>
                <p>{{ title }}</p>
                <h2>Body:</h2>
                <p>{{ body }}</p>
            </body>
        </html>
    ''', title=title, body=body)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)