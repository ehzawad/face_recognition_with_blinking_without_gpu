from flask import Flask, Response

app = Flask(__name__)

@app.route("/")
def index():
    resp = Response("Hello")
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="192.168.10.89", port=5000)