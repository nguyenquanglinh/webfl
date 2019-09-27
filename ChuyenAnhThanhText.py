from flask import Flask, request

from digitImg import model_Linear_svc

a = model_Linear_svc()

app = Flask(__name__)


@app.route('/image', methods=['POST'])
def post_image():
    d = request.files.get('file')
    d.save("anhChuaXuLy.png")
    s = a.convert("anhChuaXuLy.png")

    print("du lieu tra ve la: "+s)
    return s

@app.route('/')
def index():
    return "hello from service"


app.run(host='127.0.0.1', port = 8000, debug=True)
