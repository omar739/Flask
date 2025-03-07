from flask import Flask, render_template, request
app = Flask(__name__, template_folder="templates")
@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "GET":
        return render_template("filee.html", imagee=None)
    elif request.method == "POST":
        image_file = request.files.get('image')
        target = request.form.get('Target')
        if image_file:
            image_path = "D:/programming_Vscode/static/uploads/savedimage.jpeg"
            image_file.save(image_path)
            return render_template("filee.html", imagee=image_path)
    return render_template("filee.html", imagee=None)
if __name__ == "__main__":
    app.run(debug=True, port=8090)
