from flask import Response, render_template

from algorithm import gen_frames
from project import app, socket


@app.route("/")
def index():
    return render_template("index.html")


@socket.on("connect")
def connect():
    print("Client connected")


@socket.on("disconnect")
def disconnect():
    print("Client disconnected")


@app.route("/video_feed/")
def video_feed(source=0):
    return Response(
        gen_frames(source),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(debug=True)
