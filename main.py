from flask import Flask
from py2neo import Graph

app = Flask(__name__)
graph = Graph("bolt://localhost:7687", auth=("TeaNLP", "teanlp"))


@app.route("/")
def hello():
    graph.run("Match () Return 1 Limit 1")
    return 'Hello, Amparo!'

