import json

from py2neo import Graph


class Neo4JUtils:

    instance = None

    class __Neo4JUtils:
        def __init__(self):
            # This forces download if not done
            self._graph = Graph("bolt://localhost:7687", auth=("TeaNLP", "teanlp"))

    def __init__(self):
        if not Neo4JUtils.instance:
            Neo4JUtils.instance = Neo4JUtils.__Neo4JUtils()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    @staticmethod
    def health():
        try:
            Neo4JUtils.graph.run("Match () Return 1 Limit 1")
        except:
            return json.dumps({'result': 'OFF'})
        return json.dumps({'result': 'ON'})

    @staticmethod
    def run(value):
        Neo4JUtils.instance._graph.run(value)
