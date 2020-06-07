import json

from py2neo import Graph, NodeMatcher, cypher_escape


class Neo4JUtils:

    instance = None

    class __Neo4JUtils:
        def __init__(self):
            # This forces download if not done
            self._graph = Graph("bolt://localhost:7687", auth=("TeaNLP", "teanlp"))
            self._matcher = NodeMatcher(self._graph)

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
    def graph():
        return Neo4JUtils.instance._graph

    @staticmethod
    def matcher():
        return Neo4JUtils.instance._matcher

    @staticmethod
    def run(value):
        Neo4JUtils.instance._graph.run(value)

    @staticmethod
    def find_one(label, property_key, property_value):
        return Neo4JUtils.instance._matcher.match(label).where("_." + property_key + "='" +
                                                               property_value.replace("'", "\\'") + "'").first()

