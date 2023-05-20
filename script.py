from pykeen.triples import TriplesFactory

# Load a TSV file as a knowledge graph
triples_factory = TriplesFactory.from_path("TeKnowbase.tsv")

# Query the graph
result = triples_factory.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
