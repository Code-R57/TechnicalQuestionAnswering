# Technical Question Answering

## Description of the Files

1. wiki_extractor.py : Python file to extract Wikipedia data for the Entities present in TeKnowbase.
2. indexing.py : Python file to index thr S2ORC dataset using ElasticSearch library.
3. graph_builder.py : Python file to build a graph from the KnowledgeGraph Entities and Relations.
4. scraper.py : Python file to read a TSV file, retrie related questions for specific entities in the file, and write the resulting questions to a text file named 'questions.txt', with each question on a new line.
5. script.py : Python file to load a TSV file as a knowledge graph using PyKEEN and query the graph to retrieve triples based on specific query criteria.
6. test.py : Aa python file to load a TSV file representing a knowledge graph, extract triples from queries, and query the graph to retrieve related entities or relations.
7. questions.txt : It is a list of technical questions.
8. answers.txt : It is a file containing the answers to the questions in questions.txt .
