import spacy
import pandas as pd
import networkx as nx
from spacy.tokens import Doc, Token, Span
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re
nlp = spacy.load('en_core_web_sm')

# Read TSV file into a Pandas DataFrame
triples_df_inp = pd.read_csv('TeKnowbase.tsv', sep='\t', header=None, names=[
    'entity1', 'relation', 'entity2', 'source', 'method'])
triples_df = triples_df_inp.iloc[:, :3]

# Create a directed graph using NetworkX
knowledge_graph = nx.DiGraph()

# Add nodes to the graph for each unique entity in the triples
for entity in set(triples_df['entity1']).union(set(triples_df['entity2'])):
    knowledge_graph.add_node(entity)

# Add edges to the graph for each triple
for _, row in triples_df.iterrows():
    knowledge_graph.add_edge(
        row['entity1'], row['entity2'], relation=row['relation'])

entity_df = pd.read_csv('TeKnowbaseEntities.tsv', sep='\t')
entity_list = entity_df['entity_name'].tolist()


def split_words(string):
    # Split by any separators
    split_string = re.split(r'[/_()\[\]]', string)

    # Remove any empty strings and strip whitespace
    split_string = [word.strip() for word in split_string if word]

    # Split any conjoined words with capitalization
    words = []
    for word in split_string:
        i = 1
        while i < len(word):
            if word[i].isupper():
                words.append(word[:i])
                word = word[i:]
                i = 1
            else:
                i += 1
        words.append(word)

    return words


def extract_subject(query):
    doc = nlp(query)
    entity = ""
    fl = 0
    for np in doc.noun_chunks:
        # Check if the noun phrase is the subject of the query
        # print(np, np.root.dep_)
        if np.root.dep_ == "nsubj" and "type" not in np.text:
            entity += np.text
            fl = 1
            ind = np.start
            if doc[ind+1].text == "of":
                entity += "_of_"
        if np.root.dep_ == "pobj" and fl == 1:
            entity += np.text
        if np.root.dep_ == "appos" and fl == 1:
            entity += " "+np.text
            fl = 0
    if (len(entity) > 0):
        return entity.replace(" ", "_")
    else:
        for tok in doc:
            print(tok, tok.pos_)
            if tok.text != "type":
                if tok.pos_ == "NOUN" or tok.pos_ == "PROPN" or tok.pos_ == "ADV":
                    entity += tok.text + "_"
        return entity[:-1]


def extract_relation(query):
    doc = nlp(query)
    verb = None
    for token in doc:
        if token.dep_ == "ROOT":
            verb = token
            break

    if not verb:
        return None

    if not verb.has_extension("dobj") and not verb.has_extension("pobj"):
        return "type"

    obj = None
    for child in verb.children:
        if child.dep_ in ["dobj", "pobj"]:
            obj = child
            break

    if not obj:
        return None

    if obj.pos in ["NOUN", "PRON"]:
        return "nsubj"

    noun_chunk = obj
    for child in obj.children:
        if child.dep_ in ["compound", "amod"]:
            noun_chunk = Span(doc, noun_chunk.start, child.i+1)

    relation = None
    for child in noun_chunk.children:
        if child.dep_ == "appos":
            relation = child
        elif child.dep_ == "pobj":
            relation = child
            break

    if not relation:
        relation = noun_chunk

    return relation.text.lower().strip()


# scoring function

def extract_entity(query):
    doc = nlp(query)
    entities = []
    for chunk in doc.noun_chunks:
        if chunk.root.dep_ == 'nsubj':
            entity = chunk.text.strip()
            if len(entity.split()) == 1:
                entities.append(entity)
    if not entities:
        return None
    scores = []
    for entity in entities:
        for e in entity_list:
            score = nlp(entity).similarity(nlp(e))
            scores.append((score, entity))
    scores = sorted(scores, reverse=True)
    best_entity = None
    for score, entity in scores:
        if entity in query and entity not in query.split()[-1]:
            best_entity = entity
            break
    return best_entity


def find_best_relation(query: str, relations_file: str, entities) -> str:
    # load spacy model and parse query
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(query)

    # extract entity and relation from query
    entity = None
    relation = extract_relation(query)
    # for token in doc:
    #     if token.dep_ == "nsubj":
    #         entity = token.text
    #     if token.dep_ == "attr":
    #         relation = token.text
    # load relations file
    relations_df = pd.read_csv(relations_file, delimiter="\t")

    # relations = list(relations_df["relation_name"])
    relations = list(set(list(triples_df.iloc[:, 1])))
    # preprocess relations using same method as query
    preprocessed_relations = []
    for r in relations:
        # print("Relation: ", r)
        doc = nlp(r)
        preprocessed_r = " ".join(
            [t.lemma_ for t in doc if not t.is_stop and t.pos_ in ["NOUN", "VERB", "ADJ"]])
        preprocessed_relations.append(preprocessed_r)
    # print("List of possible: ", preprocessed_relations)
    # preprocess query relation
    doc = nlp(relation)
    preprocessed_relation = " ".join(
        [t.lemma_ for t in doc if not t.is_stop and t.pos_ in ["NOUN", "VERB", "ADJ"]])
    # print(preprocessed_relation)
    # compute similarity scores between preprocessed relation and all preprocessed relations
    vectorizer = TfidfVectorizer()
    relation_vectors = vectorizer.fit_transform(preprocessed_relations)
    # print(relation_vectors)
    query_vector = vectorizer.transform([preprocessed_relation])
    # print(query_vector)
    scores = cosine_similarity(query_vector, relation_vectors)[0]
    entity = entities[0]
    # print(entity)
    possible_relations = list(set([knowledge_graph[entity][nbr]['relation']
                              for nbr in knowledge_graph.neighbors(entity)]))
    # find best relation based on similarity score
    max_score = max(scores)
    best_relation_idx = scores.argmax()
    best_relation = relations[best_relation_idx]
    # Create a dictionary of scores and relations
    score_dict = dict(zip(relations, scores))

    # Sort the relations based on scores in descending order
    sorted_relations = sorted(score_dict, key=score_dict.get, reverse=True)
    print(len(sorted_relations))
# Iterate through sorted relations until a relation is found in possible_relations
    for relation in sorted_relations:
        if relation in possible_relations:
            best_relation = relation
            break

    # for i in range(len(scores)):
    #     print(relations[i], scores[i])
    return best_relation


def find_best_match(list_of_lists, target_list):
    best_match_index = -1
    max_matches = -1
    for i, word_list in enumerate(list_of_lists):
        matches = sum(word in target_list for word in word_list)
        if matches > max_matches:
            max_matches = matches
            best_match_index = i
    return best_match_index


def extract_triple(query):
    doc = nlp(query)
    entities = []
    entities.append(extract_subject(query))
    relations = []
    relations.append(find_best_relation(
        query, "TeKnowbaseRelations.tsv", entities))
    print("Entity identified: ", entities[0])
    print("Relation identified: ", relations[0])
    # for chunk in doc.noun_chunks:
    #     if chunk.root.dep_ == 'pobj':
    #         for tok in chunk.root.head.children:
    #             if tok.dep_ == 'relcl':
    #                 for child in tok.children:
    #                     if child.dep_ == 'dobj':
    #                         relation = tok.text + child.text.capitalize()
    #                         relations.append(relation)
    #             elif tok.dep_ == 'prep':
    #                 for child in tok.children:
    #                     if child.dep_ == 'pobj':
    #                         relation = tok.text + child.text.capitalize()
    #                         relations.append(relation)
    # if len(entities) != 1:
    #     return None
    # if len(relations) == 0:
    #     # Relation not explicitly mentioned, check if it exists in the knowledge graph
    #     entity = entities[0]
    #     # print(entity)
    #     possible_relations = list(set(
    #         [knowledge_graph[entity][nbr]['relation'] for nbr in knowledge_graph.neighbors(entity)]))
    #     # print("Possible relations: ", possible_relations)
    #     words = []
    #     for i in possible_relations:
    #         templ = split_words(i)
    #         words.append([i.lower() for i in templ])
    #     if len(possible_relations) == 0:
    #         return None
    #     best_relation = [
    #         possible_relations[find_best_match(words, query.split())]]
    #     return tuple(entities + best_relation)
    # elif len(relations) == 1:
    #     return tuple(entities + relations)
    # else:
    #     # Multiple relations found, return all of them
    return tuple(entities + list(set(relations)))


# def query_knowledge_graph(entity1, relation, entity2):
#     return knowledge_graph.has_edge(entity1, entity2) and knowledge_graph[entity1][entity2]['relation'] == relation
def query_knowledge_graph(query):
    triple = extract_triple(query)
    # print(triple)
    if triple is None:
        return None
    entity, relation = triple[:2]
    if relation not in [str(x[2]['relation']) for x in knowledge_graph.edges(data=True)]:
        return None
    connected_nodes = list(knowledge_graph[entity])
    for connected_node in connected_nodes:
        if knowledge_graph[entity][connected_node]['relation'] == relation:
            return connected_node
    return None


def query_knowledge_graph_entity(query):
    triple = extract_triple(query)
    if triple is None:
        return None
    entity1, relation = triple[0], triple[1]
    for _, row in triples_df[triples_df['entity1'] == entity1].iterrows():
        if row['relation'] == relation:
            return row['entity2']
    return None


# Example query
# result_bool = query_knowledge_graph("prisoners_dilemma is a type of?")

query = "What is particle accelerator related to?"
print("You asked: ", query)
print("Entity: ", extract_entity(query))
result_entity = query_knowledge_graph(query)
print("Answer: ", result_entity.replace("_", " "))
