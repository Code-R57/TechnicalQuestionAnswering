from elasticsearch import Elasticsearch
import gzip
import json

# Connect to Elasticsearch instance
es = Elasticsearch("http://localhost:9200")

# Index settings
index_name = "es_s2orc"
index_settings = {
    "mappings": {
        "properties": {
            "paper_id": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "english"},
            "abstract": {"type": "text", "analyzer": "english"},
            "authors": {"type": "text", "analyzer": "english"},
            "doi": {"type": "keyword"},
            "venue": {"type": "text", "analyzer": "english"},
            "year": {"type": "integer"},
            "inbound_citations": {"type": "keyword"},
            "outbound_citations": {"type": "keyword"},
            "text": {"type": "text"}
        }
    }
}

# es.indices.create(index=index_name, body=index_settings)

batch_size = 1000

data = []

print("Indexing Started")

start_index = 0

with open("files_read.txt", "r") as f:
    start_index = int(f.read())

for i in range(1, 100):
    metadata_file = f"metadata/Computer_Science/metadata_{i}.jsonl.gz"
    pdf_parses_file = f"pdf_parses/Computer_Science/pdf_parses_{i}.jsonl.gz"
    
    print(i)

    with open("files_read.txt", "w") as f:
        f.write('{}'.format(i))

    with open(pdf_parses_file, "rt") as f:
        pdf_parses = []
        for line in f:
            pdf_parses.append(json.loads(line))

    with open(metadata_file, "rt") as f:
        for line in f:
            article = json.loads(line)
            article_id = article.get("paper_id", "")
            title = article.get("title", "")
            abstract = article.get("abstract", "")
            authors = ", ".join([author.get("name","") for author in article.get("authors", [])])
            doi = article.get("doi", "")
            venue = article.get("venue", "")
            year = article.get("year", "")

            inbound_citations = article.get("inbound_citations", [])
            outbound_citations = article.get("outbound_citations", [])

            text = ""

            for paper in pdf_parses:
                if paper["paper_id"] == article_id:
                    for body_text in paper.get("body_text", []):
                        text += body_text.get("text", "") + "\n"
                    break

            data.append({
                "index": {
                    "_index": index_name,
                    "_id": article_id
                }
            })

            data.append({                
                "paper_id": article_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "doi": doi,
                "venue": venue,
                "year": year,
                "inbound_citations": inbound_citations,
                "outbound_citations": outbound_citations,
                "text": text
            })

            if len(data) == batch_size:
                es.bulk(index=index_name, body=data, request_timeout=600)
                data = []

if data:
    es.bulk(index=index_name, body=data, request_timeout=600)

print("Indexing Complete")