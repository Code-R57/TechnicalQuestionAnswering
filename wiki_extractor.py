import wikipedia as wk
import json
import csv

wiki_dump = {}
missing = []
i = 1
with open("TeKnowbaseEntities.tsv") as f:
        for line in f:
                if i<=4600:
                        i+=1
                        continue
                l = line.split('\t')
                wiki_query = l[0]
                try:
                        content = wk.page(wiki_query, auto_suggest=False).content
                        wiki_dump[wiki_query] = content
                except:
                        missing.append(wiki_query)
                i+=1
                print(i)
                if i%1000 == 0:
                        jsonString = json.dumps(wiki_dump)
                        jsonFile = open("wiki_data_4600.json", "w")
                        jsonFile.write(jsonString)
                        jsonFile.close()
#print(wiki_dump)
print("missing is ")
print(missing)
with open('missing_entities', 'w') as miss_f:
        write = csv.writer(miss_f)
        write.writerow(missing)
jsonString = json.dumps(wiki_dump)
jsonFile = open("wiki_data_4600.json", "w")
jsonFile.write(jsonString)
jsonFile.close()