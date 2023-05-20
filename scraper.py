import people_also_ask as paa
from itertools import chain
import pandas as pd

df = pd.read_csv('TeKnowbaseEntities.tsv', sep='\t')

final = []
for i in df[75:101].itertuples(index=False):
    final.append(paa.get_related_questions(i.entity_name))

finalf = list(chain.from_iterable(final))

with open(r'questions.txt', 'a') as fp:
    fp.write('\n'.join(finalf))