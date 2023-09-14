from annoy import AnnoyIndex
import numpy as np
import spacy
nlp = spacy.load('en_core_web_md')

t = AnnoyIndex(50, metric="euclidean")
words = list()
lookup = dict()
for i, line in enumerate(open("cmudict-0.7b-simvecs", encoding="latin1")):
    word, vals_raw = line.split("  ")
    word = word.lower().strip("(012)")
    vals = np.array([float(x) for x in vals_raw.split(" ")])
    t.add_item(i, vals)
    words.append(word.lower())
    lookup[word.lower()] = i
t.build(100)

def progress(src_vecs, op_vecs, n=10):
    for i in range(n+1):
        delta = i * (1.0/n)
        val = (src_vecs * (1.0-delta)) + (op_vecs * delta)
        yield val

s = "Hey mom. What? You cleaned out the garage? Nothing, I’m just surprised. That stuff’s been sitting there for years. Well, I wish you would have called earlier. I would have liked to look through it all before you got rid of everything. I’m sure Maxwell would have liked to go through it too. No, I don’t think it would have upset him. It might have been nice to go through all the stuff together, the three of us. I don’t know mom, some kind of closure. It’s all we have left of them. You got rid of all of it? Of course you did. Something of mine? In the deep freeze, weird. Huh. No, I don’t remember. I mean, I did a lot of weird stuff as a kid, but…Ok I’ll come get it. Can you bring it up to your office? I can swing by at lunch. Will you be around? No? Ok then just, uh, leave it there for me, yeah? Ok. Bye."
vecs = np.array([t.get_item_vector(lookup[w.lower()]) for w in s.split()])
print(s)
op_vecs = np.array([t.get_item_vector(lookup["humming"])]*len(vecs))
for res in progress(vecs, op_vecs, n=25):
    print(" ".join([words[nnslookup(t, nlp, words, i)[0]] for i in res]))