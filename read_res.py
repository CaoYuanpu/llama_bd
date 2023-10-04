import pickle
import torch

# Read dictionary pkl file
with open('question_js.pkl', 'rb') as fp:
    res1 = pickle.load(fp)
    # print(res1)

with open('question_species.pkl', 'rb') as fp:
    res3 = pickle.load(fp)
    # print(res1)
    
for k, v in res1.items():
    print(k)
    print(v['Top_10_tokens'])
    print(v['probs'])
    print()

for k, v in res3.items():
    print(k)
    print(v['Top_10_tokens'])
    print(v['probs'])
    print()

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-8)
res = []
for i in range(1, 5):
    for j in range(33):
        sim = cos(res1[i]['h'][j].reshape(-1), res3[i]['h'][j].reshape(-1))
        print(f'i: {i} j: {j} cossim: {sim}')
        if i == 1 and j!=0:
            res.append(sim.item())

print(res)
input()