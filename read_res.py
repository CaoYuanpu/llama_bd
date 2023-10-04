import pickle

# Read dictionary pkl file
with open('question1.pkl', 'rb') as fp:
    res1 = pickle.load(fp)
    # print(res1)

for k, v in res1.items():
    print(k)
    print(v['Top_10_tokens'])
    print(v['probs'])
    print()