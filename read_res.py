import pickle

# Read dictionary pkl file
with open('question1.pkl', 'rb') as fp:
    res1 = pickle.load(fp)
    # print(res1)

with open('question3.pkl', 'rb') as fp:
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
    
for i in range(5):
    for j in range(33):
        print(res1[i]['h'][j].shape)
        print(res3[i]['h'][j].shape)