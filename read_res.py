import pickle

# Read dictionary pkl file
with open('question1.pkl', 'rb') as fp:
    res1 = pickle.load(fp)
    print(res1)