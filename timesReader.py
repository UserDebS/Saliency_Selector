import pickle as p
import os

folder = './times'
for filename in os.listdir(folder):
    print(f'For {filename} the scores are :', end=' ')
    with open(f'{folder}/{filename}', 'rb') as file:
        print(p.load(file=file),'\n')