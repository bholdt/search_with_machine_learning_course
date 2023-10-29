import fasttext 

model = fasttext.load_model("title_model_final.bin")
file = open('in.txt','r')
# list of synonyms
fout = open('synonyms.csv', 'w')
# threshold of 0.75
threshold = 0.75

for word in file.readlines():
    word = word.replace('\n', '')
    # get nearest neighbours for wor
    neighbors = [x[1] for x in list(filter(lambda x: x[0] >= threshold, model.get_nearest_neighbors(word)))]
    if len(neighbors) > 0:
        fout.write(word + ',' + ','.join(neighbors))
fout.close()