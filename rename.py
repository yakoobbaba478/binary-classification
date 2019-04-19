import os
path = '/home/baba/Documents/classify__nn/training_set/dogs'
files = os.listdir(path)


for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, "cat_"+str(index)+'.jpg'))
