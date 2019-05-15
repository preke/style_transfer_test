import random
from gensim.models import Word2Vec


# AMAZON_PATH = '../data/amazon/'
# YELP_PATH = '../data/yelp/'

# amazon_train = []
# amazon_test  = []

# with open(AMAZON_PATH+'sentiment.train.1', 'r') as reader:
#     for line in reader:
#         new_line = '1\t' + line
#         amazon_train.append(new_line)

# with open(AMAZON_PATH+'sentiment.dev.1', 'r') as reader:
#     for line in reader:
#         new_line = '1\t' + line
#         amazon_train.append(new_line)


# with open(AMAZON_PATH+'sentiment.train.0', 'r') as reader:
#     for line in reader:
#         new_line = '0\t' + line
#         amazon_train.append(new_line)

# with open(AMAZON_PATH+'sentiment.dev.0', 'r') as reader:
#     for line in reader:
#         new_line = '0\t' + line
#         amazon_train.append(new_line)

# with open(AMAZON_PATH+'sentiment.test.1', 'r') as reader:
#     for line in reader:
#         new_line = '1\t' + line
#         amazon_test.append(new_line)

# with open(AMAZON_PATH+'sentiment.test.0', 'r') as reader:
#     for line in reader:
#         new_line = '0\t' + line
#         amazon_test.append(new_line)

# random.shuffle(amazon_train)
# print (len(amazon_train))

# random.shuffle(amazon_test)
# print (len(amazon_test))

# writer = open(AMAZON_PATH + 'train.tsv', 'w')
# for line in amazon_train:
#     writer.write(line)
# writer.close()

# writer = open(AMAZON_PATH + 'test.tsv', 'w')
# for line in amazon_test:
#     writer.write(line)
# writer.close()



# YELP_train = []
# # YELP_test  = []

# with open(YELP_PATH+'sentiment.train.1', 'r') as reader:
#     for line in reader:
#         new_line = '1\t' + line
#         YELP_train.append(new_line)

# with open(YELP_PATH+'sentiment.dev.1', 'r') as reader:
#     for line in reader:
#         new_line = '1\t' + line
#         YELP_train.append(new_line)


# with open(YELP_PATH+'sentiment.train.0', 'r') as reader:
#     for line in reader:
#         new_line = '0\t' + line
#         YELP_train.append(new_line)

# with open(YELP_PATH+'sentiment.dev.0', 'r') as reader:
#     for line in reader:
#         new_line = '0\t' + line
#         YELP_train.append(new_line)

# with open(YELP_PATH+'sentiment.test.1', 'r') as reader:
#     for line in reader:
#         new_line = '1\t' + line
#         YELP_test.append(new_line)

# with open(YELP_PATH+'sentiment.test.0', 'r') as reader:
#     for line in reader:
#         new_line = '0\t' + line
#         YELP_test.append(new_line)

# random.shuffle(YELP_train)
# print (len(YELP_train))

# # random.shuffle(YELP_test)
# # print (len(YELP_test))

# writer = open(YELP_PATH + 'train.tsv', 'w')
# for line in YELP_train:
#     writer.write(line)
# writer.close()

# writer = open(YELP_PATH + 'test.tsv', 'w')
# for line in YELP_test:
#     writer.write(line)
# writer.close()









# ## reference
# YELP_reference  = []

# with open(YELP_PATH+'reference.1', 'r') as reader:
#     for line in reader:
#         # invert label
#         new_line = '0\t' + line
#         YELP_reference.append(new_line)

# with open(YELP_PATH+'reference.0', 'r') as reader:
#     for line in reader:
#         # invert label
#         new_line = '1\t' + line
#         YELP_reference.append(new_line)


# random.shuffle(YELP_reference)
# print (len(YELP_reference))


# writer = open(YELP_PATH + 'reference.tsv', 'w')
# for line in YELP_reference:
#     writer.write(line)
# writer.close()


## Train w2v model
# list_yelp = []
# english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'']

# def punctuate(text):
#     ans = ""
#     for letter in text:
#         if letter in english_punctuations:
#             ans += ' '
#         else:
#             ans += letter
#     return ans

# with open(YELP_PATH+'train.tsv', 'r') as reader:
#     for line in reader:
#         sentence = line.split('\t')[1]
#         sentence = punctuate(sentence)
#         sentence = sentence.split(' ')
#         list_yelp.append(sentence)

# w2v_model = Word2Vec(list_yelp, min_count=5)
# w2v_model.save('yelp_word2vec.model')



file = '../data/reference.tsv'
fopen = open('ref.data', 'w')
with open(file, 'r') as reader:
    for line in reader:
        list_ = line.split('\t')
        fopen.write(list_[0] + '\t' + list_[2])
        if list_[0] == '1':
            fopen.write('0' + '\t' + list_[1] + '\n')
        if list_[0] == '0':
            fopen.write('1' + '\t' + list_[1] + '\n')    

fopen.close()






