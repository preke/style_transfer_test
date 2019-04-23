import random



AMAZON_PATH = '../data/amazon/'
YELP_PATH = '../data/yelp/'

amazon_train = []
amazon_test  = []

with open(AMAZON_PATH+'sentiment.train.1', 'r') as reader:
    for line in reader:
        new_line = '1\t' + line
        amazon_train.append(new_line)

with open(AMAZON_PATH+'sentiment.dev.1', 'r') as reader:
    for line in reader:
        new_line = '1\t' + line
        amazon_train.append(new_line)


with open(AMAZON_PATH+'sentiment.train.0', 'r') as reader:
    for line in reader:
        new_line = '0\t' + line
        amazon_train.append(new_line)

with open(AMAZON_PATH+'sentiment.dev.0', 'r') as reader:
    for line in reader:
        new_line = '0\t' + line
        amazon_train.append(new_line)

with open(AMAZON_PATH+'sentiment.test.1', 'r') as reader:
    for line in reader:
        new_line = '1\t' + line
        amazon_test.append(new_line)

with open(AMAZON_PATH+'sentiment.test.0', 'r') as reader:
    for line in reader:
        new_line = '0\t' + line
        amazon_test.append(new_line)

random.shuffle(amazon_train)
print (len(amazon_train))

random.shuffle(amazon_test)
print (len(amazon_test))

writer = open(AMAZON_PATH + 'train.tsv', 'w')
for line in amazon_train:
    writer.write(line)
writer.close()

writer = open(AMAZON_PATH + 'test.tsv', 'w')
for line in amazon_test:
    writer.write(line)
writer.close()



YELP_train = []
YELP_test  = []

with open(YELP_PATH+'sentiment.train.1', 'r') as reader:
    for line in reader:
        new_line = '1\t' + line
        YELP_train.append(new_line)

with open(YELP_PATH+'sentiment.dev.1', 'r') as reader:
    for line in reader:
        new_line = '1\t' + line
        YELP_train.append(new_line)


with open(YELP_PATH+'sentiment.train.0', 'r') as reader:
    for line in reader:
        new_line = '0\t' + line
        YELP_train.append(new_line)

with open(YELP_PATH+'sentiment.dev.0', 'r') as reader:
    for line in reader:
        new_line = '0\t' + line
        YELP_train.append(new_line)

with open(YELP_PATH+'sentiment.test.1', 'r') as reader:
    for line in reader:
        new_line = '1\t' + line
        YELP_test.append(new_line)

with open(YELP_PATH+'sentiment.test.0', 'r') as reader:
    for line in reader:
        new_line = '0\t' + line
        YELP_test.append(new_line)

random.shuffle(YELP_train)
print (len(YELP_train))

random.shuffle(YELP_test)
print (len(YELP_test))

writer = open(YELP_PATH + 'train.tsv', 'w')
for line in YELP_train:
    writer.write(line)
writer.close()

writer = open(YELP_PATH + 'test.tsv', 'w')
for line in YELP_test:
    writer.write(line)
writer.close()


