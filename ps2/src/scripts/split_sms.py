import csv
import random

random.seed(4531231326)

if __name__ == '__main__':
    with open('scripts/SMSSpamCollection.txt', 'r', newline='') as data:
        reader = csv.reader(data, delimiter='\t')
        lines = list(reader)

        random.shuffle(lines)

        print(len(lines))

        num_train = int(len(lines) * 0.8)
        num_val = int(len(lines) * 0.1)
        num_test = len(lines) - num_train

        print(num_train)
        print(num_test)

        train_items = lines[:num_train]
        val_items = lines[num_train:num_train + num_val]
        test_items = lines[num_train + num_val:]

    with open('../data/ds6_train.tsv', 'w', newline='') as train_output:
        writer = csv.writer(train_output, delimiter='\t')
        for line in train_items:
            writer.writerow(line)


    with open('../data/ds6_val.tsv', 'w', newline='') as val_output:
        writer = csv.writer(val_output, delimiter='\t')
        for line in val_items:
            writer.writerow(line)

    with open('../data/ds6_test.tsv', 'w', newline='') as test_output:
        writer = csv.writer(test_output, delimiter='\t')
        for line in test_items:
            writer.writerow(line)
