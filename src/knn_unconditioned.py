import numpy as np
import math
import time, random, mnistloader
from collections import Counter

'''
    ### KNN Tester ###

    Trains the kNN on differently sized, randomly sampled training sets, and then runs it on the testing set.
    Prints accuracy and time taken for each run.

'''

training_data, validation_data, test_data = mnistloader.load_data_wrapper()
random.shuffle(training_data)
random.shuffle(test_data)

training = np.array([point[0].flatten() for point in training_data])
training_digits = np.array([point[1] for point in training_data])
testing = np.array([point[0].flatten() for point in test_data])
testing_digits = np.array([point[1] for point in test_data])

def kNN(k, testingrange, testing, testing_digits, training, training_digits):
    success = 0
    k = int(math.sqrt(len(training)))
    for i in range(testingrange):
        vector = testing[i]
        digit = testing_digits[i]
        sq_distances = ((training - vector)**2).sum(axis=1)
        ids = sq_distances.argsort()[:k]
        results = np.array(training_digits)[ids]
        c = Counter(results)
        if c.most_common(1)[0][0] == digit:
            success += 1
    return success

def test(testingrange, testing, testing_digits, training, training_digits):
    successes = kNN(11, testingrange, testing, testing_digits, training, training_digits)
    success_rate = round(successes/testingrange, 5)
    return successes, success_rate

def timed_run(trainingrange):
    testingrange = len(test_data)

    random.shuffle(training_data)
    random.shuffle(test_data)
    training = np.array([point[0].flatten() for point in training_data])[:trainingrange]
    training_digits = np.array([point[1] for point in training_data])[:trainingrange]
    testing = np.array([point[0].flatten() for point in test_data])
    testing_digits = np.array([point[1] for point in test_data])

    start = time.time()
    successes, success_rate = test(testingrange, testing, testing_digits, training, training_digits)
    total_time = time.time() - start
    print(f'Training set size (n): {trainingrange} Accuracy: {success_rate} Time taken/s: {total_time} Testing set size: {testingrange} Number of successes: {successes}')
    return 

def main():
    runs = [timed_run(50), timed_run(150), timed_run(500), timed_run(1000), timed_run(5000), timed_run(10000), timed_run(50000)]

if __name__ == '__main__':
    main()