import numpy as np
import operator
from os import listdir

def create_data_set():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataset, labels, k):
    dataset_size = dataset.shape[0]
    diff_mat = np.tile(inX, (dataset_size, 1)) - dataset
    sq_diff_mat = diff_mat ** 2
    sq_distance = sq_diff_mat.sum(axis=1)
    distance = sq_distance ** 0.5
    sorted_dist_indices = distance.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist_indices[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file2matrix(filename):
    love_directory = {'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    with open(filename) as f:
        lines = f.readlines()
    number_of_lines = len(lines)
    return_matrix = np.zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_matrix[index, :] = list_from_line[0: 3]
        class_label_vector.append(love_directory.get(list_from_line[-1]))
        index += 1
    return return_matrix, class_label_vector


def auto_norm(dataset):
    min_vals = dataset.min(0)
    max_vals = dataset.max(0)
    ranges = max_vals - min_vals
    norm_dataset = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    norm_dataset = dataset - np.tile(min_vals, (m, 1))
    norm_dataset = norm_dataset / np.tile(ranges, (m, 1))
    return norm_dataset, ranges, min_vals


def dating_class_test():
    ho_ratio = 0.10
    dating_data_matrix, dating_labels = file2matrix('datingTestSet.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_matrix)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :],
                                      dating_labels[num_test_vecs:m], 4)
        print("the classifier came back with: %d, the real answer is: %d"
              % (classifier_result, dating_labels[i]))
        if (classifier_result != dating_labels[i]):
            error_count += 1.0
    print("-"*80)
    print("the total error rate is: %f%%" % (100 * error_count / float(num_test_vecs)))


def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input("percentage of time spent playing video games?"))
    ff_miles = float(input("frequent flier miles earned per year?"))
    ice_cream = float(input("liters of ice cream consumed per year?"))
    dating_data_matrix, dating_labels = file2matrix('datingTestSet.txt')
    norm_matrix, ranges, min_vals = auto_norm(dating_data_matrix)
    in_arr = np.array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_vals) / ranges,
                                  norm_matrix, dating_labels, 3)
    print("You will probably like this person: ",
          result_list[classifier_result - 1])


def img2vector(filename):
    with open(filename) as f:
        content = ''.join(f.readlines()).replace('\n', '')
        return np.array([int(c) for c in content], ndmin=2)


def handwriting_class_test():
    hw_labels = []
    training_file_list = listdir('trainingDigits')
    m = len(training_file_list)
    training_matrix = np.zeros((m, 1024))
    for i in range(m):
        filename_str = training_file_list[i]
        file_str = filename_str.split('.')[0]
        class_num = int(file_str.split('_')[0])
        hw_labels.append(class_num)
        training_matrix[i, :] = img2vector('trainingDigits/%s' % filename_str)

    test_file_list = listdir('testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        filename_str = test_file_list[i]
        file_str = filename_str.split('.')[0]
        class_num = int(file_str.split('_')[0])
        vector_under_test = img2vector('testDigits/%s' % filename_str)
        classify_result = classify0(vector_under_test,
                                    training_matrix,
                                    hw_labels,
                                    3)
        print("the classifier came back with: %d, the real answer is: %d"
              % (classify_result, class_num))
        if classify_result != class_num: error_count += 1.0

    print("\nthe total number of errors is: %d" % error_count)
    print("\nthe total error rate is: %.2f%%" % (100 * error_count / m_test))



if __name__ == '__main__':
    handwriting_class_test()