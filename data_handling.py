import numpy as np
import idx2numpy
import random
import pickle
import os


def convert_idx_to_array(test_or_train):
    idx_files = {'train': ('train-images.idx3-ubyte', 'train-labels.idx1-ubyte'),
                 'test': ('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')}
    images = idx2numpy.convert_from_file(idx_files[test_or_train][0])
    labels = idx2numpy.convert_from_file(idx_files[test_or_train][1])
    return images, labels


def create_empty_lists(add_random):
    dataset = []
    for digits in range(10 + add_random):
        dataset.append([])
    return dataset


def create_random_shape(dataset):
    random_shape, shape_components = np.zeros((28, 28)), []
    while len(shape_components) < 3:
        digit = random.randint(0, 9)
        if digit not in shape_components:
            shape_components.append(digit)
    for component in range(len(shape_components)):
        random_shape += random.choice(dataset[shape_components[component]])
    random_shape /= 3
    return random_shape.astype(int)


def append_random_shapes(dataset):
    shapes = 0
    for index in range(len(dataset)):
        if len(dataset[index]) > shapes:
            shapes = len(dataset[index])
    for shape in range(shapes):
        dataset[-1].append(create_random_shape(dataset))
    return dataset


def create_dataset(test_or_train, add_random):
    dataset = create_empty_lists(add_random)
    images, labels = convert_idx_to_array(test_or_train)
    for digit in range(images.shape[0]):
        dataset[labels[digit]].append(images[digit])
    if add_random:
        dataset = append_random_shapes(dataset)
    return dataset


def save_dataset(file, dataset):
    pickle_out = open(file, 'wb')
    pickle.dump(dataset, pickle_out)
    pickle_out.close()


def shuffle_dataset(dataset):
    for digit in range(len(dataset)):
        random.shuffle(dataset[digit])
    return dataset


def assemble_dataset(train_or_test, add_random):
    file = './' + train_or_test + '_ds' + add_random * '_w_random' + '.pickle'
    if os.path.isfile('./' + file):
        pickle_in = open('train_ds_w_random.pickle', 'rb')
        dataset = pickle.load(pickle_in)
        return shuffle_dataset(dataset)
    else:
        dataset = create_dataset(train_or_test, add_random)
        save_dataset(file, dataset)
        return shuffle_dataset(dataset)


def determine_number_of_batches(parameters, dataset):
    batch_nr = 10 ** 8
    for digit in range(len(dataset)):
        digits_in_dataset = len(dataset[digit])
        digits_in_batch = parameters['batch_size'] // parameters['layers'][-1]
        if batch_nr > (digits_in_dataset // digits_in_batch):
            batch_nr = digits_in_dataset // digits_in_batch
    return batch_nr


def augment_batch(batch, parameters):
    for element in range(len(batch)):
        shift = random.randint(0, 8)
        if shift < 8:
            batch[element] = parameters['translation_matrices'][shift] @ batch[element].reshape((784, 1))
    return batch


def assemble_batch(dataset, parameters, state):
    batch = []
    index = state['current_batch'] * parameters['batch_size']
    digits = parameters['layers'][-1]
    for element in range(parameters['batch_size']):
        digit = element % digits
        batch.append(dataset[digit][((element + index) // digits)].reshape((784, 1)))
    if not parameters['data_augmentation']:
        return batch
    else:
        return augment_batch(batch, parameters)


def assemble_dataset_and_update_state(epoch, parameters, state):
    dataset = assemble_dataset('train', parameters['add random'])
    state['current_epoch'] = epoch
    state['batches_per_epoch'] = determine_number_of_batches(parameters, dataset)
    return dataset, state
