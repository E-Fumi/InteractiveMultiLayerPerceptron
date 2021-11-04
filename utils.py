import pickle
import numpy as np


def prepare_translation_matrices():
    matrices, resolution = [], 784
    shifts = [1, 27, 28, 29, 755, 756, 757, 783]
    for shift in range(8):
        translation_matrix = np.zeros((resolution, resolution))
        for value in range(resolution):
            translation_matrix[(shifts[shift] + value) % resolution][value] = 1
        matrices.append(translation_matrix)
    return matrices


def check_for_value_errors(parameters, reference):
    for key in reference:
        if not isinstance(parameters[key], type(reference[key])):
            parameters[key] = reference[key]
            print(f'ValueError found in {key}: {key} reverted to default')
    return parameters


def check_architecture(parameters):
    layers = [784]
    for layer in range(len(parameters['layers'])):
        if isinstance(parameters['layers'][layer], int) and 0 < parameters['layers'][layer] < 33:
            layers.append(parameters['layers'][layer])
    layers.append(10 + parameters['add_random'])
    return layers


def check_activation(parameters):
    if parameters['activation'][0].lower() not in 'sr':
        print('Activation function not recognized, reverted to sigmoid.')
        return 0
    else:
        return int(parameters['activation'][0].lower() == 'r')


def check_weights_init(parameters):
    if parameters['weight_initialization'][0].lower() not in 'suxh':
        print('Weight initialization not recognized, reverted to standard uniform.')
        return 0
    else:
        weight_init = 0 + int(parameters['weight_initialization'][0].lower() == 'x')
        weight_init += 2 * int(parameters['weight_initialization'][0].lower() == 'h')
        return weight_init


def check_batch_size(parameters):
    if parameters['batch_size'] > 32 * parameters['layers'][-1]:
        parameters['batch_size'] = 32 * parameters['layers'][-1]
        print('batch size too large for visualization, reduced to maximum supported batch size')
    if parameters['batch_size'] % parameters['layers'][-1] != 0:
        multiple = (parameters['batch size'] // parameters['layers'][-1]) + 1
        parameters['batch_size'] = multiple * parameters['layers'][-1]
        print('batch size modified to be divisible by nr of neurons in output layer')
    return parameters['batch_size']


def initialize_parameters(parameters):
    pickle_in = open('default_parameters.pickle', 'rb')
    reference = pickle.load(pickle_in)
    parameters = check_for_value_errors(parameters, reference)
    parameters['activation'] = check_activation(parameters)
    parameters['weight_initialization'] = check_weights_init(parameters)
    parameters['layers'] = check_architecture(parameters)
    parameters['batch_size'] = check_batch_size(parameters)
    parameters['translation_matrices'] = prepare_translation_matrices()
    return parameters
