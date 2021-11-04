import numpy as np


def create_network_state():
    return {
        'activities': [0],
        'activity_derivatives': [0],
        'weight_adjustments': [0],
        'bias_adjustments': [0],
        'weights': [0],
        'biases': [0],
        'correction_terms': [(1 / 784) ** 0.5],
        'sample_accuracy': [0],
        'batch_accuracies': [0]
    }


def extend_network_state(params, network_state):
    for layer in range(1, len(params['layers'])):
        network_state['activities'].append(np.zeros((params['layers'][layer], 1)))
        network_state['activity_derivatives'].append(np.zeros((params['layers'][layer], 1)))
        network_state['weight_adjustments'].append(np.zeros((params['layers'][layer], params['layers'][layer - 1])))
        network_state['bias_adjustments'].append(np.zeros((params['layers'][layer], 1)))
        network_state['correction_terms'].append(initialize_correction_terms(params, layer))
        network_state['weights'].append(initialize_weights(params, layer, network_state))
        network_state['biases'].append(np.zeros((params['layers'][layer], 1)))
    return network_state


def initialize_correction_terms(parameters, layer):
    if parameters['correction']:
        return (1 / parameters['layers'][layer]) ** 0.5
    else:
        return 1.0


def initialize_weights(parameters, layer, state):
    correction = (len(parameters['layers']) > 7) * state['correction_terms'][layer] + (len(parameters['layers']) < 8)
    var = parameters['weight_initialization']
    current_layer, previous_layer = parameters['layers'][layer], parameters['layers'][layer - 1]
    if var == 2:
        variance = (2 / previous_layer) ** 0.5 / correction
        return np.random.normal(0.0, variance, (current_layer, previous_layer))
    elif var == 1:
        bound = ((6 / (current_layer + previous_layer)) ** 0.5) / correction
        return np.random.uniform(0 - bound, bound, (current_layer, previous_layer))
    else:
        return np.random.random_sample((current_layer, previous_layer)) * 2 - 1


def initialize_network_state(params):
    network_state = create_network_state()
    network_state = extend_network_state(params, network_state)
    return network_state


def batch_to_input_layer(data_batch, state):
    state['activities'][0] = (data_batch[state['current_iteration']]) / 255
    return state


def determine_solution(parameters, batch_index):
    solution_array = np.zeros((parameters['layers'][-1], 1))
    solution_array[(batch_index % parameters['layers'][-1]), 0] = 1
    return solution_array


def sigmoid(array):
    return 1 / (1 + np.exp(- array))


def relu(array, correction_term):
    array = np.maximum(array, 0) * correction_term
    return array


def forward_pass(parameters, state):
    act = parameters['activation']
    for layer in range(1, len(parameters['layers'])):
        w = state['weights'][layer]
        a = state['activities'][layer - 1]
        b = state['biases'][layer]
        a = (w @ a) + b
        state['activities'][layer] = (act == 0) * sigmoid(a) + (act == 1) * relu(a, state['correction_terms'][layer])
    return state


def cost_function(output, target):
    return np.sum((output - target)) ** 2


def check_prediction(given_answer, correct_answer):
    if np.argmax(given_answer) == np.argmax(correct_answer):
        return 1
    else:
        return 0


def backpropagation(parameters, state):
    act = parameters['activation']
    solution = determine_solution(parameters, state['current_iteration'])
    state['sample_accuracy'] = check_prediction(state['activities'][-1], solution)
    state['batch_accuracies'][-1] += state['sample_accuracy']
    state['activity_derivatives'][-1] = 2 * (solution - state['activities'][-1])

    for layer in range(len(parameters['layers']) - 1, 0, -1):

        dg_a = np.diag(state['activities'][layer].reshape(-1))
        i = np.identity(parameters['layers'][layer])
        d_a = state['activity_derivatives'][layer]
        row_a = state['activities'][layer - 1].reshape(1, -1)
        w_t = state['weights'][layer].T
        ct = state['correction_terms'][layer]

        state['weight_adjustments'][layer] += (act == 0) * (dg_a @ (i - dg_a) @ d_a @ row_a)
        state['weight_adjustments'][layer] += (act == 1) * (np.sign(dg_a) @ d_a @ row_a * ct)
        state['bias_adjustments'][layer] += (act == 0) * (dg_a @ (i - dg_a) @ d_a)
        state['bias_adjustments'][layer] += (act == 1) * (np.sign(dg_a) @ d_a * ct)

        if layer > 1:
            d_sigmoid = w_t @ (dg_a @ (i - dg_a) @ d_a)
            d_relu = w_t @ np.sign(dg_a) @ d_a * ct
            state['activity_derivatives'][layer - 1] = (act == 0) * d_sigmoid + (act == 1) * d_relu

    return state


def update_batch_accuracies(parameters, state):
    state['batch_accuracies'][-1] /= parameters['batch_size']
    state['batch_accuracies'].append(0)


def learn(parameters, state):
    update_batch_accuracies(parameters, state)
    for layer in range(len(parameters['layers'])):
        state['weights'][layer] += state['weight_adjustments'][layer] * parameters['learning_rate']
        state['biases'][layer] += state['bias_adjustments'][layer] * parameters['learning_rate']
        state['weight_adjustments'][layer] = np.zeros((parameters['layers'][layer], parameters['layers'][layer - 1]))
        state['bias_adjustments'][layer] = np.zeros((parameters['layers'][layer], 1))
    return state


def train(parameters, state, batch):
    state = batch_to_input_layer(batch, state)
    state = forward_pass(parameters, state)
    state = backpropagation(parameters, state)
    return state


def test(parameters, state, dataset):
    correct_predictions, total_predictions = 0, 0
    for digit in range(len(dataset)):
        total_predictions += len(dataset[digit])
        for sample in range(len(dataset[digit])):
            state['activities'][0] = dataset[digit][sample].reshape((784, 1)) / 255
            state = forward_pass(parameters, state)
            if np.argmax(state['activities'][-1]) == digit:
                correct_predictions += 1
    print(f'Model Accuracy: {(correct_predictions / total_predictions) * 100} %')
