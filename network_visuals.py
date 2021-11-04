import pygame
import numpy as np


def add_network_visual_parameters(vis_fw, parameters):
    vis_fw['inter_layer_distance'] = vis_fw['network_width'] // (len(parameters['layers']) - 1)
    vis_fw['inter_neuron_distance'] = min(60, vis_fw['network_height'] // max(parameters['layers'][1:]))
    vis_fw['neuron_radius'] = min(15, (vis_fw['inter_neuron_distance'] // 2) - 2)
    vis_fw['neuron_coordinates'] = determine_neuron_coordinates(vis_fw, parameters)
    vis_fw['bias_radius'] = vis_fw['neuron_radius'] + 1
    vis_fw['weight_threshold'] = 1.15 + (parameters['weight_initialization'] == 2) * 0.5
    vis_fw['output_labels_text'] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'R']
    vis_fw['label_font'] = pygame.font.Font('JetBrainsMono-ExtraLight.ttf', vis_fw['neuron_radius'])
    vis_fw['output_labels'] = get_labels(vis_fw)
    vis_fw['output_label_coordinates'] = get_output_label_coordinates(vis_fw)
    vis_fw['hover'] = [False, 0]


def determine_neuron_coordinates(vis_fw, parameters):
    coordinates = [[]]
    for layer in range(1, len(parameters['layers'])):
        coordinates.append([])
        for neuron in range(parameters['layers'][layer]):
            layer_height = (parameters['layers'][layer] - 1) * vis_fw['inter_neuron_distance']
            x = vis_fw['panel_width'] + (layer - 0.5) * vis_fw['inter_layer_distance'] + vis_fw['side_padding']
            y = ((vis_fw['screen_height'] - layer_height) // 2) + neuron * vis_fw['inter_neuron_distance']
            coordinates[layer].append([x, y])
    return coordinates


def get_output_label_coordinates(vis_fw):
    coordinates = []
    midway = (vis_fw['neuron_coordinates'][-1][0][0] + vis_fw['screen_width']) // 2
    offset = vis_fw['neuron_coordinates'][-1][0][0] + int(1.25 * vis_fw['side_padding'])
    x = min(midway, offset)
    for label in range(len(vis_fw['neuron_coordinates'][-1])):
        y = vis_fw['neuron_coordinates'][-1][label][1] - (vis_fw['neuron_radius'] // 2 + 2)
        coordinates.append((x, y))
    return coordinates


def get_labels(vis_fw):
    labels = []
    for label in range(len(vis_fw['neuron_coordinates'][-1])):
        label = vis_fw['label_font'].render(vis_fw['output_labels_text'][label], False, (255, 255, 255))
        labels.append(label)
    return labels


def display_output_labels(vis_fw):
    for label in range(len(vis_fw['output_labels'])):
        vis_fw['screen'].blit(vis_fw['output_labels'][label], vis_fw['output_label_coordinates'][label])


def find_max_value(array_list):
    max_value = 0
    for layer in range(1, len(array_list)):
        if np.amax(abs(array_list[layer])) > max_value:
            max_value = np.amax(abs(array_list[layer]))
    return max_value


def display_biases(vis_fw, state):
    max_bias = find_max_value(state['biases'])
    max_value = (1 - vis_fw['biases']) * 10 ** 8 + max_bias
    for layer in range(1, len(vis_fw['neuron_coordinates'])):
        for neuron in range(len(vis_fw['neuron_coordinates'][layer])):
            colour = determine_bias_colour(state, [layer, neuron], max_value)
            x = vis_fw['neuron_coordinates'][layer][neuron][0]
            y = vis_fw['neuron_coordinates'][layer][neuron][1]
            pygame.draw.circle(vis_fw['screen'], colour, (x, y), vis_fw['bias_radius'])
            pygame.draw.circle(vis_fw['screen'], (0, 0, 0), (x, y), vis_fw['neuron_radius'])


def determine_bias_colour(state, indices, max_value):
    if abs(state['biases'][indices[0]][indices[1]]) > 0.1 * max_value:
        colour_offset = int(200 * state['biases'][indices[0]][indices[1]] / max_value)
        colour = (200 - max(0, colour_offset),
                  200 - max(0, colour_offset * -1),
                  200 - max(0, colour_offset * -1))
    else:
        colour = (200, 200, 200)
    return colour


def display_activities(vis_fw, state):
    max_activity = find_max_value(state['activities'])
    epsilon = 10 ** -6
    for layer in range(1, len(vis_fw['neuron_coordinates'])):
        for neuron in range(len(vis_fw['neuron_coordinates'][layer])):
            activity = state['activities'][layer][neuron]
            intensity = int((activity / (max_activity + epsilon)) * 255)
            colour = (intensity, intensity, intensity)
            x = vis_fw['neuron_coordinates'][layer][neuron][0]
            y = vis_fw['neuron_coordinates'][layer][neuron][1]
            pygame.draw.circle(vis_fw['screen'], colour, (x, y), vis_fw['neuron_radius'])


def display_weights(vis_fw, state):
    for layer in range(2, len(state['weights'])):
        for previous_layer_neuron in range(state['weights'][layer].shape[1]):
            for neuron in range(state['weights'][layer].shape[0]):
                weight = state['weights'][layer][neuron][previous_layer_neuron]
                if abs(weight) > np.amax(abs(state['weights'][layer])) / vis_fw['weight_threshold']:
                    x1 = vis_fw['neuron_coordinates'][layer - 1][previous_layer_neuron][0]
                    y1 = vis_fw['neuron_coordinates'][layer - 1][previous_layer_neuron][1]
                    x2 = vis_fw['neuron_coordinates'][layer][neuron][0]
                    y2 = vis_fw['neuron_coordinates'][layer][neuron][1]
                    colour = get_weight_colour(state, [layer, neuron, previous_layer_neuron], 25)
                    pygame.draw.line(vis_fw['screen'], colour, (x1, y1), (x2, y2), 3)


def get_weight_colour(state, indices, scaling_factor):
    r = 0 - int(min(0, state['weights'][indices[0]][indices[1]][indices[2]] * scaling_factor))
    g = 0 + int(max(0, state['weights'][indices[0]][indices[1]][indices[2]] * scaling_factor))
    b = 0 + int(max(0, state['weights'][indices[0]][indices[1]][indices[2]] * scaling_factor))
    colour = (min(r, 255), min(g, 255), min(b, 255))
    return colour
