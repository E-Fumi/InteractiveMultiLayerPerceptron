import pygame
import numpy as np
import network_visuals as net_vis


def display_input_weights(vis_fw, state):
    if vis_fw['hover'][0]:
        surface = pygame.Surface((vis_fw['input_width'], vis_fw['input_height']), pygame.SRCALPHA)
        for row in range(vis_fw['input_rows']):
            for column in range(vis_fw['input_columns']):
                pixel = row * vis_fw['input_columns'] + column
                colour = get_input_weight_colour(state, vis_fw['hover'][1], vis_fw, pixel)
                x = column * vis_fw['input_pixel']
                y = row * vis_fw['input_pixel']
                pygame.draw.rect(surface, colour, (x, y, vis_fw['input_pixel'], vis_fw['input_pixel']))
        vis_fw['screen'].blit(surface, (vis_fw['input_coordinates'][0][0], vis_fw['input_coordinates'][0][1]))


def get_input_weight_colour(state, neuron, vis_fw, pixel):
    scaling_factor = 255 / np.amax(abs(state['weights'][1][neuron]))
    if vis_fw['mouse_click']:
        colour = get_transparent_weight_colour(state, [1, neuron, pixel], scaling_factor)
    else:
        colour = net_vis.get_weight_colour(state, [1, neuron, pixel], scaling_factor)
    return colour


def get_transparent_weight_colour(state, indices, scaling_factor):
    r = int(max(0, -state['weights'][indices[0]][indices[1]][indices[2]] * scaling_factor))
    g = int(max(0, state['weights'][indices[0]][indices[1]][indices[2]] * scaling_factor))
    b = int(max(0, state['weights'][indices[0]][indices[1]][indices[2]] * scaling_factor))
    a = int(abs(state['weights'][indices[0]][indices[1]][indices[2]] * scaling_factor * 3 / 4))
    return r, g, b, a
