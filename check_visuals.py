import pygame


def add_check_visual_parameters(vis_fw, parameters):
    vis_fw['check_radius'] = determine_check_radius(vis_fw, parameters)
    vis_fw['check_spacer'] = 3 * vis_fw['check_radius']
    vis_fw['check_width'] = (parameters['layers'][-1] * 3 - 1) * vis_fw['check_radius']
    vis_fw['check_coordinates'] = determine_check_coordinates(vis_fw, parameters)
    vis_fw['check_height'] = vis_fw['check_coordinates'][-1][1] - vis_fw['check_coordinates'][0][1]
    vis_fw['batch_size'] = parameters['batch_size']
    vis_fw['outputs'] = parameters['layers'][-1]


def determine_check_radius(vis_fw, parameters):
    available_y_space = vis_fw['panel_height'] - vis_fw['allotted_panel_height']
    available_x_space = vis_fw['panel_width']
    check_rows = (parameters['batch_size'] / parameters['layers'][-1])
    check_columns = parameters['layers'][-1]
    return min(4, min(available_y_space // (3 * check_rows - 1), available_x_space // (3 * check_columns - 1)))


def determine_check_coordinates(vis_fw, parameters):
    coordinates = []
    offset = vis_fw['check_radius'] + (vis_fw['panel_width'] - vis_fw['check_width']) // 2
    for sample in range(parameters['batch_size']):
        x = vis_fw['side_padding'] + offset + (sample % parameters['layers'][-1]) * vis_fw['check_spacer']
        y = vis_fw['allotted_panel_height'] + (sample // parameters['layers'][-1]) * vis_fw['check_spacer']
        coordinates.append([x, y])
    return coordinates


def display_check(vis_fw, state):
    x = vis_fw['check_coordinates'][state['current_iteration']][0]
    y = vis_fw['check_coordinates'][state['current_iteration']][1]
    colour = (200 - state['sample_accuracy'] * 200, state['sample_accuracy'] * 200, 0)
    pygame.draw.circle(vis_fw['screen'], colour, (x, y), vis_fw['check_radius'])
