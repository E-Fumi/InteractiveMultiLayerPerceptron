import pygame


def add_batch_visual_parameters(vis_fw, parameters):
    vis_fw['batches_per_epoch'] = parameters['batches_per_epoch']
    vis_fw['batches_width'] = vis_fw['panel_width']
    vis_fw['batches_height'] = 4 * (parameters['batches_per_epoch'] // (vis_fw['batches_width'] // 2))
    vis_fw['batches_coordinates'] = determine_batch_coordinates(vis_fw)
    vis_fw['batch_colours'] = []
    vis_fw['allotted_panel_height'] += vis_fw['padding'] + vis_fw['batches_height']


def get_batch_colour(accuracy):
    return 255 - (max(0, 2 * accuracy - 1) * 155), min(255, 2 * accuracy * 255), 0


def draw_empty_batches(vis_fw):
    for entry in range(len(vis_fw['batches_coordinates'])):
        x = vis_fw['batches_coordinates'][entry][0]
        y = vis_fw['batches_coordinates'][entry][1]
        pygame.draw.rect(vis_fw['screen'], (50, 50, 50), (x, y, 1, 1))


def display_batches(vis_fw, state):
    completed_epochs = (len(state['batch_accuracies']) - 1) // vis_fw['batches_per_epoch']
    epoch_batches = (len(state['batch_accuracies']) - 1) % vis_fw['batches_per_epoch']
    total_batches = completed_epochs * vis_fw['batches_per_epoch'] + epoch_batches
    draw_empty_batches(vis_fw)
    if not vis_fw['instructions_displayed']:
        for batch in range(completed_epochs * vis_fw['batches_per_epoch'], total_batches):
            colour = get_batch_colour(state['batch_accuracies'][batch - 1])
            x = vis_fw['batches_coordinates'][batch % vis_fw['batches_per_epoch']][0]
            y = vis_fw['batches_coordinates'][batch % vis_fw['batches_per_epoch']][1]
            pygame.draw.rect(vis_fw['screen'], colour, (x, y, 1, 1))


def determine_batch_coordinates(vis_fw):
    coordinates = []
    batches_per_row = vis_fw['batches_width'] // 2
    batches_per_column = vis_fw['batches_height'] // 4 + 1
    for batch_row in range(batches_per_column):
        y = vis_fw['allotted_panel_height'] + batch_row * 4
        for batch_column in range(batches_per_row):
            x = vis_fw['side_padding'] + batch_column * 2
            coordinates.append([x, y])
    while len(coordinates) != vis_fw['batches_per_epoch']:
        coordinates.pop()
    return coordinates
