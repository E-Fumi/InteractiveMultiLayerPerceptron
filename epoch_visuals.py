import pygame


def add_epoch_visual_parameters(vis_fw, parameters):
    vis_fw['epoch_width'] = min(15, vis_fw['panel_width'] // (3 * parameters['epochs'] + 2))
    vis_fw['epochs_width'] = vis_fw['panel_width']
    vis_fw['solid_epoch_space'] = parameters['epochs'] * vis_fw['epoch_width']
    vis_fw['non_epoch_space'] = vis_fw['epochs_width'] - vis_fw['solid_epoch_space']
    vis_fw['inter_epoch_distance'] = vis_fw['non_epoch_space'] // (parameters['epochs'] + 1)
    vis_fw['epoch_height'] = vis_fw['padding']
    vis_fw['epoch_coordinates'] = determine_epochs_coordinates(vis_fw, parameters)
    vis_fw['allotted_panel_height'] += 2 * vis_fw['padding'] + vis_fw['epoch_height']


def determine_epochs_coordinates(vis_fw, parameters):
    coordinates = []
    for epoch in range(parameters['epochs']):
        offset = vis_fw['side_padding'] + vis_fw['inter_epoch_distance']
        x = epoch * (vis_fw['epoch_width'] + vis_fw['inter_epoch_distance']) + offset
        y = vis_fw['padding']
        coordinates.append([x, y])
    return coordinates


def display_epochs(vis_fw, state):
    completed_epochs = (len(state['batch_accuracies']) - 1) // vis_fw['batches_per_epoch']
    for epoch in range(len(vis_fw['epoch_coordinates'])):
        screen = vis_fw['screen']
        grey, whiter, whitest = (50, 50, 50), (180, 180, 180), (255, 255, 255)
        x = vis_fw['epoch_coordinates'][epoch][0]
        y = vis_fw['epoch_coordinates'][epoch][1]
        pygame.draw.rect(screen, grey, (x, y, vis_fw['epoch_width'], vis_fw['epoch_height']))
        if epoch < completed_epochs:
            whiter_width = max(2, vis_fw['epoch_width'] - 2)
            whitest_width = max(1, vis_fw['epoch_width'] - 4)
            pygame.draw.rect(screen, whiter, (x + 1, y + 1, whiter_width, vis_fw['epoch_height'] - 2))
            pygame.draw.rect(screen, whitest, (x + 2, y + 2, whitest_width, vis_fw['epoch_height'] - 4))
