import pygame


def add_input_visual_parameters(vis_fw):
    vis_fw['input_columns'], vis_fw['input_rows'] = 28, 28
    vis_fw['input_pixel'] = vis_fw['panel_width'] // vis_fw['input_columns']
    vis_fw['input_height'] = 28 * vis_fw['input_pixel']
    vis_fw['input_width'] = 28 * vis_fw['input_pixel']
    vis_fw['panel_offset'] = (vis_fw['panel_width'] - vis_fw['input_width']) // 2
    vis_fw['input_coordinates'] = [[vis_fw['side_padding'] + vis_fw['panel_offset'], vis_fw['allotted_panel_height']]]
    vis_fw['allotted_panel_height'] += vis_fw['padding'] + vis_fw['input_height']
    vis_fw['mouse_click'] = False


def draw_input_frame(vis_fw):
    x = vis_fw['input_coordinates'][0][0]
    y = vis_fw['input_coordinates'][0][1]
    x_dim = vis_fw['input_width']
    y_dim = vis_fw['input_height']
    pygame.draw.rect(vis_fw['screen'], (255, 255, 255), (x - 2, y - 2, x_dim + 4, y_dim + 4))
    pygame.draw.rect(vis_fw['screen'], (0, 0, 0), (x, y, x_dim, y_dim))


def display_input(vis_fw, state):
    for row in range(vis_fw['input_rows']):
        for column in range(vis_fw['input_columns']):
            x = vis_fw['input_coordinates'][0][0] + (column * vis_fw['input_pixel'])
            y = vis_fw['input_coordinates'][0][1] + (row * vis_fw['input_pixel'])
            intensity = int((state['activities'][0][(row * vis_fw['input_columns']) + column]) * 255)
            colour = (intensity, intensity, intensity)
            pygame.draw.rect(vis_fw['screen'], colour, (x, y, vis_fw['input_pixel'], vis_fw['input_pixel']))
