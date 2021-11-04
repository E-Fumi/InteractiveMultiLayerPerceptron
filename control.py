import pygame
import sys


def quit_option(event):
    if event.type == pygame.QUIT:
        sys.exit()


def pause(event, vis_fw):
    if event.key == pygame.K_SPACE:
        vis_fw['pause'] = bool(True - vis_fw['pause'])
    return vis_fw


def check_keyboard_events(vis_fw, event):
    if event.type == pygame.KEYDOWN:
        vis_fw = pause(event, vis_fw)
        if event.key == pygame.K_KP_PLUS or event.key == pygame.K_PLUS:
            vis_fw['delay'] = min(vis_fw['delay'] + 0.025, 1)
        if event.key == pygame.K_KP_MINUS or event.key == pygame.K_MINUS:
            vis_fw['delay'] = max(vis_fw['delay'] - 0.025, 0)
        if event.key == pygame.K_m:
            vis_fw['weight_threshold'] = vis_fw['weight_threshold'] + 0.1
        if event.key == pygame.K_n:
            vis_fw['weight_threshold'] = max(1.001, vis_fw['weight_threshold'] - 0.1)
        if event.key == pygame.K_RETURN:
            vis_fw['show'] = bool(True - vis_fw['show'])
        if event.key == pygame.K_w:
            vis_fw['weights'] = bool(True - vis_fw['weights'])
        if event.key == pygame.K_b:
            vis_fw['biases'] = bool(True - vis_fw['biases'])


def check_mouse_click(vis_fw):
    if pygame.mouse.get_pressed(5)[0]:
        vis_fw['mouse_click'] = True
    else:
        vis_fw['mouse_click'] = False


def hover_detect(vis_fw):
    x_cursor, y_cursor = pygame.mouse.get_pos()
    for first_layer_neuron in range(len(vis_fw['neuron_coordinates'][1])):
        x_neuron = vis_fw['neuron_coordinates'][1][first_layer_neuron][0]
        y_neuron = vis_fw['neuron_coordinates'][1][first_layer_neuron][1]
        if ((x_neuron - x_cursor) ** 2 + (y_neuron - y_cursor) ** 2) ** 0.5 < vis_fw['neuron_radius']:
            vis_fw['hover'] = [True, first_layer_neuron]
            break
        else:
            vis_fw['hover'] = [False, 0]


def control(vis_fw):
    event = pygame.event.poll()
    quit_option(event)
    check_keyboard_events(vis_fw, event)
    check_mouse_click(vis_fw)
    hover_detect(vis_fw)
    vis_fw['instructions'] = False
    x_cursor, y_cursor = pygame.mouse.get_pos()
    d_x = vis_fw['question_mark_coordinates'][0] - x_cursor
    d_y = vis_fw['question_mark_coordinates'][1] - y_cursor
    if (d_x ** 2 + d_y ** 2) ** 0.5 < vis_fw['neuron_radius']:
        vis_fw['instructions'] = True
