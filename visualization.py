import pygame
import time
import network_visuals as net_vis
import epoch_visuals as ep_vis
import batch_visuals as b_vis
import input_visuals as in_vis
import input_weights_visuals as in_w_vis
import check_visuals as ch_vis
import instructions_visuals as ins_vis
import control as c


def initialize_visual_framework():
    pygame.init()
    pygame.display.set_caption('Interactive Multilayer Perceptron')
    screen_info = pygame.display.Info()
    screen_width = int(screen_info.current_w // 1.2)
    screen_height = int(screen_info.current_h // 1.2)
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.SRCALPHA)
    visual_framework = {'screen_width': screen_width,
                        'screen_height': screen_height,
                        'screen': screen,
                        'show': True,
                        'pause': False,
                        'weights': True,
                        'biases': True,
                        'instructions': False,
                        'delay': 0.025}
    return visual_framework


def add_general_visual_parameters(vis_fw, parameters):
    vis_fw['padding'] = vis_fw['screen_width'] // 32
    vis_fw['side_padding'] = vis_fw['screen_width'] // 25
    vis_fw['panel_width'] = (vis_fw['screen_width'] //
                             (min(len(parameters['layers']) + 1, 7))) - (2 * vis_fw['side_padding'])
    vis_fw['panel_height'] = vis_fw['screen_height'] - vis_fw['side_padding']
    vis_fw['network_width'] = vis_fw['screen_width'] - vis_fw['panel_width'] - 2 * vis_fw['side_padding']
    vis_fw['network_height'] = vis_fw['screen_height'] - 2 * vis_fw['padding']
    vis_fw['allotted_panel_height'] = 0


def add_visual_parameters(vis_fw, parameters):
    add_general_visual_parameters(vis_fw, parameters)
    net_vis.add_network_visual_parameters(vis_fw, parameters)
    ep_vis.add_epoch_visual_parameters(vis_fw, parameters)
    b_vis.add_batch_visual_parameters(vis_fw, parameters)
    in_vis.add_input_visual_parameters(vis_fw)
    ch_vis.add_check_visual_parameters(vis_fw, parameters)
    ins_vis.add_instructions_visual_parameters(vis_fw)
    return vis_fw


def initialize(parameters):
    visual_framework = initialize_visual_framework()
    visual_framework = add_visual_parameters(visual_framework, parameters)
    return visual_framework


def display_network(vis_fw, state):
    in_vis.display_input(vis_fw, state)
    net_vis.display_activities(vis_fw, state)
    ch_vis.display_check(vis_fw, state)
    in_w_vis.display_input_weights(vis_fw, state)


def batch_update(vis_fw, state):
    vis_fw['screen'].fill((0, 0, 0))
    ep_vis.display_epochs(vis_fw, state)
    b_vis.display_batches(vis_fw, state)
    in_vis.draw_input_frame(vis_fw)
    net_vis.display_output_labels(vis_fw)
    ins_vis.display_question_mark(vis_fw)
    if vis_fw['weights']:
        net_vis.display_weights(vis_fw, state)
    net_vis.display_biases(vis_fw, state)


def pause_loop(vis_fw, state):
    c.control(vis_fw)
    in_vis.display_input(vis_fw, state)
    in_w_vis.display_input_weights(vis_fw, state)
    pygame.display.update()


def delay(vis_fw):
    if vis_fw['delay']:
        start_time = time.time()
        while time.time() < start_time + (vis_fw['delay']):
            c.control(vis_fw)


def iteration_update(vis_fw, state):
    c.control(vis_fw)
    display_network(vis_fw, state)
    delay(vis_fw)
    while vis_fw['pause']:
        pause_loop(vis_fw, state)
    pygame.display.update()
    while vis_fw['instructions']:
        ins_vis.display_instructions(vis_fw)
    if vis_fw['instructions_displayed']:
        batch_update(vis_fw, state)
        vis_fw['instructions_displayed'] = False
    return vis_fw
