import pygame
import control as c


def add_instructions_visual_parameters(vis_fw):
    vis_fw['question_mark_coordinates'] = [vis_fw['screen_width'] - 3 * vis_fw['padding'] // 2,
                                           vis_fw['padding']]
    vis_fw['instructions_displayed'] = False
    vis_fw['exit_instructions'] = False
    vis_fw['instructions_font'] = pygame.font.Font('JetBrainsMono-ExtraLight.ttf', vis_fw['padding'] // 2)
    vis_fw['instructions_list'] = get_instructions_list(vis_fw)


def display_question_mark(vis_fw):
    question_mark = vis_fw['instructions_font'].render('?', False, (255, 255, 255))
    vis_fw['screen'].blit(question_mark, vis_fw['question_mark_coordinates'])


def get_instructions_list(vis_fw):
    instructions_text = [['Epochs'],
                         ['Batches'],
                         ['Input'],
                         ['Accuracy Check'],
                         ['Welcome to my interactive multilayer perceptron project'],
                         ['Feel free to tinker with setup.py'],
                         [' '],
                         ['Hover over the first layer neurons to see their input weights'],
                         ['(you can click for transparency)'],
                         [' '],
                         ['space: pause'],
                         ['+ : increase network delay'],
                         ['- : decrease network delay'],
                         ['m : show more, less significant weights'],
                         ['n : show fewer, more significant weights'],
                         ['w : hide / show weights'],
                         ['b : hide / show biases']]
    instructions_list = render_instructions(instructions_text, vis_fw)
    instructions_list = append_text_coordinates(instructions_list, vis_fw)
    return instructions_list


def render_instructions(instructions_text, vis_fw):
    colour = (255, 255, 255, 255)
    for string in range(len(instructions_text)):
        instructions_text[string][0] = vis_fw['instructions_font'].render(instructions_text[string][0], False, colour)
    return instructions_text


def append_text_coordinates(ins_list, vis_fw):
    keys = ['epoch', 'batches', 'input', 'check']
    for item in range(4):
        ins_list[item].append(center_text(vis_fw, ins_list, item, keys[item]))
    x, y = (vis_fw['screen_width'] - ins_list[4][0].get_width()) // 2, 2 * vis_fw['side_padding']
    ins_list[4].append([x, y])
    for item in range(5, len(ins_list)):
        ins_list[item].append([ins_list[item - 1][1][0], ins_list[item - 1][1][1] + vis_fw['padding']])
    return ins_list


def center_text(vis_fw, ins_list, item, key):
    x = vis_fw['side_padding'] + ((vis_fw['panel_width'] - ins_list[item][0].get_width()) // 2)
    y = vis_fw[key + '_coordinates'][0][1] + ((vis_fw[key + '_height'] - ins_list[item][0].get_height()) // 2)
    return [x, y]


def display_text(vis_fw, surface):
    for item in range(len(vis_fw['instructions_list'])):
        surface.blit(vis_fw['instructions_list'][item][0], vis_fw['instructions_list'][item][1])


def display_instructions(vis_fw):
    if not vis_fw['instructions_displayed']:
        surface = pygame.Surface((vis_fw['screen_width'], vis_fw['screen_height']), pygame.SRCALPHA)
        pygame.draw.rect(surface, (0, 0, 0, 180), (0, 0, vis_fw['screen_width'], vis_fw['screen_height']))
        display_text(vis_fw, surface)
        vis_fw['screen'].blit(surface, (0, 0))
        vis_fw['instructions_displayed'] = True
        vis_fw['exit_instructions'] = True
        pygame.display.update()
    c.control(vis_fw)
    return vis_fw
