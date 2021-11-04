import network_functions as net
import data_handling as dh
import visualization as vis


class Network:

    def __init__(self, parameters):
        self.params = parameters
        self.net_state = net.initialize_network_state(self.params)
        self.dataset = dh.assemble_dataset('train', self.params['add_random'])
        self.params['batches_per_epoch'] = dh.determine_number_of_batches(parameters, self.dataset)
        self.visual_framework = vis.initialize(self.params)

    def run(self):
        for epoch in range(self.params['epochs']):
            self.net_state['current_epoch'] = epoch
            dataset = dh.shuffle_dataset(self.dataset)

            for batch_number in range(self.params['batches_per_epoch']):
                self.net_state['current_batch'] = batch_number
                batch = dh.assemble_batch(dataset, self.params, self.net_state)
                vis.batch_update(self.visual_framework, self.net_state)

                for iteration in range(self.params['batch_size']):
                    self.net_state['current_iteration'] = iteration
                    self.net_state = net.train(self.params, self.net_state, batch)
                    self.visual_framework = vis.iteration_update(self.visual_framework, self.net_state)

                self.net_state = net.learn(self.params, self.net_state)

        dataset = dh.assemble_dataset('test', self.params['add_random'])
        net.test(self.params, self.net_state, dataset)
