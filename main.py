import Network as Neural
import setup
import utils

if __name__ == '__main__':
    parameters = utils.initialize_parameters(setup.network_parameters)
    MLP = Neural.Network(parameters)
    MLP.run()
