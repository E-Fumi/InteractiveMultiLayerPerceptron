# MLP architecture and hyper-parameters, feel free to tweak either

network_parameters = {

    # layers can be added or removed and can have anything between 1 and 32 neurons
    'layers': ['input layer', 16, 16, 'output layer'],

    # random shapes can be added to the dataset in order to teach the network to discern between digits and non-digits
    'add_random': False,

    # number of epochs
    'epochs': 5,

    # learning rate, must be a float value
    'learning_rate': 0.01,

    # currently supported activation functions are sigmoid and relu
    'activation': 'relu',

    # currently supported weight initializations are standard uniform, Xavier, and He
    'weight_initialization': 'He',

    # has to be a multiple of nr of neurons in output layer: 10 without random shapes, or 11 with random shapes
    'batch_size': 110,

    # nothing fancy, just translation by one pixel in each direction
    'data_augmentation': True,

    # use correction terms to enhance network stability
    'correction': True

}
