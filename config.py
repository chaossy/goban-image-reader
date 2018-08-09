import os

''' convertor '''
TRAIN_IMAGE_SIZE = 224
IMAGE_CHANNELS = 3
BOARD_SIZE = 19
SYN_IMAGE_SIZE = 1024
IMAGE_DATASET_NAME = 'image'
BOARD_DATASET_NAME = 'board'
BUFFER_DATA_SIZE = 1000
MAX_BUFFER_DATA_SIZE = BUFFER_DATA_SIZE + 1000

''' path '''
def _dataset_name(base):
    return base + ('_rgb' if IMAGE_CHANNELS == 3 else '_gray') + '.hdf5'
proj_path = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(proj_path, 'model')
log_file_path = os.path.join(proj_path, 'log')
syn_sgf_dir = os.path.join(proj_path, 'syn_sgf')
real_sgf_dir = os.path.join(proj_path, 'real_sgf')
real_image_dir = os.path.join(proj_path, 'real_image')

dataset_dir = os.path.join(proj_path, 'dataset')
syn_output_image_dir = os.path.join(dataset_dir, 'syn_image')
syn_training_dataset_path = os.path.join(dataset_dir, _dataset_name('syn_training_data'))
syn_test_dataset_path = os.path.join(dataset_dir, _dataset_name('syn_test_data'))
real_training_dataset_path = os.path.join(dataset_dir, _dataset_name('real_training_data'))
real_test_dataset_path = os.path.join(dataset_dir, _dataset_name('real_test_data'))

''' training '''
LR = 0.004#0.0004
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_EPSILON = 0.01
LR_REDUCE_PATIENCE = 5
MODEL_SAVE_PERIOD = 3
MOMENTUM = 0.9
EPOCHS = 100
REG = 0.00001
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
