import numpy as np
import os
import h5py
from model import PZSZModel
from config import IMAGE_DATASET_NAME, BOARD_DATASET_NAME, proj_path, IMAGE_CHANNELS, TRAIN_IMAGE_SIZE, BOARD_SIZE, BATCH_SIZE


def _generate_batch_indices(pool, batch_size):
    batch_indices_list = []
    remain = np.ones([np.shape(pool)[0]], dtype=np.bool)
    indices_pool = np.arange(0, np.shape(pool)[0])
    while True:
        remain_count = np.sum(remain)
        if remain_count == 0:
            break
        prob = remain / remain_count
        size = batch_size if remain_count > batch_size else remain_count
        # out = np.random.choice(index_pool, size=size, p=prob, replace=False)
        out = np.random.choice(indices_pool, size=size, p=prob, replace=False)
        remain[out] = False
        indices = sorted(pool[out].tolist())
        batch_indices_list.append(indices)
    return batch_indices_list


def _load_data_from_disk(image_chunk, board_chunk, image_dset, board_dset, start, end, mem_start, segment_count):
    size = (end - start) // segment_count
    for i in range(segment_count):
        if i < segment_count - 1:
            image_chunk[mem_start + size * i: mem_start + size * (i + 1), ...] = image_dset[start + size * i: start + size * (i + 1)]
            board_chunk[mem_start + size * i: mem_start + size * (i + 1), ...] = board_dset[start + size * i: start + size * (i + 1)]
        else:
            image_chunk[mem_start + size * i: mem_start + end - start, ...] = image_dset[start + size * i: end]
            board_chunk[mem_start + size * i: mem_start + end - start, ...] = board_dset[start + size * i: end]


def train():
    from config import syn_training_dataset_path
    # from keras.utils import HDF5Matrix
    # model = PZSZModel()
    # # with h5py.File(syn_training_dataset_path, 'r') as f:
    # #     images, boards = f[IMAGE_DATASET_NAME][:], f[BOARD_DATASET_NAME][:]
    # images = HDF5Matrix(syn_training_dataset_path, IMAGE_DATASET_NAME)
    # boards = HDF5Matrix(syn_training_dataset_path, BOARD_DATASET_NAME)
    # model.train(images, boards)

    with h5py.File(syn_training_dataset_path, 'r') as f:
        image_dset, board_dset = f[IMAGE_DATASET_NAME], f[BOARD_DATASET_NAME]
        dset_size = np.shape(image_dset)[0]
        assert dset_size == np.shape(board_dset)[0]
        val_dset_size = int(dset_size * 0.1)
        train_dset_size = dset_size - val_dset_size
        chunk_size = 512 * 130

        def generator(is_training):
            lower_bound = 0 if is_training else train_dset_size
            upper_bound = train_dset_size if is_training else dset_size
            chunk_count = int(np.ceil(train_dset_size / chunk_size)) if is_training else int(np.ceil(val_dset_size / chunk_size))
            remain_dset_size = (train_dset_size % chunk_size) if is_training else (val_dset_size % chunk_size)

            while True:
                base = np.random.randint(lower_bound, upper_bound)
                start = base
                for chunk_index in range(chunk_count):
                    c_size = remain_dset_size if chunk_index == chunk_count - 1 else chunk_size
                    image_chunk = np.zeros((c_size, IMAGE_CHANNELS, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE), dtype=np.uint8)
                    board_chunk = np.zeros((c_size, 3 * BOARD_SIZE * BOARD_SIZE), dtype=np.bool)
                    # indices_boundries = []
                    size = 0
                    end = start + (chunk_size if chunk_index < chunk_count - 1 else remain_dset_size)
                    if end > upper_bound:
                        # indices_boundries.append((start, upper_bound))
                        _load_data_from_disk(image_chunk, board_chunk, image_dset, board_dset, start, upper_bound, 0, 4)
                        size = upper_bound - start
                        start = lower_bound
                        end -= (upper_bound - lower_bound)
                    if start < base and end > base:
                        end = base
                    # indices_boundries.append((start, end))
                    _load_data_from_disk(image_chunk, board_chunk, image_dset, board_dset, start, end, size, 4)
                    assert np.shape(image_chunk)[0] == np.shape(board_chunk)[0]
                    start = end
                    indices_list = _generate_batch_indices(np.arange(0, np.shape(image_chunk)[0]), BATCH_SIZE)
                    for indices in indices_list:
                        yield image_chunk[indices], board_chunk[indices]
                    image_chunk = None
                    board_chunk = None

        model = PZSZModel(is_training=True)
        model.train_generator(generator, np.ceil(train_dset_size / BATCH_SIZE), val_generator=generator, val_steps_per_epoch=np.ceil(val_dset_size / BATCH_SIZE))


def evaluate(dataset_path, model_path=os.path.join(proj_path, 'best_model.hdf5')):
    model = PZSZModel(model_path)
    with h5py.File(dataset_path, 'r') as f:
        images, boards = f[IMAGE_DATASET_NAME][:], f[BOARD_DATASET_NAME][:]
    loss, acc_stone, acc_board = model.evaluate(images, boards)
    print('loss: {}, acc_stone: {}, acc_board: {}'.format(loss, acc_stone, acc_board))


def predict(model_path=os.path.join(proj_path, 'best_model.hdf5')):
    model = PZSZModel(model_path)
    from PIL import Image
    from board import Board
    board = Board()
    for i in range(417, 418):
        im = Image.open(os.path.join(proj_path, 'input', 'real_image', '{}.png'.format(str(i))))
        im.thumbnail((TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE))
        if IMAGE_CHANNELS == 1:
            im = im.convert('L')
        im_array = np.array(im)
        Image.fromarray(im_array).show()
        if IMAGE_CHANNELS == 3:
            im_array = np.swapaxes(im_array, 0, 2)
            im_array = np.swapaxes(im_array, 1, 2)
            # im_array_list[i] = im_array
            output = model.predict(np.expand_dims(im_array[:3, ...], axis=0))
            board.board = output[0]
            print('{}:'.format(i))
        else:
            im_array = np.expand_dims(im_array, axis=0)
            output = model.predict(np.expand_dims(im_array, axis=0))
            board.board = output[0]
            print('{}:'.format(i))
        board.pretty_board()


if __name__ == '__main__':
    # train()

    # from config import real_test_dataset_path
    # evaluate(real_test_dataset_path, model_path=os.path.join(proj_path, '51.hdf5'))

    predict()
