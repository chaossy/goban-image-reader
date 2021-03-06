import numpy as np
import os
import h5py
from config import IMAGE_DATASET_NAME, BOARD_DATASET_NAME, proj_path, IMAGE_CHANNELS, TRAIN_IMAGE_SIZE, BOARD_SIZE, BATCH_SIZE
import argparse
import sys


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
    from config import train_dataset_path
    with h5py.File(train_dataset_path, 'r') as f:
        image_dset, board_dset = f[IMAGE_DATASET_NAME], f[BOARD_DATASET_NAME]
        dset_size = np.shape(image_dset)[0]
        assert dset_size == np.shape(board_dset)[0]
        val_dset_size = int(dset_size * 0.1)
        train_dset_size = dset_size - val_dset_size
        chunk_size = 512 * 64

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


def evaluate(dataset_path, model_path):
    model = PZSZModel(model_path)
    with h5py.File(dataset_path, 'r') as f:
        images, boards = f[IMAGE_DATASET_NAME][:], f[BOARD_DATASET_NAME][:]
    loss, acc_stone, acc_board = model.evaluate(images, boards)
    print('loss: {}, acc_stone: {}, acc_board: {}'.format(loss, acc_stone, acc_board))


def predict(image_file_paths, model_path):
    model = PZSZModel(model_path)
    from PIL import Image, ImageOps
    from board import Board
    board = Board()
    inputs = np.zeros((0, IMAGE_CHANNELS, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE), dtype=np.uint8)
    for image_file_path in image_file_paths:
        im = Image.open(image_file_path)
        size = (TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE)
        im.thumbnail(size, Image.ANTIALIAS)
        im = ImageOps.fit(im, size, Image.ANTIALIAS)
        if IMAGE_CHANNELS == 1:
            im = im.convert('L')
        im_array = np.array(im)
        if IMAGE_CHANNELS == 3:
            im_array = np.swapaxes(im_array, 0, 2)
            im_array = np.swapaxes(im_array, 1, 2)
            im_array = im_array[:3, ...]
        else:
            im_array = np.expand_dims(im_array, axis=0)
        im_array = np.expand_dims(im_array, axis=0)
        inputs = np.concatenate((inputs, im_array))
    output = model.predict(inputs)
    assert np.shape(output)[0] == len(image_file_paths)
    for i, image_file_path in enumerate(image_file_paths):
        print('{}:'.format(image_file_path))
        board.board = output[i]
        board.pretty_board()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", action='store_true')
    parser.add_argument("--evaluate_real", "-er", action='store_true')
    parser.add_argument("--evaluate_syn", "-es", action='store_true')
    parser.add_argument("--predict", "-p", nargs="+")
    parser.add_argument("--model", "-m", help="Path of the model file for evaluating or predicting", default=os.path.join(proj_path, 'best_model.hdf5'), type=str)
    args = parser.parse_args(sys.argv[1:] if len(sys.argv) > 1 else ['-h'])
    if not os.path.isfile(args.model):
        raise ValueError('{} not found'.format(args.model))
    if args.train + args.evaluate_real + args.evaluate_syn + (0 if args.predict is None else (1 if len(args.predict) > 0 else 0)) != 1:
        raise ValueError('Choose one of [--train --evaluate_real --evaluate_syn --predict]')
    from model import PZSZModel
    from config import real_test_dataset_path, syn_test_dataset_path
    if args.train:
        train()
    elif args.evaluate_real:
        evaluate(real_test_dataset_path, model_path=args.model)
    elif args.evaluate_syn:
        evaluate(syn_test_dataset_path, model_path=args.model)
    elif args.predict:
        image_file_paths = []
        for path in args.predict:
            if os.path.isfile(path):
                image_file_paths.append(path)
            else:
                print('{} not found'.format(path))
        predict(image_file_paths, args.model)
