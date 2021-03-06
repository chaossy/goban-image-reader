from config import TRAIN_IMAGE_SIZE, BOARD_SIZE, SYN_IMAGE_SIZE, IMAGE_CHANNELS, BUFFER_DATA_SIZE, MAX_BUFFER_DATA_SIZE, proj_path
import h5py
import numpy as np
import os
from sgfmill import sgf
from board import Board
from PIL import Image, ImageDraw
import multiprocessing as mp
import ctypes as c
import time
import argparse
import sys
import random


#TODO: tensorboard
#TODO: mem usage
#TODO: fix: python process not quit


def _augment_data(data, dtype=None):
    shape = np.shape(data)
    dim = len(shape)
    assert dim >= 2
    out = np.empty((8, *np.shape(data)), dtype=(np.float64 if dtype is None else data.dtype))
    data_flip = np.flip(data, dim - 2)
    for i in range(4):
        out[i * 2] = np.rot90(data, k=i, axes=(dim - 2, dim - 1))
        out[i * 2 + 1] = np.rot90(data_flip, k=i, axes=(dim - 2, dim - 1))
    return out


def _get_image_data_from_image(im, augment=False):
    im.thumbnail((TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE))
    if IMAGE_CHANNELS == 1:
        im = im.convert('L')
    im_array = np.array(im)
    if IMAGE_CHANNELS == 3:
        im_array = np.swapaxes(im_array, 0, 2)
        im_array = np.swapaxes(im_array, 1, 2)
        im_array = im_array[:3, ...]
    else:
        im_array = np.expand_dims(im_array, axis=0)
    if augment:
        data = _augment_data(im_array, dtype=np.uint8)
    else:
        data = np.expand_dims(im_array, axis=0)
    # for i in range(8):
    #     im_array = data[i]
    #     im_array = np.swapaxes(im_array, 1, 2)
    #     im_array = np.swapaxes(im_array, 0, 2)
    #     xxx = Image.fromarray(im_array)
    #     xxx.show()
    return data


def _get_board_from_sgf(filepath, augment=False):
    board = Board()
    with open(filepath, "rb") as f:
        game = sgf.Sgf_game.from_bytes(f.read())
        if game.get_size() != BOARD_SIZE:
            return
        for i, node in enumerate(game.get_main_sequence()):
            color, move = node.get_move()
            if i == 0:
                bp, wp, ep = game.get_root().get_setup_stones()
                if len(wp) > 0 or len(ep) > 0:
                    print('error({}): illegal handicap'.format(filepath))
                    return None
                if len(bp) > 0:
                    board.add_handicap_stones({(BOARD_SIZE - 1 - y, x) for (y, x) in bp})
            else:
                try:
                    if move is None:  # 没有区分resign和pass
                        board.play(-1, -1)
                    else:
                        board.play(BOARD_SIZE - 1 - move[0], move[1])
                except Board.IllegalMove:
                    print('error({}): illegal move'.format(filepath))
                    return None
        planes = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.bool)
        planes[0, board.board == Board.BLACK] = 1
        planes[1, board.board == Board.WHITE] = 1
        planes[2, board.board == Board.EMPTY] = 1
        if augment:
            planes = _augment_data(planes, dtype=np.bool)
        else:
            planes = np.expand_dims(planes, axis=0)
        # for i in range(8):
        #     b = Board()
        #     b.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        #     b.board[planes[i, 0] == 1] = 1
        #     b.board[planes[i, 1] == 1] = -1
        #     b.pretty_board()
        flat_planes = np.reshape(planes, (np.shape(planes)[0], BOARD_SIZE * BOARD_SIZE * 3), order='F')
        return flat_planes, board.board


def _rand_bool():
    return np.random.randint(0, 2) == 0


def _rand_color(lr, hr, lg, hg, lb, hb, a):
    return np.random.randint(lr, hr), np.random.randint(lg, hg), np.random.randint(lb, hb), a


def _rand_white(l, h, a, rand_component=10):
    c = np.random.randint(l, h)
    return c, c, c, a

# class MyGaussianBlur(ImageFilter.Filter):
#     name = "GaussianBlur"
#
#     def __init__(self, radius=2, bounds=None):
#         self.radius = radius
#         self.bounds = bounds
#
#     def filter(self, image):
#         if self.bounds:
#             clips = image.crop(self.bounds).gaussian_blur(self.radius)
#             image.paste(clips, self.bounds)
#             return image
#         else:
#             return image.gaussian_blur(self.radius)


def _find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def _synthesize_board_image(board, save_image_path=None):
    bg_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256), 0)
    im = Image.new('RGBA', (SYN_IMAGE_SIZE, SYN_IMAGE_SIZE), color=bg_color)
    draw = ImageDraw.Draw(im)

    board_margin = np.random.randint(10, 100)
    board_size = SYN_IMAGE_SIZE - board_margin * 2
    board_inner_margin = np.random.randint(50, 80)
    inner_board_size = board_size - board_inner_margin * 2
    square_size = inner_board_size / (BOARD_SIZE - 1)
    stone_size = np.random.randint(square_size - 6, square_size - 2)
    board_offset_x, board_offset_y = np.random.randint(-24, 25), np.random.randint(-24, 25)
    board_start_x, board_start_y = board_margin + board_offset_x, board_margin + board_offset_y
    board_end_x, board_end_y = board_start_x + board_size, board_start_y + board_size
    inner_board_start_x, inner_board_start_y = board_start_x + board_inner_margin, board_start_y + board_inner_margin
    inner_board_end_x, inner_board_end_y = inner_board_start_x + inner_board_size, inner_board_start_y + inner_board_size
    star_size = np.random.randint(square_size / 5, square_size / 3)

    ''' board '''
    board_index = np.random.randint(1, 7)
    path = os.path.join(proj_path, 'assets', 'goban{}.png'.format(board_index))
    im_board = Image.open(path)
    im_board = im_board.resize((board_end_x - board_start_x, board_end_y - board_start_y))
    im_board = im_board.rotate(90 * np.random.randint(0, 4))
    im.paste(im_board, (board_start_x, board_start_y, board_end_x, board_end_y))

    ''' line '''
    line_and_star_color = _rand_color(0, 80, 0, 30, 0, 20, 255)
    for i in range(BOARD_SIZE):
        draw.line((inner_board_start_x, inner_board_start_y + i * square_size, inner_board_end_x, inner_board_start_y + i * square_size), fill=line_and_star_color, width=1)
        draw.line((inner_board_start_x + i * square_size, inner_board_start_y, inner_board_start_x + i * square_size, inner_board_end_y), fill=line_and_star_color, width=1)

    ''' star '''
    star_points = [(3, 3), (3, 15), (15, 3), (15, 15), (9, 9)]
    if _rand_bool():
        star_points.extend([(3, 9), (15, 9), (9, 3), (9, 15)])
    for ix, iy in star_points:
        x = inner_board_start_x + ix * square_size - star_size / 2
        y = inner_board_start_y + iy * square_size - star_size / 2
        draw.chord((x, y, x + star_size, y + star_size), fill=line_and_star_color, start=0, end=360)

    ''' stone '''
    b_index = np.random.randint(1, 9)
    path = os.path.join(proj_path, 'assets', 'b{}.png'.format(b_index))
    im_black_stone = Image.open(path)
    im_black_stone = im_black_stone.resize((stone_size, stone_size))
    im_black_stone = im_black_stone.rotate(np.random.randint(0, 360))
    w_index = np.random.randint(1, 9)
    path = os.path.join(proj_path, 'assets', 'w{}.png'.format(w_index))
    im_white_stone = Image.open(path)
    im_white_stone = im_white_stone.resize((stone_size, stone_size))
    im_white_stone = im_white_stone.rotate(np.random.randint(0, 360))
    # black_stone_color = _rand_white(0, 80, 255)
    # white_stone_color = _rand_white(200, 250, 255)
    for iy in range(BOARD_SIZE):
        for ix in range(BOARD_SIZE):
            # color = None
            # if board[iy, ix] == 1:
            #     color = black_stone_color
            # elif board[iy, ix] == -1:
            #     color = white_stone_color
            # x = inner_board_start_x + square_size * ix - stone_size / 2 + np.random.randint(-3, 3)
            # y = inner_board_start_y + square_size * iy - stone_size / 2 + np.random.randint(-3, 3)
            # if color is not None:
            #     draw.chord((x, y, x + stone_size, y + stone_size), fill=color, start=0, end=360)
            x = inner_board_start_x + square_size * ix - stone_size / 2 + np.random.randint(-3, 3)
            y = inner_board_start_y + square_size * iy - stone_size / 2 + np.random.randint(-3, 3)
            pos = (int(x), int(y), int(x + stone_size), int(y + stone_size))
            if board[iy, ix] == 1:
                im.paste(im_black_stone, pos, im_black_stone)
            elif board[iy, ix] == -1:
                im.paste(im_white_stone, pos, im_white_stone)

    ''' rotate board '''
    im_bg = Image.new('RGBA', im.size, color=bg_color)
    # im = im.rotate(np.random.randint(-5, 6), resample=Image.BICUBIC)
    coeffs_bound = 20
    coeffs = _find_coeffs([(np.random.randint(-coeffs_bound, coeffs_bound), np.random.randint(-coeffs_bound, coeffs_bound)),
                           (board_size + np.random.randint(-coeffs_bound, coeffs_bound), np.random.randint(-coeffs_bound, coeffs_bound)),
                           (board_size + np.random.randint(-coeffs_bound, coeffs_bound), board_size + np.random.randint(-coeffs_bound, coeffs_bound)),
                           (np.random.randint(-coeffs_bound, coeffs_bound), board_size + np.random.randint(-coeffs_bound, coeffs_bound))],
                          [(0, 0), (board_size, 0), (board_size, board_size), (0, board_size)])
    im = im.transform((SYN_IMAGE_SIZE, SYN_IMAGE_SIZE), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    im = Image.composite(im, im_bg, im).convert('RGB')

    ''' gaussian noise '''
    mean = np.random.randint(-5, 5)
    sigma = np.random.randint(0, 10)
    gauss = np.random.normal(mean, sigma, (SYN_IMAGE_SIZE, SYN_IMAGE_SIZE, IMAGE_CHANNELS))
    gauss = gauss.reshape(SYN_IMAGE_SIZE, SYN_IMAGE_SIZE, IMAGE_CHANNELS)
    im_array = np.array(im, dtype=np.int16) + gauss.astype(np.int16)
    im_array = np.clip(im_array, 0, 255).astype(np.uint8)
    im = Image.fromarray(im_array)

    if IMAGE_CHANNELS == 1:
        im = im.convert('L')
    if save_image_path is not None:
        im.save(save_image_path)

    return im


_c_board_list = mp.Array(c.c_bool, MAX_BUFFER_DATA_SIZE * 3 * BOARD_SIZE * BOARD_SIZE, lock=False)
_c_image_list = mp.Array(c.c_uint8, MAX_BUFFER_DATA_SIZE * IMAGE_CHANNELS * TRAIN_IMAGE_SIZE * TRAIN_IMAGE_SIZE, lock=False)
_board_list = np.frombuffer(_c_board_list, dtype=np.bool)
_board_list = np.reshape(_board_list, (MAX_BUFFER_DATA_SIZE, 3 * BOARD_SIZE * BOARD_SIZE))
_image_list = np.frombuffer(_c_image_list, dtype=np.uint8)
_image_list = np.reshape(_image_list, (MAX_BUFFER_DATA_SIZE, IMAGE_CHANNELS, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE))
_data_num = mp.Value('i', 0, lock=False)
_data_lock = mp.RLock()


def _create_dataset_worker(param):
    try:
        augment = param['augment']
        filenames = param['files']
        output_image_dir = param['output_image_dir']

        for i, filename in enumerate(filenames):
            # sgf_filepath = os.path.join(sgf_dir, filename + '.sgf')
            if filename['image'] is None:
                board_data, board = _get_board_from_sgf(filename['sgf'], augment=False)
                assert board_data is not None
                names = os.path.basename(filename['sgf']).split('.')
                filename = names[0]
                save_image_path = None if output_image_dir is None else os.path.join(output_image_dir, '{}.png'.format(filename))
                image = _synthesize_board_image(board, save_image_path=save_image_path)
                image_data = _get_image_data_from_image(image, augment=False)
                assert image_data is not None
            else:
                board_data, board = _get_board_from_sgf(filename['sgf'], augment=augment)
                assert board_data is not None
                # image_filepath = os.path.join(image_dir, filename + '.png')
                image = Image.open(filename['image'])
                image.thumbnail((TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE))
                image_data = _get_image_data_from_image(image, augment=augment)
                assert image_data is not None
            batch_size = np.shape(board_data)[0]
            with _data_lock:
                _image_list[_data_num.value: _data_num.value + batch_size] = image_data
                _board_list[_data_num.value: _data_num.value + batch_size] = board_data
                _data_num.value += batch_size
    except Exception as e:
        print(e)


def _get_sgf_filenames(sgf_dir, image_dir=None):
    filenames = []
    if sgf_dir is None:
        return filenames
    for _, _, sgf_filenames in os.walk(sgf_dir):
        # total = len(sgf_filenames)
        for i, sgf_filename in enumerate(sgf_filenames):
            # print('{0}/{1} {2}                   '.format(i + 1, total, sgf_filename), end='\r')
            names = os.path.basename(sgf_filename).split('.')
            if names[1] != 'sgf':
                continue
            filename = names[0]
            sgf_filepath = os.path.join(sgf_dir, sgf_filename)
            if image_dir is None:
                filenames.append({'sgf': sgf_filepath, 'image': None})
            else:
                image_filepath = os.path.join(image_dir, filename + '.png')
                if os.path.isfile(image_filepath):
                    filenames.append({'sgf': sgf_filepath, 'image': image_filepath})
                else:
                    print('{}/{}.png not found'.format(image_dir, filename))
    return filenames


def _create_dataset(dataset_path, syn_sgf_dir, real_sgf_dir, real_image_dir, output_image_dir, augment, process_count):
    from config import dataset_dir, IMAGE_DATASET_NAME, BOARD_DATASET_NAME

    if not os.path.isdir(dataset_dir):
        os.mkdir(dataset_dir)
    if os.path.isfile(dataset_path):
        os.remove(dataset_path)

    syn_filenames = _get_sgf_filenames(syn_sgf_dir)
    real_filenames = _get_sgf_filenames(real_sgf_dir, real_image_dir)
    filenames = syn_filenames + real_filenames
    if len(filenames) == 0:
        return

    random.shuffle(filenames)

    with h5py.File(dataset_path, 'a') as f:
        image_set = f.create_dataset(IMAGE_DATASET_NAME, shape=(0, IMAGE_CHANNELS, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE), maxshape=(None, IMAGE_CHANNELS, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE), dtype=np.uint8)
        board_set = f.create_dataset(BOARD_DATASET_NAME, shape=(0, BOARD_SIZE * BOARD_SIZE * 3), maxshape=(None, BOARD_SIZE * BOARD_SIZE * 3), dtype=np.bool)
        filecount = len(filenames)
        filecount_per_process = filecount // process_count
        dataset_count = len(syn_filenames) + len(real_filenames) * (8 if augment else 1)

        params_list = []
        for i in range(process_count):
            params = {}
            if i < process_count - 1:
                params['files'] = filenames[filecount_per_process * i: filecount_per_process * (i + 1)].copy()
            else:
                params['files'] = filenames[filecount_per_process * i: filecount].copy()
            params['augment'] = augment
            params['output_image_dir'] = output_image_dir
            params_list.append(params)

        pool = mp.Pool()
        pool.map_async(_create_dataset_worker, params_list)

        while True:
            with _data_lock:
                count = image_set.shape[0] + _data_num.value
                if _data_num.value >= BUFFER_DATA_SIZE or count == dataset_count:
                    image_set.resize(image_set.shape[0] + _data_num.value, axis=0)
                    image_set[-_data_num.value:] = _image_list[:_data_num.value]
                    board_set.resize(board_set.shape[0] + _data_num.value, axis=0)
                    board_set[-_data_num.value:] = _board_list[:_data_num.value]
                    print('{0}/{1}   '.format(count, dataset_count))
                    _data_num.value = 0
                if count == dataset_count:
                    break
            time.sleep(1)


if __name__ == '__main__':
    from config import syn_train_sgf_dir, syn_test_sgf_dir, real_train_sgf_dir, real_train_image_dir, real_test_sgf_dir, real_test_image_dir, syn_output_image_dir, train_dataset_path, syn_test_dataset_path, real_test_dataset_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", nargs="+", help='Specify the dataset you want to create, legal choices includes [train syn_test real_test]')
    parser.add_argument("--syn_train_sgf_dir", default=syn_train_sgf_dir, type=str)
    parser.add_argument("--syn_test_sgf_dir", default=syn_test_sgf_dir, type=str)
    parser.add_argument("--output_syn_image", "-o", help='Output synthetic goban images to datasets/syn_image. Useful if you are curious about what the generated image looks like. BEWARE this will eat up a lot of disk space if you have a lot of sgf files', action='store_true')
    parser.add_argument("--real_train_sgf_dir", default=real_train_sgf_dir, type=str)
    parser.add_argument("--real_train_image_dir", default=real_train_image_dir, type=str)
    parser.add_argument("--real_test_sgf_dir", default=real_test_sgf_dir, type=str)
    parser.add_argument("--real_test_image_dir", default=real_test_image_dir, type=str)
    parser.add_argument("--no_augmentation", "-n", help='Don\'t augment (rotate and flip) real datas', action='store_true')
    parser.add_argument("--process", "-p", default=3, help='Number of processes to spawn when creating dataset', type=int)
    args = parser.parse_args(sys.argv[1:] if len(sys.argv) > 1 else ['-h'])

    if args.dataset is None:
        raise ValueError('Please specify the dataset you want to create, legal choices includes [train syn_test real_test]')
    if not set(args.dataset).issubset({'real_test', 'syn_test', 'train'}):
        raise ValueError('{} not supported, legal choices includes [train syn_test real_test]'.format(args.dataset))

    output_image_dir = syn_output_image_dir if args.output_syn_image else None
    if output_image_dir is not None and not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)

    if 'real_test' in args.dataset:
        print('Creating real test dataset...')
        if not os.path.isdir(args.real_test_sgf_dir):
            raise ValueError('{} not found'.format(args.real_test_sgf_dir))
        if not os.path.isdir(args.real_test_image_dir):
            raise ValueError('{} not found'.format(args.real_test_image_dir))
        _create_dataset(real_test_dataset_path, None, args.real_test_sgf_dir, args.real_test_image_dir, None, not args.no_augmentation, args.process)

    if 'syn_test' in args.dataset:
        print('Creating synthetic test dataset...')
        if not os.path.isdir(args.syn_test_sgf_dir):
            raise ValueError('{} not found'.format(args.syn_test_sgf_dir))
        _create_dataset(syn_test_dataset_path, args.syn_test_sgf_dir, None, None, output_image_dir, False, args.process)

    if 'train' in args.dataset:
        print('Creating training dataset...')
        if not os.path.isdir(args.syn_train_sgf_dir) and not (os.path.isdir(args.real_train_sgf_dir) or os.path.isdir(args.real_train_image_dir)):
            raise ValueError('Input dir invalid')
        _create_dataset(train_dataset_path, args.syn_train_sgf_dir, args.real_train_sgf_dir, args.real_train_image_dir, output_image_dir, not args.no_augmentation, args.process)
