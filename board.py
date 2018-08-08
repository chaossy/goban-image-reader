from config import BOARD_SIZE
import numpy as np


class String:
    def __init__(self):
        self.points = set()
        self.liberty_points = set()
        self.opponent_points = set()
        self.color = Board.EMPTY

    def copy(self):
        s = String()
        s.points = self.points.copy()
        s.liberty_points = self.liberty_points.copy()
        s.opponent_points = self.opponent_points.copy()
        s.color = self.color
        return s


class Board:
    BLACK = 1
    WHITE = -1
    EMPTY = 0

    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.history_board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.stone_age_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int16)
        self.string_board = np.empty((BOARD_SIZE, BOARD_SIZE), dtype=np.object)
        self.strings = set()
        self.step = 0
        self._is_last_step_pass = False
        self._passes_white = 0
        self._passes_black = 0
        self.is_game_over = False
        self._has_handicaps = False

    def reset(self):
        self.board.fill(self.EMPTY)
        self.history_board.fill(self.EMPTY)
        self.stone_age_board.fill(0)
        self.string_board = np.empty((BOARD_SIZE, BOARD_SIZE), dtype=np.object)
        self.strings = set()
        self.step = 0
        self._is_last_step_pass = False
        self._passes_white = 0
        self._passes_black = 0
        self.is_game_over = False
        self._has_handicaps = False

    def save_state(self):
        saved_board = self.board
        saved_history_board = self.history_board
        saved_stone_age_board = self.stone_age_board
        saved_string_board = self.string_board
        saved_strings = self.strings
        saved_step = self.step
        saved_is_last_step_pass = self._is_last_step_pass
        saved_passes_black = self._passes_black
        saved_passes_white = self._passes_white
        saved_is_game_over = self.is_game_over
        self.board = self.board.copy()
        self.history_board = self.history_board.copy()
        self.stone_age_board = self.stone_age_board.copy()
        self.string_board = np.empty((BOARD_SIZE, BOARD_SIZE), dtype=np.object)
        self.strings = {string.copy() for string in saved_strings}
        for string in self.strings:
            for y, x in string.points:
                self.string_board[y, x] = string
        return saved_board, saved_history_board, saved_stone_age_board, saved_string_board, saved_strings, saved_step, saved_is_last_step_pass, saved_passes_black, saved_passes_white, saved_is_game_over

    def load_state(self, state):
        self.board, self.history_board, self.stone_age_board, self.string_board, self.strings, self.step, self._is_last_step_pass, self._passes_black, self._passes_white, self.is_game_over = state

    def can_play_at_vertex(self, vertex):
        return self.can_play(*self.vertex_to_point(vertex))

    def can_play(self, y, x):
        state = self.save_state()
        can_play = True
        try:
            self.play(y, x)
        except self.IllegalMove:
            can_play = False
        finally:
            self.load_state(state)
            return can_play

    def play_at_vertex(self, vertex):
        return self.play(*self.vertex_to_point(vertex))

    def play_pass(self):
        self.play(-1, -1)

    def add_handicap_stones(self, points):
        for y, x in points:
            self.play(y, x, self.BLACK)
        if len(points) > 0:
            self._has_handicaps = True
        self.step = 0

    def play(self, y, x, color=None):
        if color is None:
            color = self.next_color()
        is_last_step_pass = self._is_last_step_pass
        if x == -1 and y == -1:
            if color == self.BLACK:
                self._passes_black += 1
            else:
                self._passes_white += 1
            self._is_last_step_pass = True
        elif self.board[y, x] != self.EMPTY:
            raise self.IllegalMove('ILLEGAL_OCCUPIED')
        else:
            self._assert_point(y, x)
            self.board[y, x] = color
            new_string = String()
            new_string.color = color
            new_string.points.add((y, x))
            self.strings.add(new_string)
            self.string_board[y, x] = new_string
            # neighbour_one_liberty_self_strings = set()
            # neighbour_two_liberties_opponent_strings = set()

            for (adj_y, adj_x) in self._get_adjacent_points(y, x):
                neighbour_string = self.string_board[adj_y, adj_x]
                if neighbour_string is None:
                    assert self.board[adj_y, adj_x] == self.EMPTY
                    new_string.liberty_points.add((adj_y, adj_x))
                elif neighbour_string is new_string:
                    continue
                else:
                    # neighbour_liberty = len(neighbour_string.liberty_points)
                    # if neighbour_liberty == 1 and neighbour_string.color == color:
                    #     neighbour_one_liberty_self_strings.add(neighbour_string)
                    # elif neighbour_liberty == 2 and neighbour_string.color == -color:
                    #     neighbour_two_liberties_opponent_strings.add(neighbour_string)

                    if neighbour_string.color == color:
                        new_string.points |= neighbour_string.points
                        neighbour_string.liberty_points.discard((y, x))
                        self.string_board[[point[0] for point in neighbour_string.points], [point[1] for point in neighbour_string.points]] = new_string
                        new_string.liberty_points |= neighbour_string.liberty_points
                        new_string.opponent_points |= neighbour_string.opponent_points
                        self.strings.remove(neighbour_string)
                    else:
                        assert neighbour_string.color == -color
                        new_string.opponent_points.add((adj_y, adj_x))
                        neighbour_string.liberty_points.discard((y, x))
                        neighbour_string.opponent_points.add((y, x))

                        neighbour_string_liberty_count = len(neighbour_string.liberty_points)
                        if neighbour_string_liberty_count == 0:
                            removed_stones = [point[0] for point in neighbour_string.points], [point[1] for point in neighbour_string.points]
                            self.board[removed_stones] = self.EMPTY
                            self.stone_age_board[removed_stones] = 0
                            self.string_board[removed_stones] = None
                            for add_lib_y, add_lib_x in neighbour_string.opponent_points:
                                string = self.string_board[add_lib_y, add_lib_x]
                                new_liberties = string.opponent_points & neighbour_string.points
                                string.liberty_points |= new_liberties
                                string.opponent_points -= new_liberties
                            self.strings.remove(neighbour_string)

            # if len(new_string.liberty_points) == 2 and len(neighbour_one_liberty_self_strings) > 0:
            #     pass # ladder escape
            # for opponent_string in neighbour_two_liberties_opponent_strings:
            #     for y, x in opponent_string.points:
            #         string = self.string_board[y, x]
            #         if len(string.liberty_points) == 1:
            #             # ladder capture
            #             break

            if len(self.string_board[y, x].liberty_points) == 0:
                raise self.IllegalMove('ILLEGAL_SUICIDE')
            if self.board.tolist() in self.history_board.tolist():
                raise self.IllegalMove('ILLEGAL_KO')

            self._is_last_step_pass = False

        self.stone_age_board[np.where(self.board != self.EMPTY)] += 1
        if y != -1 and x != -1:
            self.stone_age_board[y, x] = 1
        self.history_board = np.roll(self.history_board, 1, axis=0)
        self.history_board[0, ...] = self.board.copy()
        self.step += 1

        if is_last_step_pass and x == -1 and y == -1:
            self.is_game_over = True
        # elif self.step >= conf['max_game_step']:
        #     self.is_game_over = True
        else:
            self.is_game_over = False

    def is_ladder_escape(self, y, x):
        state = self.save_state()
        try:
            current_color = self.next_color()
            self.play(y, x)
            escaping_string = self.string_board[y, x]
            liberty = len(escaping_string.liberty_points)
            if liberty >= 3:
                return True
            elif liberty == 1:
                return False
            else:
                assert liberty == 2
                adj_points = self._get_adjacent_points(y, x)
                for adj_escaping_y, adj_escaping_x in adj_points:
                    adj_escaping_string = self.string_board[adj_escaping_y, adj_escaping_x]
                    if adj_escaping_string is not None and adj_escaping_string.color == -current_color and len(adj_escaping_string.liberty_points) == 1:
                        return True
                for liberty_y, liberty_x in escaping_string.liberty_points:
                    assert self.board[liberty_y, liberty_x] == self.EMPTY
                    if self.is_ladder_capture(liberty_y, liberty_x, (y, x)):
                        return False
                return True
        except self.IllegalMove:
            return False
        finally:
            self.load_state(state)

    def is_ladder_capture(self, y, x, target_string_point):
        state = self.save_state()
        try:
            # current_color = self.next_color()
            self.play(y, x)
            # string = self.string_board[y, x]
            # if len(string.liberty_points) == 1:
            #     return False
            target_surrounder_strings = set()
            target_string = self.string_board[target_string_point]
            for opponent_point in target_string.opponent_points:
                target_surrounder_strings.add(self.string_board[opponent_point])
            for target_surrounder_string in target_surrounder_strings:
                if len(target_surrounder_string.liberty_points) == 1:
                    return False
            # for capturing_y, capturing_x in string.liberty_points:
            #     if board.is_ladder_capture(capturing_y, capturing_x, next(iter(string.points))):
            #         planes[0, 44, capturing_y, capturing_x] = 1
            #         break
            assert target_string is not None
            assert len(target_string.liberty_points) == 1
            escaping_y, escaping_x = next(iter(target_string.liberty_points))
            return not self.is_ladder_escape(escaping_y, escaping_x)
        except self.IllegalMove:
            return False
        finally:
            self.load_state(state)

    def is_eye(self, y, x, color):
        if self.board[y, x] != self.EMPTY:
            return False
        for (adj_y, adj_x) in self._get_adjacent_points(y, x):
            if self.board[adj_y, adj_x] != color:
                return False
        return True

    def get_winner(self):
        score_white = np.sum(self.board == self.WHITE)
        score_black = np.sum(self.board == self.BLACK)
        empties = zip(*np.where(self.board == self.EMPTY))
        for y, x in empties:
            if self.is_eye(y, x, self.BLACK):
                score_black += 1
            elif self.is_eye(y, x, self.WHITE):
                score_white += 1
        score_white += 7.5
        # score_white -= self._passes_white
        # score_black -= self._passes_black
        if score_black > score_white:
            winner = self.BLACK
        elif score_white > score_black:
            winner = self.WHITE
        else:
            winner = 0
        return winner

    def next_color(self):
        color = self.color_to_play(self.step)
        if self._has_handicaps:
            color = -color
        return color

    def get_board_strings(self):
        points = zip(*np.where(self.board != Board.EMPTY))
        string_points_list = []
        # string_liberty_list = []
        string_points_checked = set()
        for y, x in points:
            if not (y, x) in string_points_checked:
                string_points = set()
                points_checked = set()
                string_liberty = self.get_string(y, x, string_points, points_checked)
                string_points_checked |= string_points
                string_points_list.append((string_points, string_liberty))
                # string_liberty_list.append(string_liberty)
        return string_points_list#, string_liberty_list

    def get_string(self, y, x, string=None, string_points_checked=None):
        self._assert_point(y, x)
        color = self.board[y, x]
        assert (color != self.EMPTY)
        if string is None:
            string = set()
        if string_points_checked is None:
            string_points_checked = set()
        string_points_checked.add((y, x))
        string.add((y, x))
        string_liberty = 0
        adjacent_points = self._get_adjacent_points(y, x)
        for adj_y, adj_x in adjacent_points:
            if (adj_y, adj_x) not in string_points_checked:
                string_points_checked.add((adj_y, adj_x))
                if self.board[adj_y, adj_x] == self.EMPTY:
                    string_liberty += 1
                elif self.board[adj_y, adj_x] == color:
                    string.add((adj_y, adj_x))
                    string_liberty += self.get_string(adj_y, adj_x, string, string_points_checked)
        return string_liberty

    def pretty_board(self, board=None, should_print=True):
        if board is None:
            board = self.board
        str = '   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8\r\n'
        i = 0
        for y in range(BOARD_SIZE):
            str += '{0:2d} '.format(i)
            for x in range(BOARD_SIZE):
                if board[y, x] == self.EMPTY:
                    str += '· '
                elif board[y, x] == self.BLACK:
                    str += '● '
                elif board[y, x] == self.WHITE:
                    str += '○ '
            str += '\r\n'
            i += 1
        str += '   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8\r\n'
        if should_print:
            print(str)
        return str

    def _get_adjacent_points(self, y, x):
        self._assert_point(y, x)
        result = set()
        if x - 1 >= 0:
            result.add((y, x - 1))
        if x + 1 < BOARD_SIZE:
            result.add((y, x + 1))
        if y - 1 >= 0:
            result.add((y - 1, x))
        if y + 1 < BOARD_SIZE:
            result.add((y + 1, x))
        return result

    @classmethod
    def color_to_play(cls, step):
        return cls.BLACK if step % 2 == 0 else cls.WHITE

    @classmethod
    def point_to_vertex(cls, y, x):
        if y == -1 and x == -1:
            return BOARD_SIZE ** 2
        else:
            cls._assert_point(y, x)
            return y * BOARD_SIZE + x

    @classmethod
    def vertex_to_point(cls, vertex):
        assert vertex >= 0, vertex < BOARD_SIZE ** 2 + 1
        if vertex == BOARD_SIZE ** 2:
            return -1, -1
        else:
            return vertex // BOARD_SIZE, vertex % BOARD_SIZE

    @classmethod
    def _assert_point(cls, y, x):
        assert (x >= 0, x < BOARD_SIZE) and (y >= 0, y < BOARD_SIZE)

    class IllegalMove(Exception):
        pass
