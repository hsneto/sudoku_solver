from copy import deepcopy
import matplotlib.pyplot as plt

class Solver(object):
  """
  Algorithm to solve sudoku from a list
  """
  
  def __init__(self):
    pass

  def load_sudoku(self, puzzle):
    self._puzzle = puzzle
    self.puzzle = self._read_puzzle(puzzle)
    self._puzzle_answered = False
    
  def _read_puzzle(self, puzzle):
    """
    Replace 0 with set of possible values
    """

    grid = deepcopy(puzzle)
    for i in range(9):
      for j in range(9):
        if grid[i][j] == 0:
          grid[i][j] = set(range(1,10))
    return grid

  def _get_row_values(self, puzzle, idx):
    """
    Get taken values in row
    """
    row = puzzle[idx]
    return set([x for x in row if not isinstance(x, set)])

  def _get_column_values(self, puzzle, idx):
    """
    Get taken values in column
    """
    column = [puzzle[x][idx] for x in range(9)]
    return set([x for x in column if not isinstance(x, set)])

  def _get_block_values(self, puzzle, idx_x, idx_y):
    """
    Get taken values in cell block
    """
    idx_x, idx_y = idx_x // 3, idx_y // 3
    values = set()
    for i in range(3*idx_x, 3*idx_x+3):
      for j in range(3*idx_y, 3*idx_y+3):
        cell = puzzle[i][j]
        if not isinstance(cell, set):
          values.add(cell)

    return values

  def _remove_values(self, puzzle, idx_x, idx_y):
    """
    Remove taken values from possible values for a cell
    """

    values = self._get_row_values(puzzle, idx_x)
    values = values.union(self._get_column_values(puzzle, idx_y))
    values = values.union(self._get_block_values(puzzle, idx_x, idx_y))

    puzzle[idx_x][idx_y] -= values

  def _propagate_step(self, puzzle):
    new_units = False

    for i in range(9):
      for j in range(9):
        if isinstance(puzzle[i][j], set):
          self._remove_values(puzzle, i, j)

          if len(puzzle[i][j]) == 1:
            puzzle[i][j] = puzzle[i][j].pop()
            new_units = True

          elif len(puzzle[i][j]) == 0:
            return False, None

    return True, new_units

  def _propagate(self, puzzle):
    """
    Propagate until we reach a fixpoint
    """
    while True:
      solvable, new_unit = self._propagate_step(puzzle)

      if not solvable:
        return False

      if not new_unit:
        return True

  def _done(self, puzzle):
    """
    Check if the puzzle is solved
    """
    for row in puzzle:
      for cell in row:
        if isinstance(cell, set):
          return False
    return True

  def solve(self, puzzle=None):
    """
    Solve Sudoku
    """
    
    if puzzle is None:
      puzzle = self.puzzle

    solvable = self._propagate(puzzle)

    if not solvable:
      return None
    if self._done(puzzle):
      self.puzzle = puzzle
      self._puzzle_answered = True
      return puzzle

    for i in range(9):
      for j in range(9):
        cell = puzzle[i][j]

        if isinstance(cell, set):
          for value in cell:
            new_state = deepcopy(puzzle)
            new_state[i][j] = value
            solved = self.solve(new_state)
            if solved is not None:
              return solved
          return None

  def show(self, color=(0.0,0.75,0.0), filename=None):
    """
    Plot solved sudoku.

    Args:
      color --> RGB values to plot answer from the solved sudoku. Default color is \
      Green.
      filename --> filename and path to save the solved sudoku. If None, \
      Sudoku will not be saved!
    """

    # check if the puzzle was solved
    if not self._puzzle_answered:
      print('Puzzle is not solved!')
      return

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks([1,2,3,4,5,6,7,8,9])
    ax.set_yticks([1,2,3,4,5,6,7,8,9])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    ax.tick_params(bottom=False, left=False, 
                   labelbottom=False, labelleft=False)

    for i in range(9):
      for j in range(9):
        if self._puzzle[i][j] != 0:
          ax.annotate(str(self.puzzle[i][j]), xy=(j+.40, 8-i+.35))
        else:
          ax.annotate(str(self.puzzle[i][j]), xy=(j+.40, 8-i+.35), color=color)
    
    if filename is not None:
      plt.savefig(filename, bbox_inches='tight', dpi=200)
    plt.show()