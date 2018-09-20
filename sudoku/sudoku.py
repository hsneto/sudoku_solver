from sudoku.solver import Solver
from sudoku.digits import predict

import cv2
import numpy as np
from operator import itemgetter
from numpy.linalg import norm

class Sudoku(Solver):
  """
  Get sudoku information from image
  """
  
  def __init__(self):
    # verify if sudoku was loaded before use other functions
    self._is_loaded = False
    
  def load_sudoku(self, path, width=None, height=None):
    """
    Load grayscale image from path
    
    Args:
      path   (str) --> path to the image
      width  (int) --> resize image with width
      height (int) --> resize image with height
    """
    img = cv2.imread(path, 0)

    if width is not None:
      ratio = img.shape[1]/img.shape[0]
      img = cv2.resize(img, (width, int(width/ratio)))
    elif height is not None:
      ratio = img.shape[1]/img.shape[0]
      img = cv2.resize(img, (int(height*ratio), height))
                             
    self.img = img
    self._is_loaded = True
  
  def _preprocess_img(self, img, skip_dilate=False):
    """
    Return high frequecy features from image
    """
    # aply gaussian blurring
    img = cv2.GaussianBlur(img.copy(), (9,9), 0)

    # apply otsu method
    img_out = 255 * np.ones(img.shape, np.uint8)  
    img_out = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    cv2.bitwise_not(img_out, img_out)
    
    if skip_dilate:
      return img_out

    # return dilated output image
    kernel = np.array([[0,1,0], [1,1,1], [0,1,0]], np.uint8)
    return cv2.dilate(img_out, kernel)
  
  def _get_blob(self, img):
    """
    Remove noise outside sudoku grid
    """
    # get images' contours
    _, cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True) 

    mask = np.zeros_like(img)
    img_out = np.zeros_like(img)
    cv2.drawContours(mask, cnts, 0, 255, -1)
    img_out[mask == 255] = img[mask == 255]

    # return cleaned image and the contourn of sudoku grid 
    return img_out, cnts[0]  
  
  def _find_corners(self, cnts):
    """
    Find the corners from a contour
    """
    br, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in cnts]), key=itemgetter(1))
    tl, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in cnts]), key=itemgetter(1))
    bl, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in cnts]), key=itemgetter(1))
    tr, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in cnts]), key=itemgetter(1))

    return [cnts[tl][0], cnts[tr][0], cnts[br][0], cnts[bl][0]]   

  def _crop_and_warp(self, img, corners):
    """
    Crops and warps a rectangular section from an image into a square of similar size.
    """
    tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]
    src = np.array([tl, tr, br, bl], dtype='float32')
    side = max([
      norm(br-tl), 
      norm(tl-bl), 
      norm(br-bl), 
      norm(tl-tl)])

    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(src, dst)
    
    return cv2.warpPerspective(img.copy(), M, (int(side), int(side)))
  
  def _infer_grid(self, img):
    """
    Infers 81 cell grid from a square image.
    """
    squares = []
    square_size = img.shape[0]/9

    for i in range(9):
      for j in range(9):
        tl = (j*square_size, i*square_size)
        br = ((j+1)*square_size, (i+1)*square_size)
        squares.append((tl, br))
    return squares
  
  def _extract_from_rect(self, img, rect):
    '''Extract cell from image'''
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]
  
  def _scale_and_centre(self, img, size, margin=0, background=0):
    """
    Scales and centres an image onto a new background square.
    """

    def centre_pad(lenght):
      """
      Handles centering for a given length that may be odd or even.
      """
      side1 = int((size-lenght) / 2)
      if lenght % 2 == 0:
        side2 = side1
      else:
        side2 = side1 + 1

      return side1, side2

    def scale(r, x):
      return int(r * x)

    height, width = img.shape[:2]

    if height > width:
      t_pad = int(margin / 2)
      b_pad = t_pad
      ratio = (size - margin) / height
      height, width = scale(ratio, height), scale(ratio, width)
      l_pad, r_pad = centre_pad(width)
    else:
      l_pad = int(margin / 2)
      r_pad = l_pad
      ratio = (size - margin) / width
      height, width = scale(ratio, height), scale(ratio, width)
      t_pad, b_pad = centre_pad(height)

    img = cv2.resize(img, (width, height))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))
  
  def _find_largest_feature(self, img, scan_tl=None, scan_br=None):
    """
    Uses the fact the `floodFill` function returns a bounding box of 
    the area it filled to find the biggestconnected pixel structure in 
    the image. Fills this structure in white, reducing the rest to black.
    """
    img = img.copy()
    height, width = img.shape[:2]

    max_area = 0
    seed_point = (None, None)

    if scan_tl is None:
      scan_tl = [0, 0]
    if scan_br is None:
      scan_br = [width, height]

    for x in range(scan_tl[0], scan_br[0]):
      for y in range(scan_tl[1], scan_br[1]):
        if img.item(y, x) == 255 and x < width and y < height:
          area = cv2.floodFill(img, None, (x, y), 64)
          if area[0] > max_area:
            max_area = area[0]
            seed_point = (x, y)

    for x in range(width):
      for y in range(height):
        if img.item(y, x) == 255 and x < width and y < height:
          cv2.floodFill(img, None, (x, y), 64)

    mask = np.zeros((height+2, width+2), np.uint8)

    if all([p is not None for p in seed_point]):
      cv2.floodFill(img, mask, seed_point, 255)

    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
      for y in range(height):
        if img.item(y, x) == 64:
          cv2.floodFill(img, mask, (x, y), 0)

        if img.item(y, x) == 255:
          top = y if y < top else top
          bottom = y if y > bottom else bottom
          left = x if x < left else left
          right = x if x > right else right
        
    bbox = [[left, top], [right, bottom]]
    return np.array(bbox, dtype='float32')

  def _extract_digit(self, img, rect, size):
    """
    Extract digit from cell
    """
    digit = self._extract_from_rect(img, rect)
    
    height, width = digit.shape[:2]
    margin = int(np.mean([height, width]) / 2.5)
    bbox = self._find_largest_feature(digit, [margin, margin], [width - margin, height - margin])
    digit = self._extract_from_rect(digit, bbox)
    
    width = bbox[1][0] - bbox[0][0]
    height = bbox[1][1] - bbox[0][1]
    
    # Ignore any small bounding boxes
    if width > 0 and height > 0 and (width * height) > 100 and len(digit) > 0:
      return self._scale_and_centre(digit, size, 4)
    else:
      return np.zeros((size, size), np.uint8)

  def _get_digits(self, img, squares, size):
    """
    Get digit from cell
    """
    digits = []
    img = self._preprocess_img(img, skip_dilate=True)
    for square in squares:
      digits.append(self._extract_digit(img, square, size))
    
    return digits

  def _combine_digits(self, digits, color=255):
    """
    Combine all cells.
    This functions is NOT used to solve the sudoku, \
    only to view the extract digits,
    """
    rows = []
    with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, color) for img in digits]
    for i in range(9):
      row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
      rows.append(row)

    return np.concatenate(rows)

  def solve(self, model_dir='./model/'):
    """
    Get processed sudoku grid and solved it.

    Args:
      model_dir --> Directory that contains the trained model (ckpt files) \
      to recognise the digits.
    """
    assert self._is_loaded,"Coudn\'t find sudoku image!"
      
    edge = self._preprocess_img(self.img)
    edgec, cnts = self._get_blob(edge)
    corners = self._find_corners(cnts)
    warped = self._crop_and_warp(self.img, corners)
    squares = self._infer_grid(warped)
    cells = self._get_digits(warped, squares, 28)

    digits = {}
    for i, digit in enumerate(cells):
      if np.max(digit) == 0:
        digits[i] = (digit, None)
      else:
        digits[i] = (digit, not None)
    
    digits_v2 = []

    for i in range(81):
      if digits[i][1]:
        digits_v2.append(digits[i][0])

    predictions = predict(digits_v2, model_dir=model_dir)
    predictions = predictions.tolist()

    grid = []
    for i in range(81):
      if digits[i][1] is None:
        grid.append(0)
      else:
        grid.append(predictions.pop(0))

    # reshape grid to 9x9
    grid = np.asarray(grid).reshape(9,9).tolist()

    try:
      Solver.load_sudoku(self, grid)
      solved = Solver.solve(self)
      return solved

    except:
      print('Sudoku could not be solved!')
      return

  # Check show method from parent Class Solver



