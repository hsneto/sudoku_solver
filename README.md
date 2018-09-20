# Sudoku Solver

![sdk](https://github.com/hsneto/sudoku_solver/blob/master/docs/test.png)

## Steps

0. Load input image:
![im](https://github.com/hsneto/sudoku_solver/blob/master/docs/sudoku2.jpg)

1. Pre-processing:
![im-pp](https://github.com/hsneto/sudoku_solver/blob/master/docs/steps/edges.png)

2. Find sudoku grid:
![im-gr](https://github.com/hsneto/sudoku_solver/blob/master/docs/steps/edges-2.png)

3. Get perspective view:
![im-wr](https://github.com/hsneto/sudoku_solver/blob/master/docs/steps/warped.png)

4. Extract sudoku grid and digits:
![im-gr-di](https://github.com/hsneto/sudoku_solver/blob/master/docs/steps/grid.png)

5. Recognize digits:
  * [digits.py](https://github.com/hsneto/sudoku_solver/blob/master/sudoku/digits.py)
![tensorboard](https://github.com/hsneto/sudoku_solver/blob/master/docs/steps/summary.png)

6. Solve sudoku:
  * [solver.py](https://github.com/hsneto/sudoku_solver/blob/master/sudoku/solver.py)
![im-sv](https://github.com/hsneto/sudoku_solver/blob/master/docs/steps/solved.png)

7. EXTRA: -->[EXAMPLE](https://github.com/hsneto/sudoku_solver/blob/master/example.ipynb)<--

## Dataset

### [Pickle dataset](https://github.com/hsneto/sudoku_solver/blob/master/dataset/digits-dataset)

### Usange:

```python
def load_digits_data(filename='./dataset/digits-dataset'):
  with open(filename, 'rb') as f:
    x_train, y_train, x_test, y_test = pickle.load(f)
  return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_digits_data()
```

### The Chars74K dataset:
**Character Recognition in Natural Images**

http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

**EnglishFnt.tgz (51.1 MB): characters from computer fonts with 4 variations (combinations of italic, bold and normal). Kannada (657+ classes)**

http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz

## Trained model
[model.ckpt files](https://github.com/hsneto/sudoku_solver/tree/master/model)

```sh
Digit recognizer will be train ...
  Model dir: ./model/
  Dataset dir: ./dataset/digits-dataset
  Summary dir: ./summary/
train accuracy on step 0: 0.2199999988079071
train accuracy on step 100: 1.0
train accuracy on step 200: 0.9800000190734863
test accuracy:  0.975
Model saved in path: ./model/digits_model.ckpt
INFO:tensorflow:Restoring parameters from ./model/digits_model.ckpt
Model restored.
```

---
## Authors

* **Humberto da Silva Neto**

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/hsneto/sudoku_solver/blob/master/LICENSE) file for details

