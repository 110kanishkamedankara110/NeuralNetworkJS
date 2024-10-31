class Matrix {
  constructor(rows, columns) {
    if (rows == null || columns == null) {
      console.error(
        "Matrix constructor requires valid row and column dimensions. Please provide both the number of rows and columns."
      );
      return undefined;
    }
    this.rows = rows;
    this.columns = columns;
    this.data = [];

    for (let i = 0; i < this.rows; i++) {
      this.data[i] = [];
      for (let j = 0; j < this.columns; j++) {
        this.data[i][j] = 0;
      }
    }
  }

  // Display the matrix
  viewMatrix() {
    console.table(this.data);
  }

  // Transpose the matrix
  static transpose(m) {
    let result = new Matrix(m.columns, m.rows);
    m.data.forEach((r, i) => {
      r.forEach((c, j) => {
        result.data[j][i] = c;
      });
    });
    return result;
  }
  static deserialize(data) {
    if (typeof data === 'string') {
      data = JSON.parse(data);
    }

    let matrix = new Matrix(data.rows, data.columns);
    matrix.data = data.data;  // Assign the array of arrays directly
    return matrix;
  }

  serialize() {
    return JSON.stringify({
      rows: this.rows,
      columns: this.columns,
      data: this.data
    });
  }
  // Element-wise or scalar multiplication
  multiply(n) {
    if (n instanceof Matrix) {
      if (this.rows !== n.rows || this.columns !== n.columns) {
        console.error('Dimensions of matrices must match for element-wise multiplication.');
        return;
      }
      return this.map((e, i, j) => e * n.data[i][j]);
    } else {
      return this.map(e => e * n);
    }
  }

  // Static method for matrix multiplication (dot product)
  static multiply(m, n) {
    if (m.columns !== n.rows) {
      console.error(
        `Shape mismatch: rows of B (${n.rows}), and columns of A (${m.columns}) must match.`
      );
      return undefined;
    }
    let result = new Matrix(m.rows, n.columns);

    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < n.columns; j++) {
        let sum = 0;
        for (let k = 0; k < m.columns; k++) {
          sum += m.data[i][k] * n.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }
    return result;
  }

  // Add matrices or scalar
  add(n) {
    if (n instanceof Matrix) {
      if (this.rows !== n.rows || this.columns !== n.columns) {
        console.error('Dimensions of matrices must match for addition.');
        return;
      }
    }
    this.data.forEach((r, i) => {
      r.forEach((c, j) => {
        this.data[i][j] += n instanceof Matrix ? n.data[i][j] : n;
      });
    });
  }

  // Randomize the matrix with values between -1 and 1
  randomize() {
    this.data.forEach((r, i) => {
      r.forEach((c, j) => {
        this.data[i][j] = Math.random() * 2 - 1;  // Range from -1 to 1
      });
    });
  }

  // Apply a function element-wise to the matrix
  map(func) {
    this.data.forEach((r, i) => {
      r.forEach((c, j) => {
        this.data[i][j] = func(c, i, j);
      });
    });
  }

  // Static map method for applying a function to a matrix
  static map(m, func) {
    let result = new Matrix(m.rows, m.columns);
    m.data.forEach((r, i) => {
      r.forEach((c, j) => {
        result.data[i][j] = func(c, i, j);
      });
    });
    return result;
  }

  // Create a matrix from an array
  static fromArray(arr) {
    let m = new Matrix(arr.length, 1);
    for (let i = 0; i < arr.length; i++) {
      m.data[i][0] = arr[i];
    }
    return m;
  }

  // Subtract two matrices
  static subtract(a, b) {
    if (a.rows !== b.rows || a.columns !== b.columns) {
      console.error('Dimensions of matrices must match for subtraction.');
      return undefined;
    }
    let result = new Matrix(a.rows, a.columns);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.columns; j++) {
        result.data[i][j] = a.data[i][j] - b.data[i][j];
      }
    }
    return result;
  }

  // Convert matrix to array
  toArray() {
    let arr = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.columns; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }
}
