const gridContainer = document.getElementById('grid');
let grid_data = [];
let cell_data = [];
let drag = false;
let grid = [];
// Create a 28x28 grid
for (let i = 0; i < 28; i++) {
    grid_data[i] = [];
    cell_data[i] = [];
    for (let j = 0; j < 28; j++) {
        const cell = document.createElement('div');
        grid_data[i][j] = 0;
        cell_data[i][j] = cell;
        cell.classList.add('cell');
        cell.addEventListener('mousedown', () => {
            cell.style.backgroundColor = `rgba(0, 0, 0, ${1 * grid_data[i][j]})`;
            checkNeighbors(i - 1, j);
            checkNeighbors(i, j - 1);
            checkNeighbors(i, j + 1);
            checkNeighbors(i + 1, j);
            drag = true;
        });
        cell.addEventListener('mousemove', () => {
            if (drag) {
                grid_data[i][j] = 1;
                cell.style.backgroundColor = `rgba(0, 0, 0, ${1 * grid_data[i][j]})`;
                checkNeighbors(i - 1, j);
                checkNeighbors(i, j - 1);
                checkNeighbors(i, j + 1);
                checkNeighbors(i + 1, j);
            }

        })
        cell.addEventListener('mouseup', () => {
            drag = false;
        })
        gridContainer.appendChild(cell);
    }
}

function clearGrid() {
    for (let i = 0; i < cell_data.length; i++) {
        for (let j = 0; j < cell_data[0].length; j++) {
            cell = cell_data[i][j];
            grid_data[i][j] = 0;
            cell.style.backgroundColor = 'darkcyan';
        }
    }
    grid = [];
    document.getElementById('result').innerHTML = '';
}

function checkNeighbors(neighborX, neighborY, cell) {
    if (neighborX >= 0 && neighborX < grid_data.length && neighborY >= 0 && neighborY < grid_data[0].length) {
        if (grid_data[neighborX][neighborY] + 0.2 < 1) {
            grid_data[neighborX][neighborY] += 0.2;
            cell_data[neighborX][neighborY].style.backgroundColor = `rgba(0, 0, 0, ${1 * grid_data[neighborX][neighborY]})`;
        }
    }
}

function predict() {
    if (grid.length == 0) {
        for (let i = 0; i < grid_data.length; i++) {
            for (let j = 0; j < grid_data[0].length; j++) {
                grid.push(grid_data[i][j]);
            }
        }

        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data: grid })
        })
            .then(response => response.json())
            .then(data => document.getElementById('result').innerText = "I think it's " + data.prediction)
            .catch(error => console.error('There was a problem with the fetch operation:', error));
    }
}