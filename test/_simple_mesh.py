import meshio

points = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
]
cells = [
    ("triangle", [[0, 1, 2], [1, 3, 2]])
]

mesh = meshio.Mesh(points, cells)