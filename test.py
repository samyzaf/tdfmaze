from tdfmaze import *

maze = np.array([
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.],
    [ 1.,  1.,  1.,  1.,  1.,  1.,  1.],
])

flags = [(3,0), (3,2), (3,4), (3,6)]
env = TdfMaze(maze, flags)

model = build_model(env)

qt = Qtraining(
    model,
    env,
    n_epoch = 200,
    max_memory = 500,
    data_size = 100,
    name = 'model_1'
)

qt.train()

qt.save('model1')
print("Compute Time:", qt.seconds)





