import numpy as np
import matplotlib.pyplot as pl

def h(x, w):
    return np.dot(w.T, x)

def error2txt(y, yp):
    N = y.shape[0]
    error = 0
    for n in np.arange(N):
        if y[n,0] != yp[n,0]:
            error = error + 1
    txt = 'error = ' + str(error) + r'/' + str(N)
    return txt

def w2txt(w):
    title = r'$\mathbf{w} = ['
    D = w.shape[0]
    for d in np.arange(D):
        title = title + '{:.4f}'.format(w[d,0])
        if d != D-1:
            title = title + ','
    title = title + ']$'
    return title
    

def plot_scatter(x, y, fname):
    pl.figure()
    ni = (y==-1)[:,0]
    pi = (y==+1)[:,0]
    pl.scatter(x[ni, 1], x[ni, 2], s=64, color='b', marker='o')
    pl.scatter(x[pi, 1], x[pi, 2], s=64, color='r', marker='s')
    pl.ylim((-1, 5))
    pl.xlim((-2, 14))
    pl.axis('off')
    if fname != '':
        pl.savefig(fname)
    
    
def plot_scatter_with_w(x, y, w, title, fname):
    pl.figure()
    ni = (y==-1)[:,0]
    pi = (y==+1)[:,0]
    pl.scatter(x[ni, 1], x[ni, 2], s=64, color='b', marker='o')
    pl.scatter(x[pi, 1], x[pi, 2], s=64, color='r', marker='s')
    N = 1000
    x1 = np.linspace(np.min(x[:,1]), np.max(x[:,1]), N)
    x2 = -(w[0,0] * 1.0 + w[1,0] * x1) / w[2,0]
    pl.plot(x1, x2, color='k', linewidth=2)
    pl.title(title)
    pl.ylim((-1, 5))
    pl.xlim((-2, 14))
    if fname != '':
        pl.savefig(fname)

def print_result(t, err_count, dataSize):
    print("t = %d,error = %d/%d" % (t, err_count, dataSize));  

def check_error(x, y, w):
    err_count = 0;
    
    for i in np.arange(x.shape[0]):
        check = h(x[i], w) * y[i];

        if check[0] <= 0:
            err_count = err_count+1;

    return err_count;

def pegasos_algorithm(x, y, lam, T):
    w = np.zeros((x.shape[1],1))
    #complete here
    t=0;
    error = check_error(x, y, w);
    print_result(t, error, x.shape[0]);

    for t in np.arange(T):
        idex = np.random.randint(0, x.shape[0]);
        lr = 1.0 / (lam*(t+1.0));

        check_boundary = y[idex]*h(x[idex], w);

        if(check_boundary < 1.0):
            w = (1.0 - lr*lam)*w + lr*y[idex]*x[idex][:,np.newaxis];
                
        elif(check_boundary >= 1.0):
            w = (1.0 - lr*lam)*w;
        
        error = check_error(x, y, w);
        print_result(t+1.0, error, x.shape[0]);

    # print w
    # plot_scatter_with_w(x, y, w, "Pegasos", "pegasos_algorithm.png");
    return w

def svm(world, player):
    size = world.shape[0]
    movesLeft = game.numberMovesLeft(world)
    if movesLeft == 0:
        return world

    madeMove = False
    data_dat = np.load('data/NN_natural_3_3.dat.npz')
    tempWorld = world.copy()
    iterations = 0

    while madeMove == False and iterations < 1000:
        np.random.seed(7)

    # pl.close('all')
    # np.random.seed(7)
    # N = 40
    # m_blue = [2.0, 2.0]
    # m_red = [10.0, 2.0]
    # blue = np.random.randn(N, 2) + m_blue
    # red = np.random.randn(N,2) + m_red
    # x = np.vstack((blue, red))
    # y = np.vstack((-1*np.ones((blue.shape[0],1)), +1*np.ones((red.shape[0],1))))
    # x = np.hstack((np.ones((x.shape[0],1)), x))

    N = 1.0
    C = 100.0
    lam = 2.0 / (N * C) 
    T = 1000
    w = pegasos_algorithm(x_train, y_train, lam, T)

    print(w)

    if(iterations >= 1000):
        world, x, y = game.rndMoveXY(world, 1)
    else:
        world[x][y] = player
    
    return world,x,y

def SVMVsNN(board_size, tprint=False):
    world = game.initGameWorld(board_size)
    movesLeft = game.numberMovesLeft(world)
    hasWon = False
    player1won = False
    player2won = False
    moveCount = 0

    while(movesLeft > 0) and (hasWon == False):
        # player 1
        if (movesLeft > 0) and (hasWon == False):
            newWorld, x, y = game.rndMoveXY(world, 1)
            hasWon = game.checkWin(newWorld, 1) 
            
            if hasWon:
                player1won = True

            moveCount = moveCount+1 
            
        if (movesLeft > 0):
            movesLeft = game.numberMovesLeft(world)
            
        # player 2
        if (movesLeft > 0) and (hasWon == False):
            if board_size == 3:
                newWorld, x, y = svm(world, -1)
            else:
                newWorld, x, y = game.rndMoveXY(world, -1)

            hasWon = game.checkWin(newWorld, -1)

            if hasWon and not player1won:
                player2won = True
            # print(newWorld)
            world = newWorld
            moveCount = moveCount+1         

        if (movesLeft > 0):
            # print(world)
            # game.printWorld(world)
            movesLeft = game.numberMovesLeft(world)
            
    if(tprint):
        if(game.checkDraw(world, moveCount)):
            print("It's a draw!")

        else:
            if player1won:
                print("player 1 wins!")
            else:
                print("player 2 wins!")
        game.printWorld(world)

    if(game.checkDraw(world, moveCount)):
        return 0

    else:
        if player1won:
            return 1
        else:
            return -1

def SVMVsRnd(board_size, tprint=False):
    world = game.initGameWorld(board_size)
    movesLeft = game.numberMovesLeft(world)
    hasWon = False
    player1won = False
    player2won = False
    moveCount = 0

    while(movesLeft > 0) and (hasWon == False):
        # player 1
        if (movesLeft > 0) and (hasWon == False):
            if board_size == 3:
                newWorld, x, y = svm(world, 1)
            else:
                newWorld, x, y = game.rndMoveXY(world, 1)

            hasWon = game.checkWin(world, 1) 
            
            if hasWon:
                player1won = True

            moveCount = moveCount+1 
            
        if (movesLeft > 0):
            movesLeft = game.numberMovesLeft(world)
            
        # player 2
        if (movesLeft > 0) and (hasWon == False):
            newWorld, x, y = game.rndMoveXY(world, 1)
            hasWon = game.checkWin(world, -1)

            if hasWon and not player1won:
                player2won = True

            world = newWorld
            moveCount = moveCount+1         

        if (movesLeft > 0):
            movesLeft = game.numberMovesLeft(world)
            
    if(tprint):
        if(game.checkDraw(world, moveCount)):
            print("It's a draw!")

        else:
            if player1won:
                print("player 1 wins!")
            else:
                print("player 2 wins!")
        game.printWorld(world)

    if(game.checkDraw(world, moveCount)):
        return 0

    else:
        if player1won:
            return 1
        else:
            return -1

SVMVsNN(3);














