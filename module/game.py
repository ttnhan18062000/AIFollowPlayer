import pygame
import numpy as np
import random
import math
import time
from module.network import NeuralNetwork
from module.aStarSearch import astar_search

red = (255,0,0)
cyan = (0,255,255)
black = (0,0,0)
white = (255,255,255)
green = (0,255,0)
gray = (200,200,200)

squareSize = 15
lineWidth = 2
numRow = 30
numCol = 50
numRowVision = 7
numColVision = 7
vision = np.zeros((numColVision,numRowVision))
startBoardHeight = 50
marginBoardHeight = 100
marginBoardWidth = 30
marginVisionHeight = 100
marginVisionWidth = 84
boardHeight = lineWidth*numRow + squareSize*numRow
windowHeight = 650
boardWidth = lineWidth*numCol + squareSize*numCol
windowWidth = 1300
maze = [
[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, ],
[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, -1, ],
[-1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, ],
[-1, 0, 0, 0, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, ],
[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, 0, 0, -1, -1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, -1, -1, -1, -1, ],
[-1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, ],
[-1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, ],
[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, ],
]

def getAngle(a, b, c):
    try:
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)
    except:
        return 0

class Player():
    def __init__(self, pos):
        self.pos = pos
    def move(self, dir, board):
        if self.isMovable(dir, board):
            board[self.pos[0]][self.pos[1]] = 0
            if dir == 'up':
                self.pos[1] -= 1
            elif dir == 'down':
                self.pos[1] += 1
            elif dir == 'left':
                self.pos[0] -= 1
            elif dir == 'right':
                self.pos[0] += 1
            board[self.pos[0]][self.pos[1]] = 1
            return True
        return False
    def moveRandomly(self, board):
        tmp = random.randint(0, 3)
        dirs = ['up','down','left','right']
        while self.isMovable(dirs[tmp], board) == False:
            tmp = random.randint(0, 3)
        self.move(dirs[tmp], board)
    def isMovable(self, dir, board):
        if dir == 'up' and (self.pos[1] - 1 < 0 or board[self.pos[0]][self.pos[1] - 1] == -1):
            return False
        elif dir == 'down' and (self.pos[1] + 1 > numRow - 1 or board[self.pos[0]][self.pos[1] + 1] == -1):
            return False
        elif dir == 'left' and (self.pos[0] - 1 < 0 or board[self.pos[0] - 1][self.pos[1]] == -1):
            return False
        elif dir == 'right' and (self.pos[0] + 1 > numCol - 1 or board[self.pos[0] + 1][self.pos[1]] == -1):
            return False
        return True


class Game():
    def __init__(self):
        self.score = 0
        self.penalty = 0
        self.remainStep = (numCol+numRow)*2
        self.maxStep = (numCol+numRow)*2
        self.isCatched = 0
        self.board = np.zeros((numCol,numRow))
        for i in range(numCol):
            for j in range(numRow):
                self.board[i][j] = maze[i][j]
        self.follower = self.initPlayer()
        self.runner = self.initPlayer(self.follower.pos)
        self.followerStartPos = self.follower.pos
        self.startDistance = astar_search((self.follower.pos[0],self.follower.pos[1]),(self.runner.pos[0],self.runner.pos[1]))
        self.isToggled = False
        self.isShowed = False
        self.isContinousPenalty = False
        self.penaltyStreak = 0
        self.isDead = 0
        self.followerVision = ''
    def initPlayer(self, otherPlayerPos= None):
        x = random.randint(1, numCol - 1)
        y = random.randint(1, numRow - 1)
        while self.board[x][y] == -1 or (otherPlayerPos is not None and (abs(x - otherPlayerPos[0]) + abs(y - otherPlayerPos[1])) < 20):
            x = random.randint(1, numCol - 1)
            y = random.randint(1, numRow - 1)
        self.board[x][y] = 1
        return Player([x,y])
    def start(self, network = None):
        if network is None:
            network = NeuralNetwork()
        clock = pygame.time.Clock()
        isRun = True
        while isRun:
            self.update(network)
            if self.isCatched == 1:
                isRun = False
                break
            if self.remainStep == 0:
                break
            if self.isShowed == True:
                if self.isToggled == False:
                    pygame.init()
                    dis = pygame.display.set_mode((windowWidth, windowHeight))
                    self.drawBoard(dis)
                    self.isToggled = True
                self.render(dis, network)
                clock.tick(40)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        isRun = False
                        break
        self.score = self.fitness()
        return self.score
    def update(self, network):
        #self.runner.moveRandomly(self.board)
        currentDistance = astar_search((self.follower.pos[0],self.follower.pos[1]),(self.runner.pos[0],self.runner.pos[1]))
        if self.maxStep - self.remainStep > 2*(self.startDistance - currentDistance):
            print(self.remainStep)
            self.isDead = 1
            self.remainStep = 0
            return
        runnerVision = self.getRunnerDirection()
        vision = self.getVision()
        tmp = np.zeros((1, vision.size))
        tmp[0] = vision
        self.followerVision = np.concatenate((tmp, runnerVision), axis=1)
        output = network.filterOutput(network.forward(self.followerVision))
        nextStep = np.where(output == 1)[0]
        move = ''
        if nextStep == 0:
            move = 'up'
        elif nextStep == 1:
            move = 'down'
        elif nextStep == 2:
            move = 'left'
        else:
            move = 'right'
        if self.follower.move(move, self.board) == False:
            self.penalty += 1
            self.isContinousPenalty = True
            if self.isContinousPenalty:
                self.penaltyStreak += 1
                if self.penaltyStreak == 20:
                    self.isDead = 1
                    self.remainStep = 0
                    return
        else:
            self.isContinousPenalty = False
            self.remainStep -= 1
        if self.follower.pos[0] == self.runner.pos[0] and self.follower.pos[1] == self.runner.pos[1]:
            self.isCatched = 1
    def fitness(self):
        return (self.maxStep - self.remainStep) - self.penalty*5 + self.isCatched*1000 + (100 - astar_search((self.follower.pos[0],self.follower.pos[1]),(self.runner.pos[0],self.runner.pos[1])))*3 - self.isDead*1000
    def getVision(self):
        vision = []
        directions = [[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1]]
        for direction in directions:
            res = self.raycast(-1, direction)
            vision.append(res['distance'])
        return np.array(vision)
    def raycast(self, target, direction):
        startPos = [self.follower.pos[0], self.follower.pos[1]]
        distance = 0
        while (startPos[0] >= 0 and startPos[0] < numCol and startPos[1] >= 0 and startPos[1] < numRow):
            if self.board[startPos[0]][startPos[1]] == target and startPos[0] != self.follower.pos[0] and startPos[1] != self.follower.pos[1]:
                return {'distance': distance, 'isFound': True}
            startPos[0] += direction[0]
            startPos[1] += direction[1]
            distance += abs(direction[0]) + abs(direction[1])
        return {'distance': distance, 'isFound': False}
    def getRunnerDirection(self):
        posHead = self.follower.pos
        upVector = np.array([posHead[0], 0])
        headVector = np.array([posHead[0], posHead[1]])
        appleVector = np.array([self.runner.pos[0], self.runner.pos[1]])
        angle = getAngle(upVector, headVector, appleVector)
        runnerPosition = 0
        if angle >= 157.5:
            runnerPosition = 4
        elif angle >= 112.5:
            if posHead[0] < self.runner.pos[0]:
                runnerPosition = 3
            else:
                runnerPosition = 5

        elif angle >= 67.5:
            if posHead[0] < self.runner.pos[0]:
                runnerPosition = 2
            else:
                runnerPosition = 6

        elif angle >= 22.5:
            if posHead[0] < self.runner.pos[0]:
                runnerPosition = 1
            else:
                runnerPosition = 7

        else:
            runnerPosition = 0
        vectorizePosition = np.zeros((1, 8))
        vectorizePosition[0][runnerPosition] = abs(self.runner.pos[0] - posHead[0]) + abs(self.runner.pos[1] - posHead[1])

        return vectorizePosition
    def drawBoard(self, dis):
        for i in range(numRow+1):
            pygame.draw.line(dis, gray, (0, i*squareSize + i*lineWidth + startBoardHeight), (boardWidth, i*squareSize + i*lineWidth + startBoardHeight), lineWidth)
        for i in range(numCol+1):
            pygame.draw.line(dis, gray, (i * squareSize + i*lineWidth, startBoardHeight), (i * squareSize + i*lineWidth, startBoardHeight + boardHeight), lineWidth)
        for i in range(numCol):
            for j in range(numRow):
                if self.board[i][j] == -1:
                    pygame.draw.rect(dis, white, [i * (lineWidth + squareSize) + lineWidth,
                                                 j * (lineWidth + squareSize) + lineWidth + startBoardHeight,squareSize,squareSize])
        pygame.draw.rect(dis, cyan, [self.follower.pos[0] * (lineWidth + squareSize) + lineWidth,
                                      self.follower.pos[1] * (lineWidth + squareSize) + lineWidth + startBoardHeight, squareSize,
                                      squareSize])
        pygame.draw.rect(dis, red, [self.runner.pos[0] * (lineWidth + squareSize) + lineWidth,
                                      self.runner.pos[1] * (lineWidth + squareSize) + lineWidth + startBoardHeight, squareSize,
                                      squareSize])
        pygame.display.update()

    def render(self, dis, network):
        network.render(dis, self.followerVision, [900, 325])
        #self.displayAppleDirection(self.getAppleDirection(),dis)
        #self.displayVision(dis)
        for i in range(numCol):
            for j in range(numRow):
                if self.board[i][j] == 0:
                    color = black
                else:
                    color = white
                pygame.draw.rect(dis, color, [i * (lineWidth + squareSize) + lineWidth,
                                      j * (lineWidth + squareSize) + lineWidth + startBoardHeight, squareSize, squareSize])
        pygame.draw.rect(dis, cyan, [self.follower.pos[0] * (lineWidth + squareSize) + lineWidth,
                                      self.follower.pos[1] * (lineWidth + squareSize) + lineWidth + startBoardHeight, squareSize,
                                      squareSize])
        pygame.draw.rect(dis, red, [self.runner.pos[0] * (lineWidth + squareSize) + lineWidth,
                                     self.runner.pos[1] * (lineWidth + squareSize) + lineWidth + startBoardHeight,
                                     squareSize,
                                     squareSize])
        pygame.display.update()
"""
while 1:
    game = Game()
    game.isShowed = True
    print(game.start())
"""

