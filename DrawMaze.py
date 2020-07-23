import pygame
import numpy as np
import random
import math
from module.network import NeuralNetwork

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

def drawBoard():
    pygame.init()
    dis = pygame.display.set_mode((windowWidth, windowHeight))
    for i in range(numRow + 1):
        pygame.draw.line(dis, gray, (0, i * squareSize + i * lineWidth + startBoardHeight),
                         (boardWidth, i * squareSize + i * lineWidth + startBoardHeight), lineWidth)
    for i in range(numCol + 1):
        pygame.draw.line(dis, gray, (i * squareSize + i * lineWidth, startBoardHeight),
                         (i * squareSize + i * lineWidth, startBoardHeight + boardHeight), lineWidth)
    pygame.display.update()
    return dis


def Draw():
    currentPos = [0,0]
    board = np.zeros((numCol,numRow))
    clock = pygame.time.Clock()
    isPrint = False
    count = 0
    pygame.key.set_repeat(1,40)
    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
            if event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                if isPrint:
                    board[currentPos[0]][currentPos[1]] = -1
                currentPos[1] -= 1

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                if isPrint:
                    board[currentPos[0]][currentPos[1]] = -1
                currentPos[1] += 1
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                if isPrint:
                    board[currentPos[0]][currentPos[1]] = -1

                currentPos[0] -= 1
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                if isPrint:
                    board[currentPos[0]][currentPos[1]] = -1
                currentPos[0] += 1
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_5:
                if isPrint == False:
                    isPrint = True
                else:
                    isPrint = False
                print(isPrint)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_8:
                print('[')
                for i in range(numCol):
                    print('[',end = '')
                    for j in range(numRow):
                        print(str(int(board[i][j])) + ", ", end = '')
                    print('], ')
                print(']\n')
        for i in range(numCol):
            for j in range(numRow):
                if board[i][j] == 0:
                    pygame.draw.rect(dis, black, [i * (lineWidth + squareSize) + lineWidth,
                                              j * (lineWidth + squareSize) + lineWidth + startBoardHeight, squareSize,
                                              squareSize])
                else:
                    pygame.draw.rect(dis, white, [i * (lineWidth + squareSize) + lineWidth,
                                                  j * (lineWidth + squareSize) + lineWidth + startBoardHeight,
                                                  squareSize,
                                                  squareSize])
        pygame.draw.rect(dis, cyan, [currentPos[0] * (lineWidth + squareSize) + lineWidth,
                                      currentPos[1] * (lineWidth + squareSize) + lineWidth + startBoardHeight, squareSize,
                                      squareSize])
        pygame.display.update()
        clock.tick(30)

dis = drawBoard()
Draw()