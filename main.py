#import dependencies
import os, pygame, random
from pygame.math import Vector2
from math import sqrt
from time import sleep
import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import concatenate, Activation, Input, Dense, Dropout
from keras.optimizers import Adam
from losswise.libs import LosswiseKerasCallback

#centre pygame window
os.environ['SDL_VIDEO_CENTERED'] = '1'

#tuple with dimensions of pygame window
dimensions = (800, 600)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

time_offset = 0

#Game objects will represent generations for training and testing Neural Network
class Game:

    #parameters for Game instances
    def __init__(self, duration, dims, sample_interval, learning_time, generation):
        '''

        :param duration: game duration in seconds
        :param dims: dimensions of window
        :param sample_interval: record data and update actuations every n frames

        '''
        self.sample_interval = sample_interval
        self.learning_time = learning_time
        self.generation = generation
        self.duration = duration
        self.screen_width = dims[0]
        self.screen_height = dims[1]

        #initialize pygame, starts clock
        pygame.init()

        #set up window
        self.gameDisplay = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Machine Learning Shooters')

        #create clock variable
        self.clock = pygame.time.Clock()

        #create fonts
        self.font1 = pygame.font.Font(None, 20)
        self.font2 = pygame.font.Font(None, 35)

        #persistant actuations for each player, updated at sample_interval
        self.change_list = [
            (0, 0, 0),
            (0, 0, 0)
        ]

        #closes game window
        self.closed = False

        #initiate scores and their renders
        self.score_left = 0
        self.score_right = 0
        self.score_left_render = self.font2.render(str(self.score_left), True, RED)
        self.score_right_render = self.font2.render(str(self.score_right), True, RED)

        #generation render
        self.generation_render = self.font1.render('Generation: ' + str(self.generation), True, BLUE)

        #keeps track of time elapsed after last sample_interval activation
        self.update_time1 = 0

        #keeps track of time elapsed after last learning_time activation
        self.update_time2 = 0


    def check_events(self):
        '''

        checks for events that close windows, such as close button
        sets closed variable to True, ending game loop

        '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True
                pygame.quit()
                pygame.display.quit()

    def change_score(self, side):
        '''

        :param side: increment score for this side
        if no side is specified, score is reset
        '''
        if side == 'left':
            self.score_left += 1
        elif side == 'right':
            self.score_right += 1
        else:
            self.score_left = 0
            self.score_right = 0

        #render updated score
        self.score_left_render = self.font2.render(str(self.score_left), True, RED)
        self.score_right_render = self.font2.render(str(self.score_right), True, RED)

    def draw_arena(self):
        '''

        draws arena lines and blits previously rendered texts (scores, fps)

        '''
        pygame.draw.line(self.gameDisplay, BLACK, (self.screen_width / 2, 0),
                         (self.screen_width / 2, self.screen_height), 1)
        pygame.draw.line(self.gameDisplay, BLACK, (self.screen_width * 3 / 8, 0),
                         (self.screen_width * 3 / 8, self.screen_height), 1)
        pygame.draw.line(self.gameDisplay, BLACK, (self.screen_width * 5 / 8, 0),
                         (self.screen_width * 5 / 8, self.screen_height), 1)

        self.gameDisplay.blit(self.score_left_render, (100, 10))
        self.gameDisplay.blit(self.score_right_render, (700, 10))
        self.gameDisplay.blit(self.generation_render, (10, 570))

        try:
            fps = self.font1.render(str(int(self.clock.get_fps())), True, RED)
            self.gameDisplay.blit(fps, (10, 10))
        except:
            pass

    def loop(self):
        global time_offset
        global player_list
        '''

        one pass-through of game loop
        calls other methods in class
        if sample_interval arrives, updates player actuations and records data
        if game ends, writes data
        also updates bullets and draws everything through other methods

        '''

        self.check_events()

        #time elapsed in frame
        if self.clock.get_time() == 0:
            frame_time = 1
        else:
            frame_time = self.clock.get_time()

        #game time runs out
        if pygame.time.get_ticks() - time_offset >= self.duration * 1000:
            print('done')

            '''

            if self.score_left > self.score_right:
                player2.reverse_data()
            elif self.score_right > self.score_left:
                player1.reverse_data()
            '''

            for player in player_list:
                player.write_data()

            self.closed = True
            #pygame.quit()
            #pygame.display.quit()
            return

        #run frame
        else:
            self.gameDisplay.fill(WHITE)
            self.draw_arena()

            #sample interval activated
            if self.update_time1 >= frame_time * self.sample_interval:
                self.update_time1 = 0

                for i in range(len(player_list)):
                    if self.generation == 1:
                        inputs = None
                        player_list[i].capture_data()
                    else:
                        inputs = player_list[i].capture_data()

                    actuation = player_list[i].ai.outputs_to_actuations(
                        player_list[i].ai.run_model(inputs))



                    self.change_list[i] = actuation[:3]
                    #if actuation[-1]:
                    player_list[i].shoot()

            #learning time activated
            if (self.update_time2 >= frame_time * self.learning_time) and self.generation != 1:
                self.update_time2 = 0

                for player in player_list:
                    if player.scored_on:
                        player.reverse_data()
                        print(str(player) + ' data reversed')
                    elif not player.scored:
                        player.delete_data()
                        print(str(player) + ' data deleted')
                    #elif not player.scored and not player.scored_on:
                    #    player.reverse_data()


                    player.scored = False
                    player.scored_on = False


            #increment time elapsed by one frame
            for i in range(len(player_list)):

                player_list[i].actuation(self.change_list[i])

                player_list[i].update_bullets()
                player_list[i].draw()

            self.update_time1 += frame_time
            self.update_time2 += frame_time
            pygame.display.update()

            self.clock.tick(60)

class Player:

    def __init__(self, pos, v, va, side, colour):
        '''

        :param pos: initial position (x, y, angle)
        :param v: translational velocity
        :param va: angular velocity
        :param side: 'left' or 'right' side to determine scores
        :param colour: colour of player

        '''
        self.pos = pos
        #avoiding aliasing
        self.i_pos = np.array(pos)
        self.v = v
        self.va = va
        self.side = side
        self.colour = colour

        #directions
        self.dx = 0
        self.dy = 0
        self.da = 0

        #list of bullets created by player
        self.bullets = []

        self.game = None
        self.gameDisplay = None
        self.xlim = None
        self.opponent = None
        self.ai = None

        #sample data (states, not actuations yet)
        self.samples = []

        #keep track of performance
        self.scored = False
        self.scored_on = False
        self.reversed_data = -100

    def __str__(self):
        if self.side == 'left':
            return 'player1'
        else:
            return 'player2'

    def set_game(self, gm):


        self.game = gm
        self.reset_pos()

        self.gameDisplay = gm.gameDisplay

        # set x boundaries based on side
        if self.side == 'left':
            self.xlim = (0, self.game.screen_width * 3 / 8)
        else:
            self.xlim = (self.game.screen_width * 5 / 8, self.game.screen_width)

    def set_opponent(self, opponent):
        self.opponent = opponent

    def set_ai(self, ai):
        self.ai = ai

    def actuation(self, change):
        '''

        :param change: actuation directions (x dir, y dir, a dir)

        '''
        self.dx = change[0]
        self.dy = change[1]
        self.da = change[2]

        new_x = self.pos[0] + self.dx * self.v
        new_y = self.pos[1] + self.dy * self.v

        if self.xlim[0] < new_x < self.xlim[1]:
            self.pos[0] = new_x

        if 0 < new_y < self.game.screen_height:
            self.pos[1] = new_y

        self.pos[2] += self.da * self.va
        if self.pos[2] >= 360:
            self.pos[2] = 0

    def shoot(self):
        self.bullets.append(Bullet(self, 15))

    def update_bullets(self):
        '''

        check bullet collisions, check if bullets in frame

        '''

        for bullet in self.bullets[:]:

            if bullet.player_collision():
                self.game.change_score(self.side)
                self.scored = True
                self.opponent.scored_on = True
                self.bullets.remove(bullet)
                del bullet


            elif (0 < bullet.pos[0] < self.game.screen_width) and \
                    (0 < bullet.pos[1] < self.game.screen_height):
                bullet.update_loc()
                bullet.draw_()

            else:
                self.bullets.remove(bullet)
                del bullet

    def reset_pos(self):
        self.pos = list(self.i_pos)

    def draw(self):
        vector = Vector2()
        vector.from_polar((1000, self.pos[2]))

        pygame.draw.line(self.gameDisplay, RED, (self.pos[0], self.pos[1]),
                         (self.pos[0], self.pos[1]) + vector, 1)
        pygame.draw.circle(self.gameDisplay, self.colour, (self.pos[0], self.pos[1]), 18, 18)

    def capture_data(self):
        if self.side == 'left':
            sample = [
                np.interp(self.pos[0], (0, self.game.screen_width * 3 / 8), (0, 1)),
                np.interp(self.pos[1], (0, self.game.screen_height), (0, 1)),
                np.interp(self.pos[2], (0, 360), (0, 1)),
                np.interp(self.opponent.pos[0], (self.game.screen_width * 5 / 8,
                                              self.game.screen_width), (0, 1)),
                np.interp(self.opponent.pos[1], (0, self.game.screen_height), (0, 1)),
                np.interp(self.opponent.pos[2], (0, 360), (0, 1)),
                np.interp(self.opponent.dx, (-1, 1), (0, 1)),
                np.interp(self.opponent.dy, (-1, 1), (0, 1)),
                np.interp(self.opponent.da, (0, 360), (0, 1)),
            ] + list(self.ai.current_outputs)
        else:
            sample = [
                np.interp(self.pos[0], (self.game.screen_width * 5 / 8,
                                     self.game.screen_width), (0, 1)),
                np.interp(self.pos[1], (0, self.game.screen_height), (0, 1)),
                np.interp(self.pos[2], (0, 360), (0, 1)),
                np.interp(self.opponent.pos[0], (0, self.game.screen_width * 3 / 8), (0, 1)),
                np.interp(self.opponent.pos[1], (0, self.game.screen_height), (0, 1)),
                np.interp(self.opponent.pos[2], (0, 360), (0, 1)),
                np.interp(self.opponent.dx, (-1, 1), (0, 1)),
                np.interp(self.opponent.dy, (-1, 1), (0, 1)),
                np.interp(self.opponent.da, (0, 360), (0, 1))
            ] + list(self.ai.current_outputs)

        self.samples.append(sample)

        return sample[:9]

    def reverse_data(self):
        if self.reversed_data > len(self.samples) - 10:
            return
        for i in range(int(self.game.learning_time // self.game.sample_interval)):
            sample = self.samples[i][9:]

            i1 = []
            i2 = []
            i3 = []

            for j in range(0, 2):
                if not sample[j]:
                    i1.append(j)
            for j in range(3, 5):
                if not sample[j]:
                    i2.append(j)
            for j in range(6, 8):
                if not sample[j]:
                    i3.append(j)

            i_list = [i1, i2, i3]
            #print(i_list)

            sample = [0 if x == 1 else 1 for x in sample]

            for l in i_list:
                sample[random.choice(l)] = 0

            self.samples[i][9:] = sample
            self.reversed_data = len(self.samples)

    def delete_data(self):

        for i in sorted(range((self.game.learning_time // self.game.sample_interval)), reverse=True):
            del self.samples[i]

    def write_data(self):
        with open(self.side + '_data.txt', 'w') as file:
            file.truncate()
            for sample in self.samples:
                file.write('%s\n' % sample)

class Bullet:

    def __init__(self, player, v):
        '''

        :param player: parent player object
        :param v: bullet velocity
        '''
        self.player = player
        self.pos = player.pos[:2]
        self.vector = Vector2()
        self.vector.from_polar((v, player.pos[2]))

    def __eq__(self, other):
        return self.pos == other.pos

    def update_loc(self):
        self.pos += self.vector

    def draw_(self):
        pygame.draw.circle(self.player.gameDisplay, BLACK,
                           (int(self.pos[0]), int(self.pos[1])), 6, 6)

    def player_collision(self):
        try:
            dist = sqrt((self.pos[0] - self.player.opponent.pos[0])**2 +
                        (self.pos[1] - self.player.opponent.pos[1])**2)

            if dist < 24:
                self.player.scored = True
                self.player.opponent.scored = True
                return True
        except:
            return False

class AI:

    def __init__(self, l):
        self.current_outputs = [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0]
        self.won = True
        self.training = False

        self.options1 = [-1, 0, 1]
        self.options2 = [0, 1]

        input_layer = Input(shape=(9,))
        dense1 = Dense(300, activation='relu')(input_layer)
        dense2 = Dense(400, activation='relu')(dense1)
        drop = Dropout(rate=0.3)(dense2)
        dense3 = Dense(400, activation='relu')(drop)
        dense4 = Dense(300, activation='relu')(dense3)

        out1 = Dense(3)(dense4)
        out2 = Dense(3)(dense4)
        out3 = Dense(3)(dense4)
        out4 = Dense(2)(dense4)

        out1 = Activation('softmax')(out1)
        out2 = Activation('softmax')(out2)
        out3 = Activation('softmax')(out3)
        out4 = Activation('softmax')(out4)

        output_layer = concatenate([out1, out2, out3, out4])
        #output_layer = Activation('relu')(output_layer)

        self.model = Model(inputs=input_layer, outputs=output_layer)

        self.model.compile(Adam(lr=l),
                          loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'])

        self.model.summary()



        '''
        
        NEURAL NETWORKS
        
        '''

        #X AXIS MOVEMENT
        '''

        self.xmovement = Sequential([
            Dense(50, input_shape=(9,), activation='relu'),
            Dense(100, activation='relu'),
            Dense(50, activation='relu'),
            Dense(3, activation='softmax')
        ])

        self.xmovement.compile(Adam(lr=l),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        #Y AXIS MOVEMENT

        self.ymovement = Sequential([
            Dense(50, input_shape=(9,), activation='relu'),
            Dense(100, activation='relu'),
            Dense(50, activation='relu'),
            Dense(3, activation='softmax')
        ])

        self.ymovement.compile(Adam(lr=l),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        #ANGLE MOVEMENT

        self.amovement = Sequential([
            Dense(50, input_shape=(9,), activation='relu'),
            Dense(100, activation='relu'),
            Dense(50, activation='relu'),
            Dense(3, activation='softmax')
        ])

        self.amovement.compile(Adam(lr=l),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        #SHOOTING

        self.shot = Sequential([
            Dense(20, input_shape=(9,), activation='relu'),
            Dense(50, activation='relu'),
            Dense(50, activation='relu'),
            Dense(2, activation='softmax')
        ])

        self.shot.compile(Adam(lr=l),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        '''

    def train(self, file, d, b, e):
        global time_offset

        data = []
        for line in open(file, 'r'):
            datum = []
            for number in line[1:-2].split(', '):
                datum.append(float(number))
            data.append(datum)

        samples = []
        labels = []
        xlabels = []
        ylabels = []
        alabels = []
        slabels = []

        for i in range(len(data)):
            samples.append(data[i][:9])
            labels.append(data[i][9:])
            xlabels.append(data[i][9:12])
            ylabels.append(data[i][12:15])
            alabels.append(data[i][15:18])
            slabels.append(data[i][18:20])

        samples = np.array(samples[-d:])
        labels = np.array(labels[-d:])
        xlabels = np.array(xlabels[-d:])
        ylabels = np.array(ylabels[-d:])
        alabels = np.array(alabels[-d:])
        slabels = np.array(slabels[-d:])

        self.training = True

        self.model.fit(samples,
                  labels,
                  #[xlabels, ylabels, alabels, slabels],
                  batch_size=b,
                  epochs=e,
                  shuffle=True,
                  verbose=1)

        time_offset = pygame.time.get_ticks()
        self.training = False

        '''

        self.xmovement.fit(samples,
                  xlabels,
                  batch_size=b,
                  epochs=e,
                  shuffle=True,
                  verbose=1
                  )
        self.ymovement.fit(samples,
                           ylabels,
                           batch_size=b,
                           epochs=e,
                           shuffle=True,
                           verbose=1
                           )
        self.amovement.fit(samples,
                           alabels,
                           batch_size=b,
                           epochs=e,
                           shuffle=True,
                           verbose=1
                           )
        self.shot.fit(samples,
                           slabels,
                           batch_size=b,
                           epochs=e,
                           shuffle=True,
                           verbose=1
                           )
        
        '''

    def fit_to_first(self, file, d, b, e):
        l = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, 0.02)

        self.train(file, d, b, e)

        K.set_value(self.model.optimizer.lr, l)



    def run_model(self, ins):
        #print(ins)
        try:
            #list(ins)
            inputs = np.array([np.array(ins)])

            out = self.model.predict(inputs)[0]

            #print(out)

            for i in range(3):
                if out[i] < max(out[:3]):
                    out[i] = int(0)
                else:
                    out[i] = int(1)
            for i in range(3, 6):
                if out[i] < max(out[3:6]):
                    out[i] = int(0)
                else:
                    out[i] = int(1)
            for i in range(6, 9):
                if out[i] < max(out[6:9]):
                    out[i] = int(0)
                else:
                    out[i] = int(1)
            for i in range(9, 11):
                if out[i] < max(out[9:11]):
                    out[i] = int(0)
                else:
                    out[i] = int(1)

            #print(out)

            '''
            x = self.xmovement.predict(inputs)
            y = self.ymovement.predict(inputs)
            a = self.amovement.predict(inputs)
            s = self.shot.predict(inputs)
            '''

            #print(x)
            #print(x + y + a + s)
            #print(out)
            return out

            #return np.array(x + y + a + s)

        except Exception as e:
            print(e)
            return

    def outputs_to_actuations(self, outputs):
        #print(outputs)

        #generate random outputs
        try:
            #outputs = outputs[0]
            len(outputs)
        except:

            n1 = random.randint(0, 2)
            n2 = random.randint(3, 5)
            n3 = random.randint(6, 8)
            n4 = random.randint(9, 10)

            outputs = np.zeros(11, dtype=int)

            outputs[n1] = 1
            outputs[n2] = 1
            outputs[n3] = 1
            outputs[n4] = 1

        self.current_outputs = outputs

        mp = self.options1 * 3 + self.options2

        actuations = []
        for i in range(len(outputs)):
            if outputs[i]:
                actuations.append(mp[i])

        return tuple(actuations)

#params
game_duration = 8
sample_int = 5
data_size = 120
learning_rate = 0.00075
batches = 10
epochs = 20
learn_time = 120

#player list
player1 = Player([50, 400, 0], 1, 2, 'left', BLUE)
player2 = Player([550, 400, 180], 1, 2, 'right', GREEN)

ai1 = AI(learning_rate)
ai2 = AI(learning_rate)

player1.set_ai(ai1)
player2.set_ai(ai2)

player1.set_opponent(player2)
player2.set_opponent(player1)

player_list = [player1, player2]

first = True

for gen in range(1000):
    game = Game(game_duration, dimensions, sample_int, learn_time, gen + 1)

    player1.set_game(game)
    player2.set_game(game)

    while not game.closed:
        game.loop()

    for player in player_list:
        for bullet in player.bullets:
            player.bullets.remove(bullet)
            del bullet

    player1.ai.train('left_data.txt', data_size, batches, epochs)
    player2.ai.train('right_data.txt', data_size, batches, epochs)


    first = False
pygame.quit()