import numpy as np
import pandas as pd
pd.set_option('precision',2)

from psychopy import core, visual, event, monitors
from imageio import imread
from matplotlib.pyplot import imshow, cm, subplot, title
from os.path import isfile

import warnings
warnings.filterwarnings('ignore')

# set some global variables
DEBUG = False
params = dict()
params['num_of_trials'] = 50 # a block always consists of 50 trials
params['fps'] = 60.0         # assume same frames per second for everyone
params['ifi'] = 1.0 / params['fps']              # inter frame interval
params['slack'] = params['ifi'] * 0.25           # slack time before flip
params['presentation_time'] = 10 * params['ifi'] # unit in [sec]
params['fixation_time'] = 10 * params['ifi']     # wait time for fix cross
params['blank_time'] = 5 * params['ifi']         # blank before stim
params['feedback_time'] = 10 * params['ifi']     # for feedback

# and now for the stimulus
params['gray'] = 127.0
params['noise_level'] = 25.0
A = imread('A.png')      # read image A from file
A = A[::-4,::4]          # downsample to 25x25 pixels
A = 1-np.mean(A,2)/255.0 # and make grayscale white on black
params['signal'] = A
params['background'] = np.ones((25,25)) * params['gray']

# generate a stimulus from the signal and random noise
def stimulus(intensity):
    noise = np.random.randn(25,25) * params['noise_level']
    signal = params['signal'] * intensity
    stim = params['background'] + signal + noise
    stim[stim>255]=255
    stim[stim<0]=0
    stim = np.round(stim)
    return stim/255.0

def imshow_stimuli(intensity):
    subplot(1,2,1)
    S0 = stimulus(0)
    imshow(S0,cmap=cm.gray,aspect='equal',vmin=0,vmax=1,origin='lower')
    title('noise only (0)')
    subplot(1,2,2)
    S1 = stimulus(intensity)
    imshow(S1,cmap=cm.gray,aspect='equal',vmin=0,vmax=1,origin='lower')
    title('noise + signal ('+str(intensity)+ ')')

def load_data(subject):
    params['filename'] = 'data_'+subject+'.csv'
    if isfile(params['filename']):
        data = pd.read_csv(params['filename'])
    else:
        data = {'subject':[],
                'task':[],
                'block':[],
                'intensity':[],
                'p':[],
                'trial':[],
                'stimulus':[],
                'response':[],
                'hit':[],
                'false_alarm':[],
                'correct':[],
                'RT':[]}
        data = pd.DataFrame(data)
    return data

def setup(intensity,task):
    if task == 'yes-no':
        text = 'Press y if you see an A and n otherwise. Any key to start (ESC to quit).'
    elif task == '2AFC':
        text = 'Press left if you see an A on the left and right if you see it on the right. Any key to start (ESC to quit).'
    else:
        error('Unknown task: should be yes-no or 2AFC')
    # this will create a warning that no monitor is specified
    # but we don't care because we only have pixel images
    if DEBUG:
        win = visual.Window([800,600], allowGUI=True,
                            units='pix', color=[0.5,0.5,0.5])
    else:
        win = visual.Window(fullscr=True,
                            units='pix', color=[0.5,0.5,0.5])
        win.setMouseVisible(False)
    message = visual.TextStim(win, pos=[0,100],text=text)
    message.draw()
    stim = visual.ImageStim(win, image=stimulus(0),
                            pos=(-15,0), size=(25,25))
    stim.draw()
    stim = visual.ImageStim(win, image=stimulus(intensity),
                            pos=(+15,0), size=(25,25))
    stim.draw()
    win.flip()
    event.waitKeys()
    return win

def run_block(subject,intensity,p=0.5,task='yes-no'):
    data = load_data(subject)
    if len(data)>0:
        block = data['block'].max()+1
    else:
        block = 1
    win = setup(intensity,task)
    v = ((0,-5), (0,5), (0,0), (5,0), (-5,0))
    fix = visual.ShapeStim(win, vertices=v, lineWidth=2, closeShape=False,
                           lineColor="white")
    pos = visual.ShapeStim(win, vertices=v, lineWidth=2, closeShape=False,
                           lineColor="green")
    neg = visual.ShapeStim(win, vertices=v, lineWidth=2, closeShape=False,
                           lineColor="red")
    # we will do the timing based on the internal clock
    # and based on frames. This is not ideal given the very
    # short presentation times but since this has to run
    # on many different machines (potentially with fps>60Hz)
    # this is still the most failsafe way, I guess. Note
    # that we subtract a slack to return control just before
    # the flip. Hopefully in this way we'll stay in sync.
    # The blanks are purposely flexible in timing. Important
    # is only the stimulus presentation.
    for i in range(params['num_of_trials']):
        core.wait(params['blank_time']-params['slack'])
        # prepare trial
        s = int(np.random.rand() < p)
        if task == 'yes-no':
            # 0 is a noise trial
            # 1 is a signal trial
            # p is the probability for a signal trial
            if s:
                stim = visual.ImageStim(win, image=stimulus(intensity),
                                        pos=(0,0), size=(25,25))
            else:
                stim = visual.ImageStim(win, image=stimulus(0),
                                        pos=(0,0), size=(25,25))
        else: # task is 2AFC
            # 0 is signal left
            # 1 is signal right
            # p is the probability of a right signal trial
            if s:
                stim_left = visual.ImageStim(win, image=stimulus(0),
                                             pos=(-15,0), size=(25,25))
                stim_right = visual.ImageStim(win, image=stimulus(intensity),
                                             pos=(+15,0), size=(25,25))
            else:
                stim_left = visual.ImageStim(win, image=stimulus(intensity),
                                             pos=(-15,0), size=(25,25))
                stim_right = visual.ImageStim(win, image=stimulus(0),
                                             pos=(+15,0), size=(25,25))
        # fixation cross
        fix.draw()
        win.flip()
        core.wait(params['fixation_time']-params['slack'])
        # blank
        win.flip()
        core.wait(params['blank_time']-params['slack'])
        # stimulus
        if task == 'yes-no':
            # stimulus
            stim.draw()
        else: # 2AFC task
            # stimulus
            stim_left.draw()
            stim_right.draw()
        win.flip()
        start = core.getTime()
        core.wait(params['presentation_time']-params['slack'])
        # response
        win.flip()
        keys = event.waitKeys(keyList=['y','Y','n','N','left','right','escape'])
        stop = core.getTime()
        rt = stop - start
        for key in keys:
            if key in ['y','Y','right']:
                r = 1
            elif key in ['n','N','left']:
                r = 0
            elif key == 'escape':
                r = -1
        if r == -1:
            break
        # feedback
        if s==r:
            pos.draw()
        else:
            neg.draw()
        win.flip()
        core.wait(params['feedback_time']-params['slack'])
        # log data
        ndata = {'subject': subject,
                 'task': task,
                 'block': block,
                 'intensity': intensity,
                 'p': p,
                 'trial': int(i),
                 'stimulus': int(s),
                 'response': int(r),
                 'hit': int(s and r),
                 'false_alarm': int(not(s) and r),
                 'correct': int(s==r),
                 'RT': rt}
        data = data.append(ndata, ignore_index=True)
    # after a number of trials have been done save and quit
    if r>-1: # don't save on escape
        data.to_csv(params['filename'],header=True,index=False)
    win.setMouseVisible(True)
    win.close()
    core.quit()
    return data

def summarize(data, group = ['intensity','p'], mode='all'):
    grouped = data.groupby(group)
    # hit rate
    N1 = grouped['stimulus'].sum()
    H = grouped['hit'].sum() / N1
    # false alarm rate
    N0 = grouped.size()-grouped['stimulus'].sum()
    FA = grouped['false_alarm'].sum() / N0
    # percent correct
    N = N0 + N1
    PC = grouped['correct'].sum() / N
    # aggregate
    if mode == 'pc':
        summary = pd.DataFrame([PC, N]).T
        summary.columns=['PC','N']
    elif mode == 'roc':
        summary = pd.DataFrame([FA, H, N0, N1]).T
        summary.columns=['FA','H','N0','N1']
    elif mode == 'all':
        summary = pd.DataFrame([FA, H, PC, N0, N1, N]).T
        summary.columns=['FA','H','PC','N0','N1','N']
    return summary
