import numpy as np
import pandas as pd
pd.set_option('precision',2)

from psychopy import core, visual, event, monitors
from imageio import imread
from matplotlib.pyplot import *
from scipy.optimize import minimize
from scipy.stats import norm
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
A = 1-np.mean(A,2)       # and make grayscale white on black
params['signal'] = A.copy()
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
        text = 'Press y if you see an A and n otherwise.\nAny key to start (ESC to quit).'
    elif task == '2AFC':
        text = 'Press left if you see an A on the left and right if you see it on the right.\nAny key to start (ESC to quit).'
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

def run_block(subject,intensity,p=0.5,num_of_blocks=1,task='yes-no'):
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
                           lineColor="lawngreen")
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
    num_of_trials = params['num_of_trials'] * num_of_blocks
    for i in range(num_of_trials):
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
                 'trial': int(i+1),
                 'stimulus': int(s),
                 'response': int(r),
                 'hit': int(s and r),
                 'false_alarm': int(not(s) and r),
                 'correct': int(s==r),
                 'RT': rt}
        data = data.append(ndata, ignore_index=True)
        # after a number of trials have been done save
        if ((i+1) % params['num_of_trials']) == 0:
            data.to_csv(params['filename'],header=True,index=False)
            pc = data[data['block']==block]['correct'].mean()
            text = '%d of %d trials done. %.2f%% correct in last block.\nAny key to continue.'%(i+1,num_of_trials,pc*100)
            block = block + 1
            message = visual.TextStim(win, pos=[0,100],text=text)
            message.draw()
            win.flip()
            event.waitKeys()
    # and quit
    win.setMouseVisible(True)
    win.close()
    try:
        core.quit()
    except:
        print('Done, psychopy core quit.')
    return data

def summarize(data, p=0.5, intensity=None, mode='pc', group=None):
    if p is not None and intensity is not None:
        df = data[data['p']==p and data['intensity']==intensity]
        if group is None:
            group = 'block'
    elif p is not None:
        df = data[data['p']==p]
        if group is None:
            group = ['p','intensity']
    else:
        df = data[data['intensity']==intensity]
        if group is None:
            group = ['intensity','p']
    grouped = df.groupby(group)
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
        summary.columns=['pc','N']
    elif mode == 'roc':
        summary = pd.DataFrame([FA, H, N0, N1]).T
        summary.columns=['f','h','N0','N1']
    elif mode == 'all':
        summary = pd.DataFrame([FA, H, PC, N0, N1, N]).T
        summary.columns=['f','h','pc','N0','N1','N']
    return summary

def weibull(x,pvec,q=0.5):
    # Weibull since intensity cannot be smaller than zero
    # parametrization as in Kuss, Jäkel, & Wichmann (2005)
    m = pvec[0]
    s = pvec[1]
    c = 2*s*m/np.log(2)
    k = np.log(np.log(2))
    y = 1-np.exp(-np.exp(c * (np.log(x)-np.log(m)) + k))
    return y * (1-q) + q

def inv_weibull(y,pvec,q=0.5):
    assert y >= q
    m = pvec[0]
    s = pvec[1]
    c = 2*s*m/np.log(2)
    k = np.log(np.log(2))
    x = np.exp((np.log(np.log(((y-q)/(1-q)-1)*(-1))*(-1))-k)/c+np.log(m))
    return x

def psychometric_function(data):
    s = summarize(data,group='intensity')
    x = s.index.values
    y = s['pc'].values
    n = s['N'].values
    # fit a weibull with least squares
    # we don't want the hassle with lapses and lsq is more robust than ml
    def err(pvec):
        return np.sum(n * (weibull(x,pvec)-y)**2)
    res = minimize(err, [np.mean(s.index), 0.5])
    xx = np.linspace(0,np.max(x)*1.1,1000)
    pvec = res['x']
    pc = [0.55, 0.65, 0.75, 0.85, 0.95]
    thresholds = [inv_weibull(p,pvec) for p in pc]
    for p,t in zip(pc,thresholds):
        plot([0,t,t],[p,weibull(t,pvec),0],color='grey',linestyle=':')
        print('%d%% threshold: %.0f'%(p*100,t))
    plot(xx,weibull(xx,pvec))
    scatter(x,y,marker='o',s=n/2.0)
    xlabel('intensity')
    ylabel('proportion correct')
    grid()
    return thresholds

def plot_roc_confidence(df,color='tab:blue',ses=2):
    for f,h,n0,n1 in zip(df['f'], df['h'], df['N0'], df['N1']):
        f_se = np.sqrt(f*(1-f)/n0)
        h_se = np.sqrt(h*(1-h)/n1)
        plot([f-ses*f_se,f+ses*f_se],[h,h],color=color)
        plot([f,f],[h-ses*h_se,h+ses*h_se],color=color)
        
def plot_roc(df, conf=True, labels=True):
    color = 'tab:blue'
    plot([0,1],[0,1],'k')
    scatter(df['f'], df['h'], s=25, color=color)
    if conf: #let's you see confidence intervals (+-2 standard errors)
        plot_roc_confidence(df,color=color) # but they are huge unless you collect much more data
    if labels:
        for f,h,p in zip(df['f'], df['h'], df.index):
            text(f+0.01, h+0.02, str(p))
    xlabel('False Alarm Rate'), ylabel('Hit Rate'), title('ROC')
    axis('square'), xlim([0,1]), ylim([0,1])
