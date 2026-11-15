#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interference and Facilitation in a Keypress Replication of the Stroop
Colour-Naming Task
-------------------------------------------------------------------------
PSY412 - Attention, Monsoon Semester 2026
Author : Archi Prajapati (AU2320197)
Built for: PsychoPy Coder, v2023.2+

DESIGN (from the proposal)
  * Single-factor, within-subjects: Congruency (congruent / incongruent)
  * 4 ink colours: red, blue, green, yellow  (keys R, B, G, Y)
  * Words: RED, BLUE, GREEN, YELLOW on a grey background (no neutral XXXX condition)
  * Trial: fixation (+, 500 ms) -> word (until keypress, max 2000 ms)
           -> blank jittered ITI (500-800 ms)
  * 12 practice trials (with feedback) + 120 experimental trials
    (60 per condition) in 3 self-paced blocks of 40, randomised within block
  * DVs: RT (ms, correct trials) and accuracy

HOW TO RUN
  Open this file in PsychoPy Coder and press Run (green arrow), or:
      python stroop_keypress_experiment.py
  Press ESCAPE at any time to quit (data collected so far is already saved).

OUTPUT (in ./data/)
  <ID>_stroop_keypress_<timestamp>.csv          one row per trial (practice + main)
  <ID>_stroop_keypress_<timestamp>_summary.txt  per-condition summary for the experimenter
"""

import csv
import os
import random
import sys
from datetime import datetime
from statistics import mean

import psychopy
from psychopy import core, gui, logging, visual
from psychopy.hardware import keyboard

# =============================================================================
# 1. SETTINGS  (everything you might want to tweak lives here)
# =============================================================================
EXP_NAME = 'stroop_keypress'
DEBUG = False                      # True = small window instead of full screen

# --- Stimuli -----------------------------------------------------------------
COLOURS = ['red', 'blue', 'green', 'yellow']
WORD_TEXT = {c: c.upper() for c in COLOURS}      # RED, BLUE, GREEN, YELLOW
CONDITIONS = ['congruent', 'incongruent']

# Ink colours in rgb255. Calibrate/adjust for your monitor if needed.
INK_RGB = {
    'red':    (230, 30, 30),
    'blue':   (40, 90, 245),
    'green':  (30, 165, 60),
    'yellow': (235, 215, 0),
}
BG_RGB = (128, 128, 128)           # grey background
TEXT_RGB = (0, 0, 0)               # instruction text (black)
STIM_HEIGHT = 0.12                 # text height, in 'height' units (fraction of screen height)

# --- Response keys -----------------------------------------------------------
KEY_FOR = {'red': 'r', 'blue': 'b', 'green': 'g', 'yellow': 'y'}
RESP_KEYS = list(KEY_FOR.values())

# --- Timing (seconds) --------------------------------------------------------
FIX_DUR = 0.500
MAX_RT = 2.000
ITI_MIN, ITI_MAX = 0.500, 0.800
FEEDBACK_DUR = 0.600               # practice only

# --- Trial counts ------------------------------------------------------------
N_PER_CONDITION = 60               # -> 120 experimental trials (set 40 for 80 trials)
N_BLOCKS = 3                       # -> 40 trials per block
MIN_VALID_RT = 0.200               # RTs < 200 ms are excluded (analysis plan)

# Avoid the exact same word+ink appearing twice in a row (reduces repetition
# priming). Set to False for fully unconstrained random order.
AVOID_IMMEDIATE_REPEATS = True

# --- Text shown to participants ---------------------------------------------
CONTACT = '<add your email here>'

CONSENT_TEXT = (
    "INFORMED CONSENT\n\n"
    "This is a short psychology experiment on attention (about 10 minutes).\n"
    "You will see words on screen and respond by pressing keys.\n\n"
    "Participation is voluntary and unpaid. Your responses are anonymous and "
    "will be used only for a course project. You may stop at any time by "
    "pressing ESCAPE, without any penalty.\n\n"
    "Press SPACE if you agree to take part."
)

INSTRUCTION_TEXT = (
    "INSTRUCTIONS\n\n"
    "On each trial a '+' appears, followed by a colour word "
    "printed in a coloured ink.\n\n"
    "Your task is to respond to the INK COLOUR, and ignore what the word says.\n"
    "Press the key that matches the ink colour:\n"
    "R = red     B = blue     G = green     Y = yellow\n\n"
    "Respond as quickly and as accurately as you can.\n"
    "Keep your fingers resting near the R, B, G and Y keys."
)

PRACTICE_TEXT = (
    "PRACTICE\n\n"
    "First you will do 12 practice trials. You will get feedback after each "
    "response.\n\nRemember: respond to the INK COLOUR, not the word."
)

MAIN_INTRO_TEXT = (
    "Practice complete.\n\n"
    "Now the real experiment begins. There are {n_blocks} blocks of {n_trials} "
    "trials, and you can rest between blocks. There will be NO feedback now.\n\n"
    "Remember: respond to the INK COLOUR as fast and as accurately as you can."
)

DEBRIEF_TEXT = (
    "Thank you for taking part!\n\n"
    "This study examined the Stroop effect: people are usually slower and "
    "make more errors when a word names a different colour from its ink "
    "(e.g. the word RED in blue ink), because reading is automatic and "
    "competes with naming the colour. We compared this with trials where "
    "the word matched the ink.\n\n"
    "Your data are anonymous. If you have any questions, contact: " + CONTACT +
    "\n\nPress SPACE to finish."
)


# =============================================================================
# 2. TRIAL LIST BUILDERS
# =============================================================================
def make_trial(condition, ink, word):
    meaning = next(c for c in COLOURS if WORD_TEXT[c] == word)
    return dict(condition=condition, ink=ink, word=word,
                word_meaning=meaning, correct_key=KEY_FOR[ink])


def shuffle_no_repeats(trials, rng, max_tries=2000):
    """Shuffle in place; optionally forbid identical consecutive word+ink."""
    for _ in range(max_tries):
        rng.shuffle(trials)
        if not AVOID_IMMEDIATE_REPEATS:
            return
        if all(not (a['word'] == b['word'] and a['ink'] == b['ink'])
               for a, b in zip(trials, trials[1:])):
            return
    # extremely unlikely fallback: keep the last shuffle as-is


def build_practice(rng):
    """12 practice trials: 6 congruent + 6 incongruent, every ink colour 3 times."""
    trials = []
    inks = COLOURS[:]
    rng.shuffle(inks)
    for i, ink in enumerate(inks):
        others = [c for c in COLOURS if c != ink]
        trials.append(make_trial('congruent', ink, WORD_TEXT[ink]))
        trials.append(make_trial('incongruent', ink, WORD_TEXT[rng.choice(others)]))
        if i < 2:      # two inks get an extra congruent trial ...
            trials.append(make_trial('congruent', ink, WORD_TEXT[ink]))
        else:          # ... the other two get an extra incongruent trial
            trials.append(make_trial('incongruent', ink, WORD_TEXT[rng.choice(others)]))
    shuffle_no_repeats(trials, rng)
    return trials


def build_main_blocks(rng):
    """
    120 trials = 60 per condition, split into 3 blocks of 40.
    * Congruent: 15 trials per ink colour.
    * Incongruent: 15 trials per ink colour, cycling through the 3 conflicting
      words (5 each, fully balanced).
    * Each block gets 20 trials of each condition.
    * Trial order is randomised within each block.
    """
    reps_per_ink = N_PER_CONDITION // len(COLOURS)
    pools = {c: [] for c in CONDITIONS}
    for ink in COLOURS:
        others = [c for c in COLOURS if c != ink]
        rng.shuffle(others)
        for i in range(reps_per_ink):
            pools['congruent'].append(make_trial('congruent', ink, WORD_TEXT[ink]))
            pools['incongruent'].append(
                make_trial('incongruent', ink, WORD_TEXT[others[i % len(others)]]))

    blocks = [[] for _ in range(N_BLOCKS)]
    base, rem = divmod(N_PER_CONDITION, N_BLOCKS)
    for c_idx, cond in enumerate(CONDITIONS):
        pool = pools[cond]
        rng.shuffle(pool)
        extra_blocks = {(c_idx + k) % N_BLOCKS for k in range(rem)}
        start = 0
        for b in range(N_BLOCKS):
            size = base + (1 if b in extra_blocks else 0)
            blocks[b].extend(pool[start:start + size])
            start += size
    for blk in blocks:
        shuffle_no_repeats(blk, rng)
    return blocks


# =============================================================================
# 3. EXPERIMENT CLASS
# =============================================================================
FIELDNAMES = [
    'participant', 'age', 'gender', 'normal_vision', 'no_colour_deficiency',
    'date', 'psychopy_version', 'frame_rate_hz', 'seed',
    'phase', 'block', 'trial_in_block', 'trial_overall',
    'condition', 'word', 'word_meaning', 'ink', 'correct_key',
    'resp_key', 'rt_ms', 'correct', 'timeout', 'valid_rt', 'iti_ms',
]


class StroopExperiment:
    def __init__(self, info, out_path, seed):
        self.info = info
        self.out_path = out_path
        self.seed = seed
        self.rng = random.Random(seed)
        self.rows = []
        self.trial_counter = 0

        # --- window ----------------------------------------------------------
        self.win = visual.Window(
            size=(1280, 720) if DEBUG else (1024, 768),
            fullscr=not DEBUG, color=BG_RGB, colorSpace='rgb255',
            units='height', allowGUI=False, winType='pyglet')
        self.win.mouseVisible = False
        fps = self.win.getActualFrameRate(nIdentical=10, nMaxFrames=120,
                                          nWarmUpFrames=10, threshold=1)
        self.fps = fps if fps else 60.0
        self.kb = keyboard.Keyboard()

        # --- stimuli ---------------------------------------------------------
        self.fixation = visual.TextStim(
            self.win, text='+', height=STIM_HEIGHT, color=TEXT_RGB,
            colorSpace='rgb255', font='Arial', bold=True)
        self.stim = visual.TextStim(
            self.win, text='', height=STIM_HEIGHT, color=TEXT_RGB,
            colorSpace='rgb255', font='Arial', bold=True)
        self.feedback = visual.TextStim(
            self.win, text='', height=0.06, color=TEXT_RGB,
            colorSpace='rgb255', font='Arial', bold=True)
        self.message = visual.TextStim(
            self.win, text='', height=0.04, wrapWidth=1.4, color=TEXT_RGB,
            colorSpace='rgb255', font='Arial', alignText='center',
            pos=(0, 0.05))
        self.prompt = visual.TextStim(
            self.win, text='', height=0.035, color=TEXT_RGB,
            colorSpace='rgb255', font='Arial', pos=(0, -0.42))
        # key legend: coloured patch with its response letter underneath
        self.legend = []
        for i, colour in enumerate(COLOURS):
            x = (i - 1.5) * 0.25
            patch = visual.Rect(self.win, width=0.12, height=0.12, pos=(x, -0.27),
                                fillColor=INK_RGB[colour], lineColor=None,
                                colorSpace='rgb255')
            label = visual.TextStim(self.win, text=KEY_FOR[colour].upper(),
                                    height=0.06, pos=(x, -0.37), color=TEXT_RGB,
                                    colorSpace='rgb255', font='Arial', bold=True)
            self.legend += [patch, label]

        # --- output file (flushed after every trial) -------------------------
        self.fh = open(out_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.fh, fieldnames=FIELDNAMES)
        self.writer.writeheader()
        self.fh.flush()

    # ------------------------------------------------------------------ utils
    def frames(self, seconds):
        return max(1, int(round(seconds * self.fps)))

    def quit(self):
        try:
            self.fh.flush()
            self.fh.close()
        finally:
            self.win.close()
            core.quit()

    def check_escape(self):
        if self.kb.getKeys(keyList=['escape'], waitRelease=False):
            self.quit()

    def show_screen(self, text, prompt='Press SPACE to continue',
                    legend=False, keys=('space',)):
        """Show a message and wait for one of `keys`. ESC quits."""
        self.message.text = text
        self.message.pos = (0, 0.12) if legend else (0, 0.05)
        self.prompt.text = prompt
        self.kb.clearEvents()
        while True:
            self.message.draw()
            if legend:
                for item in self.legend:
                    item.draw()
            if prompt:
                self.prompt.draw()
            self.win.flip()
            pressed = self.kb.getKeys(keyList=list(keys) + ['escape'],
                                      waitRelease=False)
            if pressed:
                if any(k.name == 'escape' for k in pressed):
                    self.quit()
                return pressed[0].name

    # ------------------------------------------------------------- one trial
    def run_trial(self, trial, phase, block, trial_in_block):
        self.trial_counter += 1
        iti = self.rng.uniform(ITI_MIN, ITI_MAX)

        # 1) fixation cross (500 ms)
        for _ in range(self.frames(FIX_DUR)):
            self.fixation.draw()
            self.win.flip()
            self.check_escape()

        # 2) colour word: until response, max 2000 ms. RT from stimulus onset.
        self.stim.text = trial['word']
        self.stim.color = INK_RGB[trial['ink']]
        self.kb.clearEvents()
        self.win.callOnFlip(self.kb.clock.reset)                         # t = 0 at onset
        self.win.callOnFlip(self.kb.clearEvents, eventType='keyboard')
        resp_key, rt = None, None
        for _ in range(self.frames(MAX_RT)):
            self.stim.draw()
            self.win.flip()
            keys = self.kb.getKeys(keyList=RESP_KEYS + ['escape'],
                                   waitRelease=False)
            if keys:
                if any(k.name == 'escape' for k in keys):
                    self.quit()
                resp_key, rt = keys[0].name, keys[0].rt
                break
        if rt is not None and rt > MAX_RT:       # response landed after deadline
            resp_key, rt = None, None

        timeout = resp_key is None
        correct = (not timeout) and resp_key == trial['correct_key']
        valid = bool(correct and rt >= MIN_VALID_RT)

        # 3) practice feedback
        if phase == 'practice':
            self.feedback.text = ('Too slow!' if timeout
                                  else 'Correct!' if correct else 'Incorrect')
            for _ in range(self.frames(FEEDBACK_DUR)):
                self.feedback.draw()
                self.win.flip()
                self.check_escape()

        # 4) blank jittered ITI (500-800 ms)
        for _ in range(self.frames(iti)):
            self.win.flip()
            self.check_escape()

        # 5) save
        row = dict(
            participant=self.info['participant'], age=self.info['age'],
            gender=self.info['gender'],
            normal_vision=self.info['normal_vision'],
            no_colour_deficiency=self.info['no_colour_deficiency'],
            date=self.info['date'], psychopy_version=psychopy.__version__,
            frame_rate_hz=round(self.fps, 2), seed=self.seed,
            phase=phase, block=block, trial_in_block=trial_in_block,
            trial_overall=self.trial_counter,
            condition=trial['condition'], word=trial['word'],
            word_meaning=trial['word_meaning'], ink=trial['ink'],
            correct_key=trial['correct_key'],
            resp_key='' if timeout else resp_key,
            rt_ms='' if timeout else round(rt * 1000, 1),
            correct=int(correct), timeout=int(timeout), valid_rt=int(valid),
            iti_ms=round(iti * 1000, 1))
        self.writer.writerow(row)
        self.fh.flush()
        self.rows.append(row)
        return row

    # --------------------------------------------------------------- summary
    def write_summary(self):
        main = [r for r in self.rows if r['phase'] == 'main']
        lines = ['Participant: %s' % self.info['participant'],
                 'Main trials: %d' % len(main), '']
        stats = {}
        for cond in CONDITIONS:
            rs = [r for r in main if r['condition'] == cond]
            rts = [r['rt_ms'] for r in rs if r['valid_rt']]
            stats[cond] = dict(
                n=len(rs),
                acc=mean(r['correct'] for r in rs) if rs else float('nan'),
                rt=mean(rts) if rts else float('nan'),
                n_valid=len(rts))
            lines.append('%-12s n=%3d  accuracy=%.3f  mean correct RT=%.1f ms (n valid=%d)' % (
                cond, stats[cond]['n'], stats[cond]['acc'],
                stats[cond]['rt'], stats[cond]['n_valid']))
        lines += ['',
                  'Interference / Stroop effect (incongruent - congruent): %.1f ms' %
                  (stats['incongruent']['rt'] - stats['congruent']['rt']),
                  '',
                  'RT exclusions: incorrect trials, timeouts, RT < %d ms.' %
                  int(MIN_VALID_RT * 1000)]
        text = '\n'.join(lines)
        print(text)
        with open(self.out_path.replace('.csv', '_summary.txt'), 'w',
                  encoding='utf-8') as f:
            f.write(text + '\n')

    # ------------------------------------------------------------------- run
    def run(self):
        self.show_screen(CONSENT_TEXT, prompt='')
        self.show_screen(INSTRUCTION_TEXT, legend=True)

        # ---- practice (12 trials, feedback) ----
        self.show_screen(PRACTICE_TEXT, prompt='Press SPACE to start practice')
        for i, trial in enumerate(build_practice(self.rng), start=1):
            self.run_trial(trial, 'practice', 0, i)

        # ---- main experiment (3 self-paced blocks) ----
        blocks = build_main_blocks(self.rng)
        n_trials = len(blocks[0])
        self.show_screen(MAIN_INTRO_TEXT.format(n_blocks=N_BLOCKS, n_trials=n_trials),
                         prompt='Press SPACE to start Block 1', legend=True)
        for b, block_trials in enumerate(blocks, start=1):
            for i, trial in enumerate(block_trials, start=1):
                self.run_trial(trial, 'main', b, i)
            if b < N_BLOCKS:
                self.show_screen(
                    "Block %d of %d complete.\n\nTake a short break if you like.\n\n"
                    "Remember: respond to the INK COLOUR." % (b, N_BLOCKS),
                    prompt='Press SPACE when you are ready for Block %d' % (b + 1),
                    legend=True)

        # ---- wrap up ----
        self.write_summary()
        self.show_screen(DEBRIEF_TEXT, prompt='')
        self.quit()


# =============================================================================
# 4. MAIN
# =============================================================================
def get_participant_info():
    info = {
        'participant': '',
        'age': '',
        'gender': ['female', 'male', 'other / prefer not to say'],
        'normal_vision': ['yes', 'no'],
        'no_colour_deficiency': ['yes', 'no'],
    }
    while True:
        dlg = gui.DlgFromDict(info, title='Stroop task (keypress)',
                              sortKeys=False,
                              labels={'normal_vision': 'Normal / corrected-to-normal vision?',
                                      'no_colour_deficiency': 'No known colour-vision deficiency?'})
        if not dlg.OK:
            core.quit()
        if str(info['participant']).strip():
            break
    if info['normal_vision'] == 'no' or info['no_colour_deficiency'] == 'no':
        d = gui.Dlg(title='Not eligible')
        d.addText('Eligibility requires normal/corrected vision and no colour-vision '
                  'deficiency. The experiment will now close.')
        d.show()
        core.quit()
    info['date'] = datetime.now().strftime('%Y-%m-%d_%H%M')
    return info


def main():
    logging.console.setLevel(logging.WARNING)
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    os.makedirs('data', exist_ok=True)

    info = get_participant_info()
    out_path = os.path.join('data', '%s_%s_%s.csv' % (
        str(info['participant']).strip(), EXP_NAME, info['date']))
    seed = int(datetime.now().timestamp())

    exp = StroopExperiment(info, out_path, seed)
    exp.run()


if __name__ == '__main__':
    main()
