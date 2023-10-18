#!/usr/bin/env python
# coding: utf-8


import pretty_midi
import mir_eval
import numpy as np
import warnings
import os
            
def normalize_features(features):
    """
    This function normalizes the features to the range [0, 1]
    
    @input features: The array of features.
    @type features: List of float
    
    @return: Normalized features.
    @rtype: List of float
    """
    # normalize tempo
    tempo = ((features[0] - 0) / (300 - 0))
    
    # put pitches into separate variables in new list
    dominantPitch = ((features[1] - 0) / (11 - 0))
    
    # rhythmic score is result of rhythmic analysis
    rhythmicScore = features[2]
    # get quantized MIDI duration    
    if (60 <= features[3] <= 240):
        quantizedDuration = 0.5
    elif (features[3] < 60):
        quantizedDuration = 0
    else:
        quantizedDuration = 1
    
    return [tempo, dominantPitch, rhythmicScore, quantizedDuration]


def get_features(path, normalized = False):
    """
    This function extracts the features from a midi file when given its path.
    
    @input path: The path to the midi file.
    @type path: String
    
    @return: The extracted features.
    @rtype: List of float
    """
    try:
        # Test for Corrupted Midi Files
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            file = pretty_midi.PrettyMIDI(path)
            
            # get tempo
            tempo = file.estimate_tempo()
            
            # get pitches
            pitchArray = file.get_pitch_class_histogram()
            maxPitch = np.argmax(pitchArray)
            maxPitchAmount = pitchArray[maxPitch]
            
            # analyse rhythm
            rhythmAnalysis = mir_eval.beat.information_gain(file.get_beats(), file.get_onsets())
            
            duration = file.get_end_time()                
            
            if normalized == True:
                return normalize_features([tempo, maxPitch, rhythmAnalysis, duration])
            else: 
                return [tempo, maxPitch, maxPitchAmount, rhythmAnalysis, duration]
    except:
        return None


