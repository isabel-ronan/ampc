from flask import Flask, render_template, request, redirect, url_for, flash, abort
import json
import os.path
from werkzeug.utils import secure_filename
from Bio.PDB import *
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
import pretty_midi
import mir_eval
import warnings
import random
from midiMaker import *
from musicalFeatures import *
import shutil

app = Flask(__name__)
app.secret_key = "super secret key"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/midi-generator', methods=['GET', 'POST'])
def generate_midi():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename.endswith('.pdb'):

            midiPaths = {}

            if os.path.exists('midiPaths.json'):
                with open('midiPaths.json') as midiPaths_file:
                    midiPaths = json.load(midiPaths_file)

            if f.filename in midiPaths.keys():
                print("File already exists.")
                dst_path = "./static/music/" + f.filename.replace(".pdb", "") + ".mid"
                return render_template('midi-generator.html', fileToPlay = dst_path)

            proteinPath = './static/proteinsUploaded/' + f.filename
            f.save(proteinPath)
            
            # initialise musical feature arrays
            featureList = []
            arrayOfPaths = []
            # MIDI scale being used
            majorMidiScaleC = [60, 62, 64, 65, 67, 69, 71, 72]
            # number of instruments used
            numberOfInstruments = 4
            # all the functions that must be called
            centerOfMass, coordinateArray, normalised_bFactors = loadProtein(proteinPath)
            listOfDistancesInteger, listOfDistancesFloat = getDistancesFromCenter(centerOfMass, coordinateArray)
            labelledDataX, labelledDataY = mappingDistances(majorMidiScaleC, listOfDistancesInteger, listOfDistancesFloat, coordinateArray)
            knn = trainKNN(labelledDataX, labelledDataY)
            folderPath = os.path.join('./static', f.filename.replace(".pdb", ""))
            if os.path.isdir(folderPath):
                pass
            else:
                os.mkdir(folderPath)
            for i in range(20):
                xyz, xyzWithDistances, threshold = getPlaneTraversalData(coordinateArray, listOfDistancesFloat)
                planeVariable = (int((len(xyz)) / 20) * (i+1))
                planePoints, arrayOfLengths, midiToMap = proteinPlaneSweep([planeVariable - 2, planeVariable - 1, planeVariable], xyz, threshold)
                testPath =  folderPath + '/' + f.filename.replace(".pdb", "-") + str(i) + ".mid"
                try:
                    midiOutput = makeMIDI(listOfDistancesFloat, xyzWithDistances, midiToMap, knn, normalised_bFactors, numberOfInstruments, testPath, writeFile = True, moreRhythmic = True)
                except Exception:
                    print("MIDI file could not be generated.")
                # analyse
                try:
                    # test for corrupted MIDI files
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        features = get_features(testPath, normalized = False)
                        featureList.append(features)
                        arrayOfPaths.append(testPath)
                except:
                        print("This protein could not be musically analysed.")

            # determine most musical MIDI file
            bestRhythm = 0
            dominantPitch = 0
            durationIndex = 0
            longestDuration = 0
            rhythmIndex = 0
            dominantPitchIndex = 0
            overallScore = 0
            overallIndex = 0

            for i in range(len(featureList)):
                try:
                    averageScore = sum(featureList[i]) / len(featureList[i])
                    if averageScore > overallScore:
                        overallScore = averageScore
                        overallIndex = i
                    if featureList[i][4] > longestDuration:
                        longestDuration = featureList[i][4]
                        durationIndex = i
                    if featureList[i][3] > bestRhythm:
                        bestRhythm = featureList[i][3]
                        rhythmIndex = i
                    if featureList[i][2] > dominantPitch:
                        dominantPitch = featureList[i][2]
                        dominantPitchIndex = i
                except:
                    print("Features could not be analysed")
            
            src_path = arrayOfPaths[overallIndex]
            dst_path = "./static/music/" + f.filename.replace(".pdb", "") + ".mid"
            shutil.copy(src_path, dst_path)
            shutil.rmtree(folderPath)
            midiPaths[f.filename] = {'midi': dst_path}
            with open('midiPaths.json', 'w') as url_file:
                json.dump(midiPaths, url_file)
            return render_template('midi-generator.html', fileToPlay = dst_path)



        else:
            flash('Please upload a valid PDB file.')
            return redirect(url_for('home'))

    else:
        return redirect(url_for('home'))
    
@app.route('/generated-files')
def generated_files():
    midiPaths = {}
    if os.path.exists('midiPaths.json'):
        with open('midiPaths.json') as midiPaths_file:
            midiPaths = json.load(midiPaths_file)
        return render_template('all-midi-files.html', codes=midiPaths.keys())
    else:
        return render_template('no-midi-files.html', codes=['Sorry! No MIDI files have been generated yet!', 'Why not try generating your own MIDI file?'])

@app.route('/<string:code>')
def redirect_to_url(code):
    path = "./static/music/" + code.replace(".pdb", "") + ".mid"
    return render_template('midi-generator.html', fileToPlay = path)

@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404

if __name__ == "__main__":
        app.run()

