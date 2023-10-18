#!/usr/bin/env python
# coding: utf-8

# import libraries
# for protein data extraction
from Bio.PDB import *
# for math
import numpy as np
# for ML
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
# for MIDI
import pretty_midi

# load protein data
def loadProtein(path):
    '''
    this function returns extracted protein data from pdb file
    @input path: location of the pdb file
    @input type: String
    
    @return: center of mass, atom coordinates, normalised combination of b-factors and occupancies
    @rtype: 1D float array, 2D float array, 1D float array
    '''
    # create a structure object
    p = PDBParser()
    # create a structure object
    structure = p.get_structure('input', path)
    # get the center of mass
    centerOfMass = structure.center_of_mass()
    
    coordinateArray = [] # used for MIDI pitch determination - distance from center of mass used for velocity
    bFactors = [] # used for duration
    occupancies = [] # also used for duration

    # create lists of coordinate values
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coordinateArray.append(atom.get_coord())
                    bFactors.append(atom.get_bfactor())
                    occupancies.append(atom.get_occupancy())
                    
    # normalise b-factors with respect to occupancies - for duration
    quantized_bFactors = []
    for i in range(len(bFactors)):
        quantizedBFactor = 0
        if bFactors[i] < 25:
            quantizedBFactor = 0.25
        elif bFactors[i] >= 25 and bFactors[i] <= 50: 
            quantizedBFactor = 0.50
        elif bFactors[i] > 50 and bFactors[i] < 75:
            quantizedBFactor = 0.75
        else:
            quantizedBFactor = 1.0
        quantized_bFactors.append(round((quantizedBFactor * occupancies[i]), 1))
        
    return centerOfMass, coordinateArray, quantized_bFactors

def getDistancesFromCenter(centerOfMass, coordinateArray):
    '''
    this function calculates atom coordinate distances from the protein's center of mass
    
    @input centerOfMass: protein's center of mass
    @type centerOfMass: 1D float array
    @input coordinateArray: atom coordinates of protein
    @type coordinateArray: 2D float array
    
    @return: atom coordinate distances from the center of mass
    @rtype: 1D integer array, 1D float array
    
    '''
    def distanceFromCenter(pointA, pointB):
        x1 = pointA[0]
        y1 = pointA[1]
        z1 = pointA[2]
        x2 = pointB[0]
        y2 = pointB[1]
        z2 = pointB[2]

        distanceFromCenter = (((x2 - x1)**2) + ((y2 - y1)**2) + ((z2 - z1)**2))**0.5
        return distanceFromCenter   
    
    distancesFromCentersArrayInt = []
    distancesFromCentersArrayFlt = []
    for i in range(len(coordinateArray)):
        distancesFromCentersArrayInt.append(int(distanceFromCenter(centerOfMass, coordinateArray[i])))
        distancesFromCentersArrayFlt.append(distanceFromCenter(centerOfMass, coordinateArray[i]))
    
    return distancesFromCentersArrayInt, distancesFromCentersArrayFlt

def mappingDistances(midiNotes, listOfDistancesInteger, listOfDistancesFloat, coordinateArray):
    '''
    this function separates the data between labelled and unlabelled data based on distance
    
    @input midiNotes: list of MIDI notes to map to 
    @type midiNotes: 1D integer array
    @input listOfDistancesInteger: atom coordinate distances from the center of mass
    @type listOfDistancesInteger: 2D integer array
    @input listOfDistancesFloat: atom coordinate distances from the center of mass
    @type listOfDistancesFloat: 2D float array
    @input coordinateArray: atom coordinates of protein
    @type coordinateArray: 2D float array
    
    @return: labelled data x and y for KNN training
    @rtype: 2D float array, 1D integer array
    '''
    # coordinate distance increment between MIDI notes for labelling
    increment = int((max(listOfDistancesInteger) - min(listOfDistancesInteger)) / len(midiNotes))
    # distances to map to
    do = min(listOfDistancesInteger)
    re = do + increment
    mi = re + increment
    fa = mi + increment
    sol = fa + increment
    la = sol + increment
    ti = la + increment
    highDo = ti + increment
    # data to map
    xCoords = [val[0] for val in coordinateArray]
    yCoords = [val[1] for val in coordinateArray]
    zCoords = [val[2] for val in coordinateArray]
    data = list(zip(xCoords, yCoords, zCoords, listOfDistancesInteger, listOfDistancesFloat))
    # arrays for knn training and testing
    labelledDataX = []
    labelledDataY = []
    for i in range(len(data)):
        if (data[i][3] == do):
            labelledDataX.append((data[i][0], data[i][1], data[i][2], data[i][4]))
            labelledDataY.append(midiNotes[0])
        elif (data[i][3] == re):
            labelledDataX.append((data[i][0], data[i][1], data[i][2], data[i][4]))
            labelledDataY.append(midiNotes[1])
        elif (data[i][3] == mi):
            labelledDataX.append((data[i][0], data[i][1], data[i][2], data[i][4]))
            labelledDataY.append(midiNotes[2])
        elif (data[i][3] == fa):
            labelledDataX.append((data[i][0], data[i][1], data[i][2], data[i][4]))
            labelledDataY.append(midiNotes[3])
        elif (data[i][3] == sol):
            labelledDataX.append((data[i][0], data[i][1], data[i][2], data[i][4]))
            labelledDataY.append(midiNotes[4])
        elif (data[i][3] == la):
            labelledDataX.append((data[i][0], data[i][1], data[i][2], data[i][4]))
            labelledDataY.append(midiNotes[5])
        elif (data[i][3] == ti):
            labelledDataX.append((data[i][0], data[i][1], data[i][2], data[i][4]))
            labelledDataY.append(midiNotes[6])
        elif (data[i][3] == highDo):
            labelledDataX.append((data[i][0], data[i][1], data[i][2], data[i][4]))
            labelledDataY.append(midiNotes[7])
        else:
            pass
    return labelledDataX, labelledDataY

def trainKNN(labelledDataX, labelledDataY):
    '''
    @input labelledDataX: list of labelled coordinates to train KNN 
    @type labelledDataX: 2D float array
    @input labelledDataY: list of labels as MIDI notes
    @type labelledDataY: 1D integer array
    
    @return: the KNN classifier trained on the labelled data
    @rtype: sklearn.neighbors.KNeighboursClassifier
    '''
    # make knn object
    knn = KNeighborsClassifier(n_neighbors = 2)
    # split data into test and train
    x_train, x_test, y_train, y_test = train_test_split(labelledDataX, labelledDataY, random_state=0)
    # train knn model
    knn.fit(x_train, y_train)
    # assess knn accuracy
    print("KNN Model Accuracy: " + str(knn.score(x_test, y_test)))
    return knn

def getPlaneTraversalData(coordinateArray, listOfDistancesFloat):
    '''
    this function calculates the plane traversal values to sweep the protein coordinate data
    
    @input coordinateArray: atom coordinates of protein
    @type coordinateArray: 2D float array
    @input listOfDistancesFloat: atom coordinate distances from the center of mass
    @type listOfDistancesFloat: 2D float array
    
    @return: list of 3D coordinates to use in plane traversal, list of features to use in note prediction, point-to-plane threshold
    @rtype: 2D float array, 2D float array, float 
    '''
    # data to zip
    xCoords = [val[0] for val in coordinateArray]
    yCoords = [val[1] for val in coordinateArray]
    zCoords = [val[2] for val in coordinateArray]
    # create array for plane traversal
    xyz = list(list(x) for x in zip(xCoords, yCoords, zCoords))
    xyz = np.array(xyz)
    
    # create array of features for prediction
    xyzWithDistances = list(list(x) for x in zip(xCoords, yCoords, zCoords, listOfDistancesFloat))  
    xyzWithDistances = np.array(xyzWithDistances) 
    
    # threshold set-up
    tree = KDTree(np.array(xyz), leaf_size=2)
    # tree query will average k nearest neighbours for each point in the point cloud
    # two arrays produced - point distances and point indexes respectively
    nearest_dist, nearest_ind = tree.query(xyz, k=8)
    # average over each neighbor candidate sorted from the closes to the farthest 
    # filtering out the first element per row in list (as each point calculates the distance from itself)
    arrayOfThresholdPossibilities = np.mean(nearest_dist[:,1:], axis = 0)
    # use average closest distance to determine the threshold
    threshold = arrayOfThresholdPossibilities[0] * 0.1
    
    return xyz, xyzWithDistances, threshold

def get_plane_equation_from_points(P, Q, R): 
    '''
    this function calculates a plane equation given 3 3D coordinate points
    
    @input P: atom coordinate of protein
    @type P: 1D float array
    @input Q: atom coordinate of protein
    @type Q: 1D float array
    @input R: atom coordinate of protein
    @type R: 1D float array
    
    @return: coefficients (a, b, c) and constant (d) of plane equation
    @rtype: float, float, float, float
    '''
    x1, y1, z1 = P
    x2, y2, z2 = Q
    x3, y3, z3 = R 
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return a, b, c, d 

def proteinPlaneSweep(indexNumbers, xyz, threshold, planeSweepAmount = 1000):
    '''
    this function sweeps the protein point data in a plane-like fashion and 
    calculates all points found on certain planes during the sweep
    
    @input indexNumbers: index numbers of points used to determine the plane
    @type indexNumbers: list of integer values
    @input xyz: points (with respect to indexNumbers) used to determine the plane equation 
    @type xyz: 2D float array
    @input threshold: degree of 'nearness' to the plane required to classify the point in a certain planar category
    @type threshold: float
    @input planeSweepAmount: amount of iterations in both directions
    @type planeSweepAmount: float
    
    @return: list of points on each plane, list of number of points on each plane, list of index numbers of points on each plane
    @rtype: 2D float array, integer list, integer list
    '''
    
    pts = xyz[indexNumbers]
    pts
    
    a, b, c, d = get_plane_equation_from_points(pts[0], pts[1], pts[2])
    
    planePointsA = [] # will be used to visualise protein traversed
    planePointsB = [] # will be used to visualise protein traversed
    arrayOfLengthsA = [] # will be used to determine number of MIDI tracks
    arrayOfLengthsB = [] # will be used to determine number of MIDI tracks
    midiToMapA = [] # stored indexes for knn MIDI prediction
    midiToMapB = [] # stored indexes for knn MIDI prediction
    planeSweepAmountA = planeSweepAmount
    planeSweepAmountB = planeSweepAmount
    
    planeParallel = d
    while planeSweepAmountA:
        planeSweepAmountA -= 1
        planeParallel += 1
        distance = (a * xyz[:,0] + b * xyz[:,1] + c * xyz[:,2] + planeParallel) / (np.sqrt((a ** 2) + (b ** 2) + (c ** 2)))
        idx_candidates = np.where((np.abs(distance) <= threshold))[0]
        # if there are points on the plane, record them
        if (len(idx_candidates) != 0):
            arrayOfLengthsA.append(len(idx_candidates))
            midiToMapA.append(idx_candidates)
            for i in range(len(idx_candidates)):
                planePointsA.append(xyz[idx_candidates[i]])
    
    planeParallel = d
    while planeSweepAmountB:
        planeSweepAmountB -= 1
        planeParallel -= 1
        distance = (a * xyz[:,0] + b * xyz[:,1] + c * xyz[:,2] + planeParallel) / (np.sqrt((a ** 2) + (b ** 2) + (c ** 2)))
        idx_candidates = np.where((np.abs(distance) <= threshold))[0]
        # if there are points on the plane, record them
        if (len(idx_candidates) != 0):
            arrayOfLengthsB.append(len(idx_candidates))
            midiToMapB.append(idx_candidates)
            for i in range(len(idx_candidates)):
                planePointsB.append(xyz[idx_candidates[i]])
                
                
    planePointsB.reverse()
    arrayOfLengthsB.reverse()
    midiToMapB.reverse()

    
    planePoints = np.unique((planePointsB + planePointsA), axis=0)
    arrayOfLengths = np.unique((arrayOfLengthsB + arrayOfLengthsA), axis=0)
    midiToMap = []
    for i in range(len(midiToMapB)):
        if len(midiToMapB[i]) == 1:
            midiToMap.append(midiToMapB[i])
        else:
            midiToMap.append(np.unique((midiToMapB[i]), axis=0))
    for i in range(len(midiToMapA)):
        if len(midiToMapA[i]) == 1:
            midiToMap.append(midiToMapA[i])
        else:
            midiToMap.append(np.unique((midiToMapA[i]), axis=0))
    
    return planePoints, arrayOfLengths, midiToMap

def makeMIDI(listOfDistancesFloat, xyzWithDistances, midiToMap, knn, quantized_bFactors, maxTrackAmount, fileName, writeFile = True, moreRhythmic = True):
    '''
    
    '''
    # normalise distances from center of mass - for velocity
    normalised_distances = []
    for i in range(len(listOfDistancesFloat)):
        normalizedDistance = ((listOfDistancesFloat[i] - min(listOfDistancesFloat)) / (max(listOfDistancesFloat) - min(listOfDistancesFloat)) * 100)
        normalised_distances.append(normalizedDistance)
    # set note time to zero to begin with
    noteTime = 0
    noteDuration = 0
    # create a MIDI object
    output_midi = pretty_midi.PrettyMIDI()
    listOfInstruments = []
    trackCounter = 0
    try:
        for i in range(maxTrackAmount):
            instrument = pretty_midi.Instrument(program = i)
            listOfInstruments.append(instrument)     
    except:
        print("This protein plane could not be used musically.")
        return
    for i in range(len(midiToMap)):
        for n in range(len(midiToMap[i])):
            if moreRhythmic == True:
                noteDuration = quantized_bFactors[midiToMap[i][n]]
            else: 
                noteDuration = 0.125
            new_prediction = [((xyzWithDistances[int(midiToMap[i][n])][0], xyzWithDistances[int(midiToMap[i][n])][1], xyzWithDistances[int(midiToMap[i][n])][2], xyzWithDistances[int(midiToMap[i][n])][3]))]
            # create a note instance
            note = pretty_midi.Note(velocity = int(normalised_distances[midiToMap[i][n]]), pitch = int(knn.predict(new_prediction)), start = noteTime, end = (noteTime + noteDuration))
            # add note to the instrument
            if trackCounter > (maxTrackAmount - 1):
                trackCounter = 0
                listOfInstruments[trackCounter].notes.append(note)
                trackCounter += 1
            else:
                listOfInstruments[trackCounter].notes.append(note)
                trackCounter += 1
        noteTime = noteTime + noteDuration
    # add the instruments to the PrettyMIDI object
    for i in range(maxTrackAmount):
        output_midi.instruments.append(listOfInstruments[i])    
    if writeFile == True:
        # write out the MIDI data
        output_midi.write(fileName)
    return output_midi      