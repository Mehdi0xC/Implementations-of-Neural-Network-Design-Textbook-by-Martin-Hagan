# Simple Pattern Recognition and Reconstruction using Hamming Neural Networks
# Developed by Mehdi0xC, Summer 2018
import numpy as np
import hamming

# specify total size of each pattern
height = 5
width = 5
totalSize = height*width
# pattern prototypes
pattern_0 = np.array([
0,1,1,1,0,
1,0,0,0,1,
1,0,0,0,1,
1,0,0,0,1,
0,1,1,1,0
]).reshape((totalSize, 1))

pattern_1 = np.array([
0,0,1,0,0,
0,1,1,0,0,
0,0,1,0,0,
0,0,1,0,0,
0,1,1,1,0
]).reshape((totalSize, 1))

pattern_2 = np.array([
0,1,1,1,0,
1,0,0,0,1,
0,0,1,1,0,
0,1,0,0,0,
1,1,1,1,1
]).reshape((totalSize, 1))

pattern_3 = np.array([
0,1,1,1,0,
1,0,0,0,1,
0,0,1,1,1,
1,0,0,0,1,
0,1,1,1,0
]).reshape((totalSize, 1))

pattern_4 = np.array([
0,0,1,0,0,
0,1,1,0,0,
1,1,1,1,1,
0,0,1,0,0,
0,1,1,1,0
]).reshape((totalSize, 1))

prototypes = np.array([pattern_0.T[0],pattern_1.T[0],pattern_2.T[0],pattern_3.T[0],pattern_4.T[0]])
classifier = hamming.HammingNetwork(prototypes=prototypes)

test_pattern = np.array([
0,0,1,0,0,
0,0,1,0,0,
0,0,1,1,0,
0,0,1,0,0,
0,1,1,1,0
]).reshape((totalSize, 1))

print("output for test pattern : ")
print(classifier.classify(obj=test_pattern))

print("reconstructed pattern is : ")
print(prototypes[np.argmax(classifier.classify(obj=test_pattern),0)].reshape(height, width))
