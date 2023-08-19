import numpy as np

"""
This module creates virtual representations Discrete Linear Networks for use as scalar fields.

:Classes:
    DiscreteLinearField
"""

class DiscreteLinearField:
    """
    This class encodes the parameters of a random Discrete Linear Network (DLN), samples realizations, and calculates the mollified indicator field of the DLN.

    :Attributes:
        nLines: int
            number of lines
        xMin: float
            minimum realizable x-coordinate of a line's center
        xMax: float
            maximum realizable x-coordinate of a line's center
        yMin: float
            minimum realizable y-coordinate of a line's center
        xMax: float
            maximum realizable y-coordinate of a line's center
        lengthMax: float
            maximum realizable length
        lengthMax: float
            minimum realizable length
        contrastMax: float
            maximum realizable contrast
        contrastMin: float
            minimum realizable contrast
        pointsMat: nLines-by-2 numpy array
            each row contains the x then y coordinate of a line's center
        anglesMat: nLines-by-1 numpy array
            each row contains a line's angle
        lengthsMat: nLines-by-1 numpy array
            each row contains a line's length
        contrastsMat: nLines-by-1 numpy array
            each row contains a line's contrast
        mollifierFloorWidth: float
            the thickness of the lines in mollifiedIndicator
        mollifierSigma: float
            the spread of the function represented by  mollifiedIndicator
        distanceFunction(x,y): lambda function
            vector-valued function, each entry is the distance to a different line
        mollifiedIndicator(x,y): lambda function
            scalar-valued function, mollified indicator function of a sample set of lines (function equals 1 away from lines)

        :Methods:
            __init__(nLines):
                Initializes an instance of this class (DiscreteLinearField) with a number of lines, nLines.
            samplePointsUnif(xMin,xMax,yMin,yMax):
                Sample line centers uniformly. Updates xMin, xMax, yMin, yMax and pointsMat.
            sampleAnglesUnif():
                Samples angles uniformly. Updates anglesMat.
            sampleLengthsUnif(lengthMin, lengthMax):
                Sample line lengths uniformly. Updates lengthsMin, lengthsMax and pointsMat.
            sampleContrastsUnif(contrastMin, contrastMax):
                Sample line contrasts uniformly. Updates contrastMin and contrastMax.
            computeStandardFormMat(anglesMatrix,pointsMatrix):
                Returns the standard form parameters of a set of lines (ax + by + c = 0) as a nLines-by-3 matrix.
            distanceFunctionEachLine(x,y,standardFormMat):
                Returns the distance from a point (x,y) to a set of lines.
            linearField2Distance():
                Updates the attribute distanceFunction(x,y).
            linearField2MollifiedIndicator(mollifierFloorWidth,mollifierSigma):
                Updates the attribute mollifiedIndicator(x,y), mollifierFloorWidth and mollifierSigma.
    """

    def __init__(self,nLines):
        """
        Initializes an instance of this class. 

        :Parameters:
            nLines: int
                number of lines
        """

        self.nLines = nLines

    def samplePointsUnif(self,xMin,xMax,yMin,yMax):
        """
        Sample line centers uniformly. Updates xMin, xMax, yMin, yMax and pointsMat.

        :Parameters:
            xMin: float
                minimum realizable x-coordinate of a line's center
            xMax: float
                maximum realizable x-coordinate of a line's center
            yMin: float
                minimum realizable y-coordinate of a line's center
            xMax: float
                maximum realizable y-coordinate of a line's center
        """

        nLines = self.nLines
        self.xMin = xMin
        self.xMax = xMax
        self.yMin = yMin
        self.yMax = yMax
        self.pointsMat = np.ones((nLines,2))
        self.pointsMat[:,0] = (
            (xMax-xMin)*np.random.rand(1,nLines)+xMin
        )
        self.pointsMat[:,1] = (
            (yMax-yMin)*np.random.rand(1,nLines)+yMin
        )

    def sampleAnglesUnif(self):
        """
        Samples angles uniformly. Updates anglesMat.
        """

        nLines = self.nLines
        self.anglesMat = (np.random.rand(1,nLines)-1/2)*np.pi

    def sampleLengthsUnif(self,lengthMin, lengthMax):
        """
        Sample line lengths uniformly. Updates lengthsMin, lengthsMax and pointsMat.

        :Parameters:
            lengthMax: float
                maximum realizable length
            lengthMax: float
                minimum realizable length
        """

        nLines = self.nLines
        self.lengthMax = lengthMax
        self.lengthMin = lengthMin
        self.lengthsMat = (
            (lengthMax-lengthMin)*np.random.rand(1,nLines)
            +lengthMin
        )

    def sampleContrastsUnif(self,contrastMin, contrastMax):
        """
        Sample line contrasts uniformly. Updates contrastMin and contrastMax.

        :Parameters:
            contrastMax: float
                maximum realizable contrast
            contrastMin: float
                minimum realizable contrast
        """

        nLines = self.nLines
        self.contrastMax = contrastMax
        self.contrastMin = contrastMin
        self.contrastsMat = (
            (contrastMax-contrastMin)*np.random.rand(1,nLines)
            +contrastMin
        )

    def computeStandardFormMat(self,anglesMatrix,pointsMatrix):
        """
        Returns the standard form parameters of a set of lines (ax + by + c = 0) as a nLines-by-3 matrix.

        :Parameters:
            anglesMat: nLines-by-1 numpy array
                each row contains a line's angle
            pointsMat: nLines-by-2 numpy array
                each row contains the x then y coordinate of a line's center

        :Returns:
            returnMat: nLines-by-3 numpy array
                each row contains the parameters of a standard form line, a, b then c.
        """

        returnMat = np.zeros((self.nLines,3))
        returnMat[:,0] = np.sin(anglesMatrix)
        returnMat[:,1] = -np.cos(anglesMatrix)
        returnMat[:,2] = (
            -np.multiply(returnMat[:,1],pointsMatrix[:,0])
            -np.multiply(returnMat[:,0],pointsMatrix[:,1])
        )
        return returnMat

    def distanceFunctionEachLine(self,x,y,standardFormMat):
        """
        Returns the distance from a point (x,y) to a set of lines.

        :Parameters:
            x: float
                x-coordinate of the point
            y: float
                y-coordinate of the point
            standardFormMat: nLines-by-3 array
                An array that contains the standard form parameters (ax + by + c = 0) of a set of lines, rows should be formatted as a, b then c.

        :Returns:
            np.sqrt(distSquVec): nLines-by-1 array
                Each array entry is the distance from the point to the line.
        """

        numer = (
        x*standardFormMat[:,0] 
        + y*standardFormMat[:,1] 
        + standardFormMat[:,2]
        )
        denom = (
            standardFormMat[:,0]**2 
            + standardFormMat[:,1]**2    
        )
        distSquVec = (numer**2)/denom
        return np.sqrt(distSquVec)

    def linearField2Distance(self):
        """
        Updates the attribute distanceFunction(x,y).
        """
        sfMatParallel = self.computeStandardFormMat(
            self.anglesMat,self.pointsMat
        )
        distanceVectorParallel = lambda x,y: (
        self.distanceFunctionEachLine(x,y,sfMatParallel)
        )
        sfMatPerpendicular = self.computeStandardFormMat(
            self.anglesMat+np.pi/2,self.pointsMat
        )
        distanceVectorPerpendicular = lambda x,y: (
        self.distanceFunctionEachLine(x,y,sfMatPerpendicular)
        -self.lengthsMat/2
        )
        self.distanceFunction=lambda x,y:(
            np.maximum(
                distanceVectorParallel(x,y),
                distanceVectorPerpendicular(x,y)
                )
        )

    def linearField2MollifiedIndicator(self,
            mollifierFloorWidth, mollifierSigma):
        """
        Updates the attribute mollifiedIndicator(x,y), mollifierFloorWidth and mollifierSigma.

        :Parameters:
            mollifierFloorWidth: float
                thickness of the lines
            mollifierSigma: float
                the spread of the function represented by  mollifiedIndicator
        """
        self.mollifierFloorWidth = mollifierFloorWidth
        self.mollifierSigma = mollifierSigma
        self.linearField2Distance()
        eps = lambda x, y: np.maximum( 0, 
            self.distanceFunction(x,y)-mollifierFloorWidth
            )
        self.mollifiedIndicator = lambda x, y:(
            np.min(
                1 - (1-self.contrastsMat)*np.exp(
                    -(eps(x,y)**2)/(2*mollifierSigma**2)
                )
            )
        )