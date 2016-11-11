//: Playground - noun: a place where people can play

import Foundation


// From https://github.com/thellimist/SwiftRandom/blob/master/Randoms.swift
public extension Double {
    /// SwiftRandom extension
    public static func random(lower: Double = 0, _ upper: Double = 100) -> Double {
        return (Double(arc4random()) / 0xFFFFFFFF) * (upper - lower) + lower
    }
}

public extension Int {
    /// SwiftRandom extension
    public static func random(lower: Int = 0, _ upper: Int = 100) -> Int {
        return lower + Int(arc4random_uniform(UInt32(upper - lower + 1)))
    }
}

class NeuralNetwork
{
    //private static Random rnd; JAO
    private var numInput : Int
    private var numHidden : Int
    private var numOutput : Int
    private var inputs : [Double]
    private var ihWeights : [[Double]] // input-hidden
    
    private var hBiases : [Double]
    private var hOutputs : [Double]
    private var hoWeights : [[Double]] // hidden-output
    private var oBiases : [Double]
    private var outputs : [Double]
    // back-prop specific arrays (these could be local to method UpdateWeights)
    private var oGrads : [Double] // output gradients for back-propagation
    private var hGrads : [Double] // hidden gradients for back-propagation
    // back-prop momentum specific arrays (could be local to method Train)
    private var ihPrevWeightsDelta : [[Double]]  // for momentum with back-propagation
    private var hPrevBiasesDelta : [Double]
    private var hoPrevWeightsDelta : [[Double]]
    private var oPrevBiasesDelta : [Double]

    private class func makeMatrix(rows: Int, cols: Int) -> [[Double]] // helper for ctor
    {
        let result = [[Double]](repeating: [Double](repeating: 0.0, count: cols), count: rows)
        
        return result
    }

    
    public init (numInput : Int, numHidden: Int, numOutput: Int)
    {
    //rnd = new Random(0); // for InitializeWeights() and Shuffle() JAO
    
        self.numInput = numInput
        self.numHidden = numHidden
        self.numOutput = numOutput
        self.inputs = [Double](repeating: 0.0, count: numInput)
        self.ihWeights = NeuralNetwork.makeMatrix(rows: numInput, cols: numHidden)
        self.hBiases = [Double](repeating: 0.0, count: numHidden)
        self.hOutputs = [Double](repeating: 0.0, count: numHidden)
        self.hoPrevWeightsDelta = NeuralNetwork.makeMatrix(rows: numHidden, cols: numOutput);
        self.oBiases = [Double](repeating: 0.0, count: numOutput)
        self.outputs = [Double](repeating: 0.0, count: numOutput)
        
        // back-prop related arrays below
        self.hGrads = [Double](repeating: 0.0, count: numHidden)
        self.oGrads = [Double](repeating: 0.0, count: numOutput)
        self.hoWeights = NeuralNetwork.makeMatrix(rows: numHidden, cols: numOutput)
        self.hPrevBiasesDelta = [Double](repeating: 0.0, count: numHidden)
        self.ihPrevWeightsDelta = NeuralNetwork.makeMatrix(rows: numInput, cols: numHidden);
        self.oPrevBiasesDelta = [Double](repeating: 0.0, count: numOutput)
    } // ctor

    public func toString() -> String // yikes
    {
        var s = ""
        s += "===============================\n"
        s += "numInput = \(numInput) numHidden = \(numHidden) + numOutput = \(numOutput) + \n\n"
        s += "inputs: \n"
    
        for i in 0..<inputs.count
        {
            s += "\(String(format:"%.2f", inputs[i].rounded())) "
        }
        
        s += "\n\n"
        s += "ihWeights: \n"
    
        for i in 0..<ihWeights.count
        {
            for j in 0..<ihWeights[i].count
            {
                s += "\(String(format:"%.4f",ihWeights[i][j])) "
            }
            s += "\n"
        }
        
        s += "\n"
        s += "hBiases: \n"
    
        for i in 0..<hBiases.count
        {
            s += "\(String(format:"%.4f",hBiases[i])) "
        }
        
        s += "\n\n"
        s += "hOutputs: \n"
        
        for i in 0..<hOutputs.count
        {
            s += "\(String(format:"%.4f",hOutputs[i])) "
        }
        
        s += "\n\n"
        s += "hoWeights: \n"
        
        for i in 0..<hoWeights.count
        {
            for j in 0..<hoWeights[i].count
            {
                s += "\(String(format:"%.4f", hoWeights[i][j])) "
            }
            s += "\n"
        }
        
        s += "\n"
        s += "oBiases: \n"

        for i in 0..<oBiases.count
        {
            s += "\(String(format:"%.4f", oBiases[i])) "
        }
        
        s += "\n\n"
        s += "hGrads: \n"
    
        for val in hGrads
        {
            s += "\(String(format:"%.4f",val)) "
        }
        
        s += "\n\n"
        s += "oGrads: \n"
    
        for val in oGrads
        {
            s += "\(String(format:"%.4f",val)) "
        }
        
        s += "\n\n"
        s += "ihPrevWeightsDelta: \n"
    
        for i in 0..<ihPrevWeightsDelta.count
        {
            for j in 0..<ihPrevWeightsDelta[i].count
            {
                s += "\(String(format:"%.4f",ihPrevWeightsDelta[i][j])) "
            }
            s += "\n"
        }
    
        s += "\n"
        s += "hPrevBiasesDelta: \n"
    
        for val in hPrevBiasesDelta
        {
            s += "\(String(format:"%.4f",val)) "
        }
        
        s += "\n\n"
        s += "hoPrevWeightsDelta: \n"
        
        for i in 0..<hoPrevWeightsDelta.count
        {
            for j in 0..<hoPrevWeightsDelta[i].count
            {
                s += "\(String(format:"%.4f",hoPrevWeightsDelta[i][j])) "
            }
            s += "\n"
        }
        s += "\n"
        s += "oPrevBiasesDelta: \n"
    
       
        for val in oPrevBiasesDelta
        {
            s += "\(String(format:"%.4f",val)) "
        }
        
        s += "\n\n"
        s += "outputs: \n"
    
        for val in outputs
        {
            s += "\(String(format:"%.2f",val)) "
        }
        
        s += "\n\n"
        s += "===============================\n"
        return s
    }
    
    // ----------------------------------------------------------------------------------------
    
    public func setWeights (weights: [Double])
    {
        // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
        let numWeights : Int = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput
    
        guard weights.count == numWeights else
        {
            // JAO throw new Exception("Bad weights array length: ");
            return
        }
    
        var k = 0; // points into weights param
   
        for i in 0..<numInput
        {
            for j in 0..<numHidden
            {
                ihWeights[i][j] = weights[k]
                k += 1
            }
        }
        
        for i in 0..<numHidden
        {
            hBiases[i] = weights[k]
            k += 1
        }
        
        for i in 0..<numHidden
        {
            for j in 0..<numOutput
            {
                hoWeights[i][j] = weights[k]
                k += 1
            }
        }
    
        for i in 0..<numOutput
        {
            oBiases[i] = weights[k];
            k += 1
        }
    }
    
    public func initializeWeights()
    {
        // initialize weights and biases to small random values
        let numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput
        var initialWeights = [Double](repeating: 0.0, count: numWeights)
        
        let lo = -0.01
        let hi = 0.01
    
        for i in 0..<initialWeights.count
        {
            initialWeights[i] = (hi - lo) * Double.random(lower: 0, 1) + lo
            // JAO initialWeights[i] = (hi - lo) * rnd.NextDouble() + lo;
        }
    
        setWeights(weights: initialWeights);
    }
    
    public func getWeights() -> [Double]
    {
        // returns the current set of wweights, presumably after training
        let numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput
        var result = [Double](repeating: 0.0, count: numWeights)
    
        var k = 0;
    
        for i in 0..<ihWeights.count
        {
            for j in 0..<ihWeights[0].count
            {
                result[k] = ihWeights[i][j]
                k += 1
            }
        }
        
        for i in 0..<hBiases.count
        {
            result[k] = hBiases[i];
            k += 1
        }

        for i in 0..<hoWeights.count
        {
            for j in 0..<hoWeights[0].count
            {
                result[k] = hoWeights[i][j]
                k += 1
            }
        }
        
        for i in 0..<oBiases.count
        {
            result[k] = oBiases[i]
            k += 1
        }
    
        return result;
    }
    
    // ----------------------------------------------------------------------------------------
    
    private func computeOutputs(xValues: [Double]) -> [Double]
    {
        //print ("\(xValues.count) \(numInput)") //JAO
        guard xValues.count == numInput else
        {
            //JAO throw new Exception("Bad xValues array length");
            return [Double]()
        }
        
        var hSums = [Double](repeating: 0.0, count: numHidden) // hidden nodes sums scratch array
        var oSums = [Double](repeating: 0.0, count: numOutput) // output nodes sums
        
        for i in 0..<xValues.count // copy x-values to inputs
        {
            inputs[i] = xValues[i]
        }
        
        for j in 0..<numHidden // compute i-h sum of weights * inputs
        {
            for i in 0..<numInput
            {
                hSums[j] += inputs[i] * ihWeights[i][j] // note +=
            }
        }
        
        for i in 0..<numHidden // add biases to input-to-hidden sums
        {
            hSums[i] += hBiases[i]
        }
    
        for i in 0..<numHidden // apply activation
        {
            hOutputs[i] = NeuralNetwork.hyperTanFunction(x: hSums[i]) // hard-coded
        }
    
        for j in 0..<numOutput // compute h-o sum of weights * hOutputs
        {
            for i in 0..<numHidden
            {
                oSums[j] += hOutputs[i] * hoWeights[i][j]
            }
        }
        
        for i in 0..<numOutput // add biases to input-to-hidden sum
        {
            oSums[i] += oBiases[i]
        }
        
        let softOut = NeuralNetwork.softmax(oSums: oSums) // softmax activation does all outputs at once for efficiency
        
        //Array.Copy(softOut, outputs, softOut.Length); JAO
        outputs = softOut //Arrays are value types in Swift
        
        //var retResult = [Double](repeating: 0.0, count: numOutput) // could define a GetOutputs method instead

        //Array.Copy(this.outputs, retResult, retResult.Length); JAO
        let retResult = outputs //Arrays are value types in Swift
        
        return retResult
    } // ComputeOutputs
    
    private class func hyperTanFunction(x: Double) -> Double
    {
        if x < -20.0 { return -1.0 } // approximation is correct to 30 decimals
        else if x > 20.0 { return 1.0 }
        else { return tanh(x) }
    }
    
    private class func softmax (oSums: [Double]) -> [Double]
    {
        // determine max output sum
        // does all output nodes at once so scale doesn't have to be re-computed each time
        var max = oSums[0];
    
        for i in 0..<oSums.count
        {
            if oSums[i] > max
            {
                max = oSums[i]
            }
        }
        
        // determine scaling factor -- sum of exp(each val - max)
        var scale = 0.0;
    
        for i in 0..<oSums.count
        {
            scale += exp(oSums[i] - max)
        }
    
        var result = [Double](repeating: 0.0, count: oSums.count)
    
        for i in 0..<oSums.count
        {
            result[i] = exp(oSums[i] - max) / scale
        }
        
        return result // now scaled so that xi sum to 1.0
    }
    // ----------------------------------------------------------------------------------------
    private func updateWeights (tValues: [Double], learnRate: Double, momentum: Double, weightDecay: Double)
    {
        // update the weights and biases using back-propagation, with target values, eta (learning rate),
        // alpha (momentum).
        // assumes that SetWeights and ComputeOutputs have been called and so all the internal arrays
        // and matrices have values (other than 0.0)
    
        guard tValues.count == numOutput else
        {
            return // JAO throw new Exception("target values not same Length as output in UpdateWeights");
        }
    
        // 1. compute output gradients
        for i in 0..<oGrads.count
        {
            // derivative of softmax = (1 - y) * y (same as log-sigmoid)
            let derivative = (1 - outputs[i]) * outputs[i]
            // 'mean squared error version' includes (1-y)(y) derivative
            oGrads[i] = derivative * (tValues[i] - outputs[i])
        }
    
        // 2. compute hidden gradients
        for i in 0..<hGrads.count
        {
            // derivative of tanh = (1 - y) * (1 + y)
            let derivative = (1 - hOutputs[i]) * (1 + hOutputs[i]);
            var sum = 0.0;

            for j in 0..<numOutput // each hidden delta is the sum of numOutput terms
            {
                let x = oGrads[j] * hoWeights[i][j]
                sum += x
            }
            hGrads[i] = derivative * sum;
        }
    
        // 3a. update hidden weights (gradients must be computed right-to-left but weights
        // can be updated in any order)
        for i in 0..<ihWeights.count // 0..2 (3)
        {
            for j in 0..<ihWeights[0].count // 0..3 (4)
            {
                let delta = learnRate * hGrads[j] * inputs[i] // compute the new delta
                ihWeights[i][j] += delta // update. note we use '+' instead of '-'. this can be very tricky.
                // now add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
                ihWeights[i][j] += momentum * ihPrevWeightsDelta[i][j]
                ihWeights[i][j] -= (weightDecay * ihWeights[i][j]) // weight decay
                ihPrevWeightsDelta[i][j] = delta // don't forget to save the delta for momentum
            }
        }
    
        // 3b. update hidden biases
        for i in 0..<hBiases.count
        {
            let delta = learnRate * hGrads[i] * 1.0 // t1.0 is constant input for bias; could leave out
            hBiases[i] += delta
            hBiases[i] += momentum * hPrevBiasesDelta[i] // momentum
            hBiases[i] -= (weightDecay * hBiases[i]) // weight decay
            hPrevBiasesDelta[i] = delta // don't forget to save the delta
        }
    
        // 4. update hidden-output weights
        for i in 0..<hoWeights.count
        {
            for j in 0..<hoWeights[0].count
            {
                // see above: hOutputs are inputs to the nn outputs
                let delta = learnRate * oGrads[j] * hOutputs[i]
                hoWeights[i][j] += delta
                hoWeights[i][j] += momentum * hoPrevWeightsDelta[i][j] // momentum
                hoWeights[i][j] -= (weightDecay * hoWeights[i][j]) // weight decay
                hoPrevWeightsDelta[i][j] = delta // save
            }
        }
        
        // 4b. update output biases
        for i in 0..<oBiases.count
        {
            let delta = learnRate * oGrads[i] * 1.0;
            oBiases[i] += delta;
            oBiases[i] += momentum * oPrevBiasesDelta[i]; // momentum
            oBiases[i] -= (weightDecay * oBiases[i]); // weight decay
            oPrevBiasesDelta[i] = delta; // save
        }
    } // UpdateWeights
    // ----------------------------------------------------------------------------------------
    public func train(trainData: [[Double]], maxEprochs: Int, learnRate: Double, momentum: Double, weightDecay: Double)
    {
        // train a back-prop style NN classifier using learning rate and momentum
        // weight decay reduces the magnitude of a weight value over time unless that value
        // is constantly increased
        var epoch = 0;
    
        var xValues = [Double](repeating: 0.0, count: numInput) // inputs
        var tValues = [Double](repeating: 0.0, count: numOutput)// target values
        var sequence = [Int](repeating: 0, count: trainData.count)
        
        for i in 0..<sequence.count
        {
            sequence[i] = i
        }
    
        while epoch < maxEprochs
        {
            let mse = meanSquaredError(trainData: trainData)

            if mse < 0.020 { print("break"); break } // consider passing value in as parameter
            //if (mse < 0.001) break; // consider passing value in as parameter
            NeuralNetwork.shuffle(sequence: &sequence) // visit each training data in random order

            for i in 0..<trainData.count
            {
                let idx = sequence[i];
                //Array.Copy(trainData[idx], xValues, numInput);
                xValues = [Double](trainData[idx][0..<numInput])
                
                //Array.Copy(trainData[idx], numInput, tValues, 0, numOutput); //JAO
                //tValues.replaceSubrange(numInput..<numInput+numOutput, with: trainData[idx])
                tValues = [Double](trainData[idx][numInput..<numInput+numOutput])
                
                computeOutputs(xValues: xValues) // copy xValues in, compute outputs (store them internally)
                updateWeights(tValues: tValues, learnRate: learnRate, momentum: momentum, weightDecay: weightDecay) // find better weights
            } // each training tuple
            epoch += 1
        }
    } // Train
    
    private class func shuffle(sequence: inout [Int])
    {
        for i in 0..<sequence.count
        {
            //var r = rnd.Next(i, sequence.Length); //JAO
            let r = Int.random(lower: i, sequence.count-1)
            let tmp = sequence[r];
            sequence[r] = sequence[i];
            sequence[i] = tmp;
        }
    }
    
    private func meanSquaredError(trainData: [[Double]]) -> Double // used as a training stopping condition
    {
        // average squared error per training tuple
        var sumSquaredError = 0.0;
        var xValues = [Double]() // first numInput values in trainData
        var tValues = [Double]() // last numOutput values
        
        // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
        for i in 0..<trainData.count
        {
            xValues = [Double](trainData[i][0..<numInput])
            tValues = [Double](trainData[i][numInput..<numInput+numOutput])
            
            var yValues = computeOutputs(xValues: xValues)// compute output using current weights
            
            for j in 0..<numOutput
            {
                let err = tValues[j] - yValues[j]
                sumSquaredError += err * err
            }
        }
        return sumSquaredError / Double(trainData.count)
    }
    // ----------------------------------------------------------------------------------------
    public func accuracy (testData: [[Double]]) -> Double
    {
        // percentage correct using winner-takes all
        var numCorrect = 0
        var numWrong = 0
        var xValues = [Double]() // inputs
        var tValues = [Double]() // targets
        var yValues : [Double] = [Double]() // computed Y
        
        for i in 0..<testData.count
        {
            //Array.Copy(testData[i], xValues, numInput); JAO // parse test data into x-values and t-values
            xValues = [Double](testData[i][0..<numInput])
            
            //Array.Copy(testData[i], numInput, tValues, 0, numOutput); JAO
            //tValues.replaceSubrange(numInput..<numInput+numOutput, with: testData[i])
            tValues = [Double](testData[i][numInput..<numInput+numOutput])
            
            yValues = computeOutputs(xValues: xValues)
            let maxIndex = NeuralNetwork.maxIndex(vector: yValues) // which cell in yValues has largest value?
            
            if tValues[maxIndex] == 1.0 // ugly. consider AreEqual(double x, double y)
            {
                numCorrect += 1
                //print ("correct: \(numCorrect)")
            }
            else
            {
                numWrong += 1
                //print("wrong: \(numWrong)")
            }
        }
        return (Double(numCorrect) * 1.0) / (Double(numCorrect) + Double(numWrong)) // ugly 2 - check for divide by zero
    }
    
    private class func maxIndex(vector: [Double]) -> Int // helper for Accuracy()
    {
        // index of largest value
        var bigIndex = 0
        var biggestVal = vector[0]
    
        for i in 0..<vector.count
        {
            if vector[i] > biggestVal
            {
                biggestVal = vector[i]
                bigIndex = i
            }
        }
        return bigIndex
    }
} // NeuralNetwork

func makeTrainTest(allData: [[Double]], trainData: inout [[Double]], testData: inout [[Double]])
{
    // split allData into 80% trainData and 20% testData
    //Random rnd = new Random(0); JAO
    
    // Total number of rows/cases in training set
    let totRows = allData.count
    
    // Total number of inputs & outputs for each training case
    let numCols = allData[0].count
    
    //Rows used for training
    let trainRows = Int(Double(totRows) * 0.80) // hard-coded 80-20 split
    
    // Rows used for testing
    let testRows = totRows - trainRows

    trainData = [[Double]](repeating: [Double](), count: trainRows)
    testData = [[Double]](repeating: [Double](), count: testRows)
    
    var sequence = [Int](repeating: 0, count: totRows) // create a random sequence of indexes
    
    for i in 0..<sequence.count
    {
        sequence[i] = i
    }
    
    for i in 0..<sequence.count
    {
        let r = Int.random(lower: i, sequence.count-1) // JAO rnd.Next(i, sequence.Length);
        let tmp = sequence[r]
        sequence[r] = sequence[i]
        sequence[i] = tmp
    }
    
    var si = 0 // index into sequence[]
    var j = 0 // index into trainData or testData
    
    while si < trainRows // first rows to train data
    {
        trainData[j] = [Double](repeating: 0.0, count: numCols)
        let idx = sequence[si];
        // JAO Array.Copy(allData[idx], trainData[j], numCols);
        trainData[j] = allData[idx]
        j += 1
        si += 1
    }
    
    j = 0 // reset to start of test data
    
    while si < totRows // remainder to test data
    {
        testData[j] = [Double](repeating: 0.0, count: numCols)
        let idx = sequence[si]
        // JAO Array.Copy(allData[idx], testData[j], numCols);
        testData[j] = allData[idx]
        j += 1
        si += 1
    }

} // MakeTrainTest

func normalize( dataMatrix: inout [[Double]], cols: [Int])
{
    // normalize specified cols by computing (x - mean) / sd for each value

    for col in cols
    {
        var sum = 0.0
       
        for i in 0..<dataMatrix.count
        {
            sum += dataMatrix[i][col]
        }
        
        let mean = sum / Double(dataMatrix.count)
        sum = 0.0
        
        for i in 0..<dataMatrix.count
        {
            sum += (dataMatrix[i][col] - mean) * (dataMatrix[i][col] - mean)
        }
        
        // thanks to Dr. W. Winfrey, Concord Univ., for catching bug in original code
        let sd = sqrt(sum / Double((dataMatrix.count - 1)))
        
        for i in 0..<dataMatrix.count
        {
            dataMatrix[i][col] = (dataMatrix[i][col] - mean) / sd
        }
    }
}

func showVector(vector: [Double], valsPerRow: Int, decimals: Int, newLine: Bool)
{
    for i in 0..<vector.count
    {
        if i % valsPerRow == 0 { print("") }
        //print(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
        print("\(String(format:"%.\(decimals)f",vector[i])) ", separator: "", terminator: "")
    }
    if newLine == true { print("") }
}

func showMatrix(matrix: [[Double]], numRows: Int, decimals: Int, newLine: Bool)
{
    for i in 0..<numRows
    {
        print ("   \(i): ", separator: "", terminator: "")

        for j in 0..<matrix[i].count
        {
            if matrix[i][j] >= 0.0
            {
                print(" ", separator: "", terminator: "")
            }
            else
            {
                print("-", separator: "", terminator: "")
            }
            print("\(String(format:"%.\(decimals)f",abs(matrix[i][j]))) ", separator: "", terminator: "")
        }
        print("")
    }
    
    if newLine == true
    {
        print("")
    }
}

func main()
{
    print("\nBegin Build 2014 neural network demo")
    print("\nData is the famous Iris flower set.");
    print("Data is sepal length, sepal width, petal length, petal width -> iris species");
    print("Iris setosa = 0 0 1, Iris versicolor = 0 1 0, Iris virginica = 1 0 0 ");
    print("The goal is to predict species from sepal length, width, petal length, width\n");
    print("Raw data resembles:\n");
    print(" 5.1, 3.5, 1.4, 0.2, Iris setosa");
    print(" 7.0, 3.2, 4.7, 1.4, Iris versicolor");
    print(" 6.3, 3.3, 6.0, 2.5, Iris virginica");
    print(" ......\n");
    
    let allData : [[Double]] = [
        [5.1, 3.5, 1.4, 0.2, 0, 0, 1], // sepal length, width, petal length, width
        [4.9, 3.0, 1.4, 0.2, 0, 0, 1], // Iris setosa = 0 0 1
        [4.7, 3.2, 1.3, 0.2, 0, 0, 1], // Iris versicolor = 0 1 0
        [4.6, 3.1, 1.5, 0.2, 0, 0, 1], // Iris virginica = 1 0 0
        [5.0, 3.6, 1.4, 0.2, 0, 0, 1],
        [5.4, 3.9, 1.7, 0.4, 0, 0, 1],
        [4.6, 3.4, 1.4, 0.3, 0, 0, 1],
        [5.0, 3.4, 1.5, 0.2, 0, 0, 1],
        [4.4, 2.9, 1.4, 0.2, 0, 0, 1],
        [4.9, 3.1, 1.5, 0.1, 0, 0, 1],
        [5.4, 3.7, 1.5, 0.2, 0, 0, 1],
        [4.8, 3.4, 1.6, 0.2, 0, 0, 1],
        [4.8, 3.0, 1.4, 0.1, 0, 0, 1],
        [4.3, 3.0, 1.1, 0.1, 0, 0, 1],
        [5.8, 4.0, 1.2, 0.2, 0, 0, 1],
        [5.7, 4.4, 1.5, 0.4, 0, 0, 1],
        [5.4, 3.9, 1.3, 0.4, 0, 0, 1],
        [5.1, 3.5, 1.4, 0.3, 0, 0, 1],
        [5.7, 3.8, 1.7, 0.3, 0, 0, 1],
        [5.1, 3.8, 1.5, 0.3, 0, 0, 1],
        [5.4, 3.4, 1.7, 0.2, 0, 0, 1],
        [5.1, 3.7, 1.5, 0.4, 0, 0, 1],
        [4.6, 3.6, 1.0, 0.2, 0, 0, 1],
        [5.1, 3.3, 1.7, 0.5, 0, 0, 1],
        [4.8, 3.4, 1.9, 0.2, 0, 0, 1],
        [5.0, 3.0, 1.6, 0.2, 0, 0, 1],
        [5.0, 3.4, 1.6, 0.4, 0, 0, 1],
        [5.2, 3.5, 1.5, 0.2, 0, 0, 1],
        [5.2, 3.4, 1.4, 0.2, 0, 0, 1],
        [4.7, 3.2, 1.6, 0.2, 0, 0, 1],
        [4.8, 3.1, 1.6, 0.2, 0, 0, 1],
        [5.4, 3.4, 1.5, 0.4, 0, 0, 1],
        [5.2, 4.1, 1.5, 0.1, 0, 0, 1],
        [5.5, 4.2, 1.4, 0.2, 0, 0, 1],
        [4.9, 3.1, 1.5, 0.1, 0, 0, 1],
        [5.0, 3.2, 1.2, 0.2, 0, 0, 1],
        [5.5, 3.5, 1.3, 0.2, 0, 0, 1],
        [4.9, 3.1, 1.5, 0.1, 0, 0, 1],
        [4.4, 3.0, 1.3, 0.2, 0, 0, 1],
        [5.1, 3.4, 1.5, 0.2, 0, 0, 1],
        [5.0, 3.5, 1.3, 0.3, 0, 0, 1],
        [4.5, 2.3, 1.3, 0.3, 0, 0, 1],
        [4.4, 3.2, 1.3, 0.2, 0, 0, 1],
        [5.0, 3.5, 1.6, 0.6, 0, 0, 1],
        [5.1, 3.8, 1.9, 0.4, 0, 0, 1],
        [4.8, 3.0, 1.4, 0.3, 0, 0, 1],
        [5.1, 3.8, 1.6, 0.2, 0, 0, 1],
        [4.6, 3.2, 1.4, 0.2, 0, 0, 1],
        [5.3, 3.7, 1.5, 0.2, 0, 0, 1],
        [5.0, 3.3, 1.4, 0.2, 0, 0, 1],
        [7.0, 3.2, 4.7, 1.4, 0, 1, 0],
        [6.4, 3.2, 4.5, 1.5, 0, 1, 0],
        [6.9, 3.1, 4.9, 1.5, 0, 1, 0],
        [5.5, 2.3, 4.0, 1.3, 0, 1, 0],
        [6.5, 2.8, 4.6, 1.5, 0, 1, 0],
        [5.7, 2.8, 4.5, 1.3, 0, 1, 0],
        [6.3, 3.3, 4.7, 1.6, 0, 1, 0],
        [4.9, 2.4, 3.3, 1.0, 0, 1, 0],
        [6.6, 2.9, 4.6, 1.3, 0, 1, 0],
        [5.2, 2.7, 3.9, 1.4, 0, 1, 0],
        [5.0, 2.0, 3.5, 1.0, 0, 1, 0],
        [5.9, 3.0, 4.2, 1.5, 0, 1, 0],
        [6.0, 2.2, 4.0, 1.0, 0, 1, 0],
        [6.1, 2.9, 4.7, 1.4, 0, 1, 0],
        [5.6, 2.9, 3.6, 1.3, 0, 1, 0],
        [6.7, 3.1, 4.4, 1.4, 0, 1, 0],
        [5.6, 3.0, 4.5, 1.5, 0, 1, 0],
        [5.8, 2.7, 4.1, 1.0, 0, 1, 0],
        [6.2, 2.2, 4.5, 1.5, 0, 1, 0],
        [5.6, 2.5, 3.9, 1.1, 0, 1, 0],
        [5.9, 3.2, 4.8, 1.8, 0, 1, 0],
        [6.1, 2.8, 4.0, 1.3, 0, 1, 0],
        [6.3, 2.5, 4.9, 1.5, 0, 1, 0],
        [6.1, 2.8, 4.7, 1.2, 0, 1, 0],
        [6.4, 2.9, 4.3, 1.3, 0, 1, 0],
        [6.6, 3.0, 4.4, 1.4, 0, 1, 0],
        [6.8, 2.8, 4.8, 1.4, 0, 1, 0],
        [6.7, 3.0, 5.0, 1.7, 0, 1, 0],
        [6.0, 2.9, 4.5, 1.5, 0, 1, 0],
        [5.7, 2.6, 3.5, 1.0, 0, 1, 0],
        [5.5, 2.4, 3.8, 1.1, 0, 1, 0],
        [5.5, 2.4, 3.7, 1.0, 0, 1, 0],
        [5.8, 2.7, 3.9, 1.2, 0, 1, 0],
        [6.0, 2.7, 5.1, 1.6, 0, 1, 0],
        [5.4, 3.0, 4.5, 1.5, 0, 1, 0],
        [6.0, 3.4, 4.5, 1.6, 0, 1, 0],
        [6.7, 3.1, 4.7, 1.5, 0, 1, 0],
        [6.3, 2.3, 4.4, 1.3, 0, 1, 0],
        [5.6, 3.0, 4.1, 1.3, 0, 1, 0],
        [5.5, 2.5, 4.0, 1.3, 0, 1, 0],
        [5.5, 2.6, 4.4, 1.2, 0, 1, 0],
        [6.1, 3.0, 4.6, 1.4, 0, 1, 0],
        [5.8, 2.6, 4.0, 1.2, 0, 1, 0],
        [5.0, 2.3, 3.3, 1.0, 0, 1, 0],
        [5.6, 2.7, 4.2, 1.3, 0, 1, 0],
        [5.7, 3.0, 4.2, 1.2, 0, 1, 0],
        [5.7, 2.9, 4.2, 1.3, 0, 1, 0],
        [6.2, 2.9, 4.3, 1.3, 0, 1, 0],
        [5.1, 2.5, 3.0, 1.1, 0, 1, 0],
        [5.7, 2.8, 4.1, 1.3, 0, 1, 0],
        [6.3, 3.3, 6.0, 2.5, 1, 0, 0],
        [5.8, 2.7, 5.1, 1.9, 1, 0, 0],
        [7.1, 3.0, 5.9, 2.1, 1, 0, 0],
        [6.3, 2.9, 5.6, 1.8, 1, 0, 0],
        [6.5, 3.0, 5.8, 2.2, 1, 0, 0],
        [7.6, 3.0, 6.6, 2.1, 1, 0, 0],
        [4.9, 2.5, 4.5, 1.7, 1, 0, 0],
        [7.3, 2.9, 6.3, 1.8, 1, 0, 0],
        [6.7, 2.5, 5.8, 1.8, 1, 0, 0],
        [7.2, 3.6, 6.1, 2.5, 1, 0, 0],
        [6.5, 3.2, 5.1, 2.0, 1, 0, 0],
        [6.4, 2.7, 5.3, 1.9, 1, 0, 0],
        [6.8, 3.0, 5.5, 2.1, 1, 0, 0],
        [5.7, 2.5, 5.0, 2.0, 1, 0, 0],
        [5.8, 2.8, 5.1, 2.4, 1, 0, 0],
        [6.4, 3.2, 5.3, 2.3, 1, 0, 0],
        [6.5, 3.0, 5.5, 1.8, 1, 0, 0],
        [7.7, 3.8, 6.7, 2.2, 1, 0, 0],
        [7.7, 2.6, 6.9, 2.3, 1, 0, 0],
        [6.0, 2.2, 5.0, 1.5, 1, 0, 0],
        [6.9, 3.2, 5.7, 2.3, 1, 0, 0],
        [5.6, 2.8, 4.9, 2.0, 1, 0, 0],
        [7.7, 2.8, 6.7, 2.0, 1, 0, 0],
        [6.3, 2.7, 4.9, 1.8, 1, 0, 0],
        [6.7, 3.3, 5.7, 2.1, 1, 0, 0],
        [7.2, 3.2, 6.0, 1.8, 1, 0, 0],
        [6.2, 2.8, 4.8, 1.8, 1, 0, 0],
        [6.1, 3.0, 4.9, 1.8, 1, 0, 0],
        [6.4, 2.8, 5.6, 2.1, 1, 0, 0],
        [7.2, 3.0, 5.8, 1.6, 1, 0, 0],
        [7.4, 2.8, 6.1, 1.9, 1, 0, 0],
        [7.9, 3.8, 6.4, 2.0, 1, 0, 0],
        [6.4, 2.8, 5.6, 2.2, 1, 0, 0],
        [6.3, 2.8, 5.1, 1.5, 1, 0, 0],
        [6.1, 2.6, 5.6, 1.4, 1, 0, 0],
        [7.7, 3.0, 6.1, 2.3, 1, 0, 0],
        [6.3, 3.4, 5.6, 2.4, 1, 0, 0],
        [6.4, 3.1, 5.5, 1.8, 1, 0, 0],
        [6.0, 3.0, 4.8, 1.8, 1, 0, 0],
        [6.9, 3.1, 5.4, 2.1, 1, 0, 0],
        [6.7, 3.1, 5.6, 2.4, 1, 0, 0],
        [6.9, 3.1, 5.1, 2.3, 1, 0, 0],
        [5.8, 2.7, 5.1, 1.9, 1, 0, 0],
        [6.8, 3.2, 5.9, 2.3, 1, 0, 0],
        [6.7, 3.3, 5.7, 2.5, 1, 0, 0],
        [6.7, 3.0, 5.2, 2.3, 1, 0, 0],
        [6.3, 2.5, 5.0, 1.9, 1, 0, 0],
        [6.5, 3.0, 5.2, 2.0, 1, 0, 0],
        [6.2, 3.4, 5.4, 2.3, 1, 0, 0],
        [5.9, 3.0, 5.1, 1.8, 1, 0, 0]]
    
    print("\nFirst 6 rows of entire 150-item data set:")
    showMatrix(matrix: allData, numRows: 6, decimals: 1, newLine: true)
    
    print("Creating 80% training and 20% test data matrices")
    var trainData : [[Double]] = [[Double]]()
    var testData : [[Double]] = [[Double]]()
    makeTrainTest(allData: allData, trainData: &trainData, testData: &testData)

    print("\nFirst 5 rows of training data:")
    showMatrix(matrix: trainData, numRows: 5, decimals: 1, newLine: true)
    
    print("First 3 rows of test data:")
    showMatrix(matrix: testData, numRows: 3, decimals: 1, newLine: true)
    
    normalize(dataMatrix: &trainData, cols: [0, 1, 2, 3])
    normalize(dataMatrix: &testData, cols: [0, 1, 2, 3])
    
    print("\nFirst 5 rows of normalized training data:")
    showMatrix(matrix: trainData, numRows: 5, decimals: 1, newLine: true)
    
    print("First 3 rows of normalized test data:")
    showMatrix(matrix: testData, numRows: 3, decimals: 1, newLine: true)
    
    print("\nCreating a 4-input, 7-hidden, 3-output neural network")
    print("Hard-coded tanh function for input-to-hidden and softmax for ")
    print("hidden-to-output activations");
    let numInput = 4;
    let numHidden = 7;
    let numOutput = 3;
    let nn = NeuralNetwork(numInput: numInput, numHidden: numHidden, numOutput: numOutput)
    
    print("\nInitializing weights and bias to small random values")
    nn.initializeWeights()
    
    let maxEpochs = 2000
    let learnRate = 0.05
    let momentum = 0.01
    let weightDecay = 0.0001
    print("Setting maxEpochs = 2000, learnRate = 0.05, momentum = 0.01, weightDecay = 0.0001");
    print("Training has hard-coded mean squared error < 0.020 stopping condition")
    print("\nBeginning training using incremental back-propagation\n")
    nn.train(trainData: trainData, maxEprochs: maxEpochs, learnRate: learnRate, momentum: momentum, weightDecay: weightDecay)
    print("Training complete")
    
    let weights = nn.getWeights()
    print("Final neural network weights and bias values:")
    showVector(vector: weights, valsPerRow: 10, decimals: 3, newLine: true)
    let trainAcc = nn.accuracy(testData: trainData)
    print("\nAccuracy on training data = " + String(format:"%.4f",trainAcc))
    let testAcc = nn.accuracy(testData: testData);
    print("\nAccuracy on test data = " + String(format:"%.4f",testAcc))
    print("\nEnd Build 2013 neural network demo\n")
    //Console.ReadLine(); JAO
} // Main

main()

