import SwiftUI
import MetalPerformanceShadersGraph

/**
 An implementation of the Makemore MLP from
 Andrej Karpathy's video: https://www.youtube.com/watch?v=PaCmpygFfXo&t=776s
 */
class NamesMLP: ObservableObject {
    
    @Published var initialized = false
    @Published var generatedWords : [(String, Float)] = []
    @Published var lossValue: Float = 0.0
    @Published var iterations = 0
    @Published var trainingTime: Duration = .seconds(0)
    
    let device = MTLCreateSystemDefaultDevice()!
    
    var chars: [String] = []
    var X: [[UInt8]] = []
    var Y: [UInt8] = []
    var vocabSize: Int = 0
    var blockSize = 3
    var batchSize = 32

    var graph: MPSGraph?
    
    // Tensors
    var inputsPlaceholderTensor, 
        targetsPlaceholderTensor,
        embeddingWeights,
        hiddenLayerWeights,
        hiddenLayerBiases,
        logitsLayerWeights,
        logitsLayerBiases,
        softMax,
        loss: MPSGraphTensor?
    
    var backpropOps: [MPSGraphOperation] = []
    
    func initialize() async {
        
        initialized = false
        
        print("--- Initializing ---")
        
        // --- BUILD TRAINING SET ---
        
        // Load names file as a string
        let content = try! String(contentsOf: Bundle.main.url(forResource: "names", withExtension: ".txt")!)
        
        // Split file into lines (i.e. words)
        let words = content.components(separatedBy: .newlines)
        
        print("Words count: " + String(words.count))
        
        // Join all the words back together, convert each char to a string, add to set to find out which chars are in the file, sort, add a "." at the start which will be used as a token to represent the start and end of words
        chars = ["."] + Array(Set(words.joined().map {String($0)})).sorted()
        
        print("Chars set count: " + String(chars.count))
        
        // String to Index lookup
        let stoi: [String:Int] = chars.enumerated().reduce(into: [:]) { (result, element) in
            let (i, s) = element
            result[s] = i
        }
        
        // Build the data set
        vocabSize = chars.count
        
        X = []
        Y = []
        
        for w in words {
            //print(w)
            var context = Array(repeating: UInt8(0), count: blockSize)
            for ch in w + "." {
                let ix = stoi[String(ch), default: 0]
                X.append(context)
                Y.append(UInt8(ix))
                //print(context.map { chars[Int($0)] }.joined() + "--->" + chars[ix])
                context = context.dropFirst() + [UInt8(ix)]
            }
        }
        
        print("Training samples: " + String(X.count))

        // --- BUILD NN ---
        
        let embeddingSize = 2
        
        graph = MPSGraph()
        
        guard let graph else {
            print("Couldn't create MPSGraph")
            return
        }

        // --- Create placeholders for inputs ---
        
        inputsPlaceholderTensor = graph.placeholder(
            shape: [-1 as NSNumber, // batchSize
                     blockSize as NSNumber], 
            dataType: .uInt8, name: "Input Indices")
        
        targetsPlaceholderTensor = graph.placeholder(
            shape: [-1 as NSNumber], 
            dataType: .uInt8, name: "Target Indices")

        // --- Embedding Layer --- aka: E = C[X]
        
        // Convert to one-hot
        
        // [batchSize, blockSize, vocabSize]
        let inputsOneHot = graph.oneHot(
            withIndicesTensor: inputsPlaceholderTensor!, 
            depth: vocabSize, 
            name: "Input OneHot Encoded")
        
        let embeddingWeightsValues = (0..<(vocabSize * embeddingSize))
            .map { _ in Float.random(in: -0.2..<0.2) }

        embeddingWeights = graph.variable(with: Data(bytes: embeddingWeightsValues, 
                                                         count: embeddingWeightsValues.count * MemoryLayout<Float>.size), 
                                              shape: [vocabSize as NSNumber, 
                                                      embeddingSize as NSNumber], 
                                              dataType: .float32, 
                                              name: "Embedding Weights")

        let tokenEmbeddings = graph.matrixMultiplication(
            primary: inputsOneHot, 
            secondary: embeddingWeights!, 
            name: "Logits")
        
        // Reshape so that a batch consists of 1D rows which is a block of embeddings
        let embeddingsLayerOutput = graph.reshape(tokenEmbeddings, 
                                                    shape: [-1 as NSNumber, 
                                                             blockSize * embeddingSize as NSNumber], 
                                                    name: "Tokens reshaped")
        
        print("embeddingsLayerOutput.shape: \(embeddingsLayerOutput.shape!)")
        
        // --- Hidden Layer --- aka: H = tanh(E * W1 + B1)
        
        let hiddenLayerInputWidth = blockSize * embeddingSize
        let hiddenLayerOutputWidth = 100
        
        let hiddenLayerWeightsValues = (0..<(hiddenLayerInputWidth * hiddenLayerOutputWidth))
            .map { _ in Float.random(in: -0.2..<0.2) }
        
        hiddenLayerWeights = graph.variable(with: Data(bytes: hiddenLayerWeightsValues, 
                                                           count: hiddenLayerWeightsValues.count * MemoryLayout<Float>.size), 
                                              shape: [hiddenLayerInputWidth as NSNumber, 
                                                      hiddenLayerOutputWidth as NSNumber], 
                                              dataType: .float32, 
                                              name: "Hidden Layer Weights")

        let hiddenLayerBiasValues = (0..<hiddenLayerOutputWidth)
            .map { _ in Float.random(in: -0.2..<0.2) }
        
        hiddenLayerBiases = graph.variable(with: Data(bytes: hiddenLayerBiasValues, 
                                                           count: hiddenLayerBiasValues.count * MemoryLayout<Float>.size), 
                                                shape: [hiddenLayerOutputWidth as NSNumber], 
                                                dataType: .float32, 
                                                name: "Hidden Layer Biases")

        let hiddenLayerMatMul = graph.matrixMultiplication(primary: embeddingsLayerOutput, 
                                                           secondary: hiddenLayerWeights!, 
                                                           name: "Hidden Layer Weights MatMul")
        print("hiddenLayerMatMul.shape: \(hiddenLayerMatMul.shape!)")

        let hiddenLayerAddBiases = graph.addition(hiddenLayerMatMul, 
                                                  hiddenLayerBiases!, 
                                                  name: "Hidden Layer Add Biases")
        print("hiddenLayerAddBiases.shape: \(hiddenLayerAddBiases.shape!)")

        let hiddenLayerOutput = graph.tanh(with: hiddenLayerAddBiases, 
                                           name: "Hidden Layer Activation")
        print("hiddenLayerOutput.shape: \(hiddenLayerOutput.shape!)")
        
        // --- Logits Layer --- aka: L = H * W2 + B2
        
        let logitsLayerInputWidth = hiddenLayerOutputWidth
        let logitsLayerOutputWidth = vocabSize
        
        let logitsLayerWeightsValues = (0..<(logitsLayerInputWidth * logitsLayerOutputWidth))
            .map { _ in Float.random(in: -0.2..<0.2) }
        
        logitsLayerWeights = graph.variable(with: Data(bytes: logitsLayerWeightsValues, 
                                                           count: logitsLayerWeightsValues.count * MemoryLayout<Float>.size), 
                                                shape: [logitsLayerInputWidth as NSNumber, 
                                                        logitsLayerOutputWidth as NSNumber], 
                                                dataType: .float32, 
                                                name: "Logits Layer Weights")
        
        let logitsLayerBiasValues = (0..<logitsLayerOutputWidth)
            .map { _ in Float.random(in: -0.2..<0.2) }
        
        logitsLayerBiases = graph.variable(with: Data(bytes: logitsLayerBiasValues, 
                                                          count: logitsLayerBiasValues.count * MemoryLayout<Float>.size), 
                                               shape: [logitsLayerOutputWidth as NSNumber], 
                                               dataType: .float32, 
                                               name: "Logits Layer Biases")
        
        let logitsLayerMatMul = graph.matrixMultiplication(primary: hiddenLayerOutput, 
                                                           secondary: logitsLayerWeights!, 
                                                           name: "Logits Layer Weights MatMul")
        print("logitsLayerMatMul.shape: \(logitsLayerMatMul.shape!)")

        let logitsLayerOutput = graph.addition(logitsLayerMatMul, 
                                                  logitsLayerBiases!, 
                                                  name: "Logits Layer Add Biases")
        print("logitsLayerOutput.shape: \(logitsLayerOutput.shape!)")

        softMax = graph.softMax(with: logitsLayerOutput, axis: -1, name: "SoftMax")
        print("softMax.shape: \(softMax!.shape!)")
        
        // --- Loss --- aka: softMaxCrossEntropy
        
        // [batchSize, blockSize, vocabSize]
        let targetsOneHot = graph.oneHot(
            withIndicesTensor: targetsPlaceholderTensor!, 
            depth: vocabSize, 
            name: "Targets OneHot Encoded")
        
        print("targetsOneHot.shape: \(targetsOneHot.shape!)")
        
        loss = graph.softMaxCrossEntropy(logitsLayerOutput, labels: targetsOneHot, axis: -1, reuctionType: .mean, name: "SoftMax Cross Entropy")
        print("loss.shape: \(loss!.shape!)")

        // --- Backprop --- aka: SCG
        
        let gradients = graph.gradients(of: loss!, with: [embeddingWeights!, 
                                                         hiddenLayerWeights!, 
                                                         hiddenLayerBiases!, 
                                                         logitsLayerWeights!, 
                                                         logitsLayerBiases!], name: "Gradients")

        let learningRate = graph.constant(0.05, dataType: .float32)
        
        backpropOps = []
        
        for (tensor, gradient) in gradients {
            let newTensor = graph.stochasticGradientDescent(
                learningRate: learningRate, 
                values: tensor, 
                gradient: gradient, 
                name: "SGD")
            
            backpropOps += [graph.assign(tensor, tensor: newTensor, name: "Assign")]
        }
        
        initialized = true
    }
    
    func train() async {
        
        print("--- Training ---")
    
        guard let graph else {
            print("MPSGraph not initialized")
            return
        }
        
        // --- TRAIN ---
        
        iterations = 0
        
        let clock = ContinuousClock()
        
        trainingTime = clock.measure {
            for epoch in 1 ... 5000 {

                // This is necessary so graph.run() doesn't crash
                autoreleasepool {
                    // Randomly select batchSize training pairs
                    let ix = (0 ..< batchSize).map { _ in Int.random(in: 0 ..< X.count) }
                    
                    // Extract the training pairs into arrays
                    let inputsArray = Array(ix.map { X[$0] }.joined())
                    let targetsArray = ix.map { Y[$0] }
                    
                    let inputsTensorData = arrayToMPSTensorData(array: inputsArray, 
                                                                dataType: .uInt8, 
                                                                shape: [batchSize as NSNumber, 
                                                                        blockSize as NSNumber])
                    
                    let targetsTensorData = arrayToMPSTensorData(array: targetsArray, 
                                                                 dataType: .uInt8, 
                                                                 shape: [batchSize as NSNumber])
                    
                    let results = graph.run(feeds: [inputsPlaceholderTensor!: inputsTensorData, 
                                                    targetsPlaceholderTensor!: targetsTensorData], 
                                            targetTensors: [loss!], 
                                            targetOperations: backpropOps)
                    
                    // Get the loss results
                    let lossData = results[loss!]
                    
                    var lossFloat: Float = 0
                    
                    lossData?.mpsndarray().readBytes(&lossFloat, strideBytes: nil)
                    
                    
                    if epoch % 100 == 0 {
                        print("\(epoch). loss = \(lossValue)")
                        
                        iterations = epoch
                        lossValue = lossFloat

                    }
                }
            }
        }
        
        print("Training took: \(trainingTime)")
    }
    
    func generate() async {
        print("--- Generating ---")
        
        guard let graph else {
            print("MPSGraph not created")
            return 
        }
        
        generatedWords = []
        
        for _ in 1...10 {
            var index = -1
            var wordConfidence = Float(0)
            var word = ""
            var indicesBlock = Array(repeating: UInt8(0), count: blockSize)

            while index != 0 {
                autoreleasepool {
                    let input = arrayToMPSTensorData(array: indicesBlock, 
                                                     dataType: .uInt8,
                                                     shape: [1 as NSNumber, 
                                                             blockSize as NSNumber])
                    
                    let results = graph.run(feeds: [inputsPlaceholderTensor!: input], 
                                            targetTensors: [softMax!], 
                                            targetOperations: [])
                    
                    
                    // Get the softMax results
                    let softMaxData = results[softMax!]
                    
                    //print(softMaxData)
                    
                    var softMaxArray = Array(repeating: Float(0), count: vocabSize)
                    
                    softMaxData?.mpsndarray().readBytes(&softMaxArray, strideBytes: nil)
                    
                    //print(softMaxArray)
                    var confidence = Float(0)
                   ( index, confidence) = NamesMLP.multinomial(softMaxArray)
                    
                    //print("index \(index)")
                    
                    indicesBlock = indicesBlock.dropFirst() + [UInt8(index)]
                    
                    if index != 0 {
                        wordConfidence += confidence
                        word += chars[index]
                    }
                }
            }
            generatedWords += [(word, wordConfidence)]
            //generatedText += word + ", \(wordConfidence / Float(word.count))\n"
        }
        
      //  print(generatedText)
    }
    
    // Convert a training batch to MPSGraphTensorData
    func arrayToMPSTensorData<T>(array: Array<T>, 
                                 dataType: MPSDataType, 
                                 shape: [NSNumber]) -> MPSGraphTensorData {
        var source = array // we need a var for writeBytes
        let mpsNDArray = MPSNDArray(device: device,
                                    descriptor: MPSNDArrayDescriptor(dataType: dataType, 
                                                                     shape: shape))
        
        mpsNDArray.writeBytes(&source, strideBytes: nil)
        
        return MPSGraphTensorData(mpsNDArray)
        
    }
    
    // Returns the index and confidence
    static func multinomial(_ p: [Float]) -> (Int, Float) {
        let r = Float.random(in: 0..<1)
        var acc = Float(0)
        for i in 0..<p.count {
            acc += p[i]
            if r < acc {
                return ( i, p[i])
            }
        }
        return ( p.count - 1, p.last!)
    }
    
}
