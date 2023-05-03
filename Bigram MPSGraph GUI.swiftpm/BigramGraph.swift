import SwiftUI
import MetalPerformanceShadersGraph

/**
 Bigram implementation using MPSGraph.
 Based on Andrej Karpathy's nanoGPT video: https://www.youtube.com/watch?v=kCc8FmEb1nY
 */
class BigramGraph: ObservableObject {
    
    @Published var initialized = false
    @Published var generatedText = ""
    @Published var attributedText = AttributedString()
    @Published var lossValue: Float = 0.0
    @Published var iterations = 0
    @Published var trainingTime: Duration = .seconds(0)
    
    let device = MTLCreateSystemDefaultDevice()!
    
    var fileName = ""
    var vocab: [Character] = []
    var trainData: [UInt8] = []
    var valData: [UInt8] = []
    var vocabSize: Int = 0
    var batchSize = 32
    var blockSize = 8

    var graph: MPSGraph?
    
    // Tensors
    var inputsPlaceholder, 
        targetsPlaceholder,
        embeddingWeights,
        inferenceOutput,
        trainingOutput: MPSGraphTensor?
    
    var optimizationOps: [MPSGraphOperation] = []
    
    
    /**
     - parameter fileName: Doesn't include extension, will add ".txt"
     */
    init(fileName: String) {
        self.fileName = fileName
    }
    
    /**
     Reads in the text file and creates data arrays.
     */
    func loadData() {
        // Load names file as a string
        let text = try! String(contentsOf: Bundle.main.url(forResource: fileName, 
                                                           withExtension: ".txt")!)
        
        vocab = Array(Set(text.map { $0 })).sorted()
        vocabSize = vocab.count
        
        print("Vocab size: \(vocabSize)")
        //print(vocab)
        
        // Flip the keys and values to convert from chars to indexes
        let stoi: [Character: UInt8] = Dictionary(uniqueKeysWithValues: 
                                                    vocab.enumerated()
            .map( { k, v in (v, UInt8(k)) } ))
        
        //print(stoi)
        
        let encode = { (string: String) -> [UInt8] in 
            string.map { stoi[$0]! } }
        
        let decode = { (indices: [UInt8]) -> String in
            String(indices.map { self.vocab[Int($0)] } )
        }
        
        print(encode("HI there!!!"))
        print(decode(encode("HI there!!!")))
        
        let data: [UInt8] = encode(text)
        
        let n = Int(0.9 * Double(data.count))
        
        trainData = Array(data[..<n])
        valData = Array(data[n...])
    }
    
    /**
     Loads the data from file and builds the MPSGraph.
     This function can be called to reset the graph.
     */
    func initialize() async {
        
        initialized = false
        iterations = 0
        
        print("--- Initializing ---")
        
        loadData()
        
        // --- BUILD NN ---
        
        let embeddingSize = vocabSize
        
        graph = MPSGraph()
        
        guard let graph else {
            print("Couldn't create MPSGraph")
            return
        }

        // --- Create placeholders for inputs ---
        
        inputsPlaceholder = graph.placeholder(
            shape: [-1 as NSNumber, // batchSize
                     blockSize as NSNumber], 
            dataType: .uInt8, name: "Input Indices")
        
        targetsPlaceholder = graph.placeholder(
            shape: [-1 as NSNumber, // batchSize
                     blockSize as NSNumber], 
            dataType: .uInt8, name: "Target Indices")

        // --- Embedding Layer --- aka: E = C[X]
        
        let embeddingsLayerOutput: MPSGraphTensor
        (embeddingWeights, embeddingsLayerOutput) = embeddingLayer(vocabSize: vocabSize, 
                                                             embeddingSize: embeddingSize)

        print("embeddingsLayerOutput.shape: \(embeddingsLayerOutput.shape!)")
        
        // --- Inference output --- aka: Softmax
        
        inferenceOutput = graph.softMax(with: embeddingsLayerOutput, axis: -1, name: "softMax")

        print("inferenceOutput.shape: \(inferenceOutput!.shape!)")

        // --- Training output --- aka: loss SoftmaxCrossEntropy
        
        // [batchSize, blockSize, vocabSize]
        let targetsOneHot = graph.oneHot(
            withIndicesTensor: targetsPlaceholder!, 
            depth: vocabSize, 
            name: "Targets OneHot Encoded")

        trainingOutput = graph.softMaxCrossEntropy(
            embeddingsLayerOutput, 
            labels: targetsOneHot,
            axis: -1,
            reuctionType: .mean, 
            name: "Loss: SoftMax Cross Entropy")

        print("trainingOutput.shape: \(trainingOutput!.shape!)")

        // --- Backprop --- aka: Adam optimizer
        
        let gradients = graph.gradients(of: trainingOutput!, 
                                        with: [embeddingWeights!], 
                                        name: "Gradients")
        
        optimizationOps = adamOptimizer(gradients: gradients)
        
        initialized = true
    }
    
    /**
     Adds an embedding layer to the graph.
     
     - returns: A tuple of the weights variable and the output tensor.
     */
    func embeddingLayer(vocabSize: Int, embeddingSize: Int) -> (MPSGraphTensor, MPSGraphTensor) {
        
        // One-hot the inputs
        
        // [batchSize, blockSize, vocabSize]
        let inputOneHot = graph!.oneHot(
            withIndicesTensor: inputsPlaceholder!, 
            depth: vocabSize, 
            name: "Input OneHot Encoded")
        
        let embeddingWeightsValues = (0..<(vocabSize * embeddingSize))
            .map { _ in Float.random(in: -0.2..<0.2) }
        
        let embeddingWeights = graph!.variable(with: Data(bytes: embeddingWeightsValues, 
                                                          count: vocabSize * embeddingSize * 4), 
                                               shape: [vocabSize as NSNumber, 
                                                       embeddingSize as NSNumber], 
                                               dataType: .float32, 
                                               name: "Embedding Weights")
        
        let embeddingsLayerOutput = graph!.matrixMultiplication(
            primary: inputOneHot, 
            secondary: embeddingWeights, 
            name: "Token embeddings")
        
        return (embeddingWeights, embeddingsLayerOutput)
    }
    
    // The adam optimizer is a little complicated, part of the calculation is beta1^t,
    // where t is the time step. The adam function doesn't want to calculate it for us
    // so we need to do it ourselves and pass in as beta1power and beta2power.
    // This is presumably because we can calculate it at each step by just 
    // multiplying by the original beta1.
    func adamOptimizer(gradients: [MPSGraphTensor: MPSGraphTensor]) -> [MPSGraphOperation] {
        
        var ops: [MPSGraphOperation] = []
        
        // Adam constants
        
        let learningRate = graph!.constant(0.01, dataType: .float32)
        let beta1 = graph!.constant(0.9, dataType: .float32)
        let beta2 = graph!.constant(0.999, dataType: .float32)
        let epsilon = graph!.constant(1e-8, dataType: .float32)
        
        // Adam variables
        
        let beta1powerArray = [Float(0.9)]
        let beta1power = graph!.variable(with: Data(bytes: beta1powerArray, count: 4), 
                                        shape: [1 as NSNumber], 
                                        dataType: .float32, 
                                        name: "beta1power")
        
        let beta2powerArray = [Float(0.999)]
        let beta2power = graph!.variable(with: Data(bytes: beta2powerArray, count: 4), 
                                        shape: [1 as NSNumber], 
                                        dataType: .float32, 
                                        name: "beta2power")
        
        for (tensor, gradient) in gradients {

            // Multiply elements of the shape array to get the size of the variable buffers
            let arraySize = tensor.shape!.reduce(1) { (result, dim) -> Int in
                result * Int(truncating: dim)
            }
            
            // Each tensor will have its own copy of the momentum and velocity
            // tensors, which will be the same shape as the original tensor
            
            let momentumArray = Array(repeating: Float(0), count: arraySize) // must be same size as tensor
            let momentum = graph!.variable(with: Data(bytes: momentumArray, count: arraySize * 4), 
                                          shape: tensor.shape!, 
                                          dataType: tensor.dataType, 
                                          name: "momentum")
            
            let velocityArray = Array(repeating: Float(0), count: arraySize) // must be same size as tensor
            let velocity = graph!.variable(with: Data(bytes: velocityArray, count: arraySize * 4), 
                                          shape: tensor.shape!, 
                                          dataType: tensor.dataType, 
                                          name: "velocity")
            
            let adamResults = graph!.adam(learningRate: learningRate, 
                                         beta1: beta1, 
                                         beta2: beta2, 
                                         epsilon: epsilon, 
                                         beta1Power: beta1power,
                                         beta2Power: beta2power,
                                         values: tensor, 
                                         momentum: momentum, 
                                         velocity: velocity, 
                                         maximumVelocity: nil, 
                                         gradient: gradient, 
                                         name: nil)
            
            let newTensor = adamResults[0]
            let newMomentum = adamResults[1]
            let newVelocity = adamResults[2]
            
            // Update the beta power values
            let newBeta1power = graph!.multiplication(beta1power, beta1, name: "beta1power update")
            let newBeta2power = graph!.multiplication(beta1power, beta2, name: "beta2power update")
            
            // Assign new values and keep a copy of the ops
            ops += [graph!.assign(beta1power, tensor: newBeta1power, name: "Assign beta1power")]
            ops += [graph!.assign(beta2power, tensor: newBeta2power, name: "Assign beta2power")]
            
            ops += [graph!.assign(momentum, tensor: newMomentum, name: "Assign momentum")]
            ops += [graph!.assign(velocity, tensor: newVelocity, name: "Assign velocity")]
            
            ops += [graph!.assign(tensor, tensor: newTensor, name: "Assign")]
        }
        
        return ops
    }
    
    // Gets a batch of data from the training or testing set split
    // - parameter split: "train" for training set, otherwise validation set
    func getBatch(_ split: String) -> ([UInt8], [UInt8]) {
        let data = split == "train" ? trainData : valData
        
        // Randomly pick a starting index in the data set for each batch
        let ix = (0..<batchSize).map { _ in Int.random(in: 0..<data.count - blockSize) }
        // Grab the input data for each batch
        let x = Array(ix.map { data[$0 ..< ($0 + blockSize)] }.joined())
        // Grab the target data for each batch
        let y = Array(ix.map { data[($0 + 1) ..< ($0 + blockSize + 1)] }.joined())
        
        return (x, y)
    }

    // Get a batch in MPSGraph format
    func getMPSBatch(_ split:String) -> (MPSGraphTensorData, MPSGraphTensorData) {
        let (xb, yb) = getBatch(split)
        
        let inputTensorData = arrayToMPSTensorData(array: xb, 
                                                   dataType: .uInt8, 
                                                   shape: [batchSize as NSNumber, 
                                                           blockSize as NSNumber])
        
        let targetTensorData = arrayToMPSTensorData(array: yb, 
                                                    dataType: .uInt8, 
                                                    shape: [batchSize as NSNumber, 
                                                            blockSize as NSNumber])
        
        return (inputTensorData, targetTensorData)
    }

    
    func train() async {
        
        print("--- Training ---")
    
        guard let graph else {
            print("MPSGraph not initialized")
            return
        }
        
        // --- TRAIN ---
        
        let clock = ContinuousClock()

        trainingTime = clock.measure {
            for epoch in 1 ... 5000 {

                // This is necessary so graph.run() doesn't crash
                autoreleasepool {
                    
                    let (input, target) = getMPSBatch("train")

                    let results = graph.run(feeds: [inputsPlaceholder!: input, 
                                                  targetsPlaceholder!: target], 
                                            targetTensors:  [trainingOutput!], 
                                            targetOperations: optimizationOps)

                    // Get the loss results
                    let lossData = results[trainingOutput!]
                    
                    var lossFloat: Float = 0
                    
                    lossData?.mpsndarray().readBytes(&lossFloat, strideBytes: nil)
                    
                    if epoch % 100 == 0 {
                        print("\(epoch). loss = \(lossValue)")
                        
                        iterations += 100
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
        
        generatedText = ""
        attributedText = AttributedString()
        var indicesBlock = Array(repeating: UInt8(0), count: blockSize)

        for _ in 1...100 {

            autoreleasepool {
                let input = arrayToMPSTensorData(array: indicesBlock, 
                                                 dataType: .uInt8,
                                                 shape: [1 as NSNumber, 
                                                         blockSize as NSNumber])
                
                let results = graph.run(feeds: [inputsPlaceholder!: input], 
                                        targetTensors: [inferenceOutput!], 
                                        targetOperations: [])
                
                
                // Get the softMax results
                let softMaxData = results[inferenceOutput!]
                
                //print(softMaxData)
                
                var softMaxArray = Array(repeating: Float(0), count: blockSize * vocabSize)
                
                softMaxData?.mpsndarray().readBytes(&softMaxArray, strideBytes: nil)
                
                // We only want the last block/time
                let softMaxLast = Array(softMaxArray[((blockSize - 1) * vocabSize) ..< (vocabSize * blockSize)])
                
                //print(softMaxLast)
                
                let (index, confidence) = BigramGraph.multinomial(softMaxLast)
                
                //print("index \(index)")
                
                indicesBlock = indicesBlock.dropFirst() + [UInt8(index)]
                
                let nextChar = String(vocab[index])
                generatedText += nextChar
                
                // We also generate an attributed text string where each character's
                // color reflects its confidence
                var nextAttributedChar = AttributedString(nextChar)
                nextAttributedChar.foregroundColor = Color(red: 1.0, green: Double(confidence*10), blue: Double(confidence*10))
                attributedText += nextAttributedChar
            }
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
