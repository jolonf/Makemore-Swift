import SwiftUI

/**
 An implementation of the Makemore Bigram model from
 Andrej Karpathy's video: https://youtu.be/PaCmpygFfXo
 */
func makemore() -> ([[Int]], [String]) {
    
    // Load names file as a string
    let content = try! String(contentsOf: Bundle.main.url(forResource: "names", withExtension: ".txt")!)
    
    // Split file into lines (i.e. words)
    let words = content.components(separatedBy: .newlines)
    
    // Join all the words back together, convert each char to a string, add to set to find out which chars are in the file, sort, add a "." at the start which will be used as a token to represent the start and end of words
    let chars = ["."] + Array(Set(words.joined().map {String($0)})).sorted()
    
    // String to Index lookup
    let stoi: [String:Int] = chars.enumerated().reduce(into: [:]) { (result, element) in
        let (i, s) = element
        result[s] = i
    }
    
    // Bigram matrix
    var N = [[Int]](repeating: [Int](repeating: 0, count: chars.count), count: chars.count)

    // Create the bigram matrix
    for w in words {
        let chs = "." + w + "."
        for (ch1, ch2) in zip(chs, chs.dropFirst()) {
            let ix1: Int = stoi[String(ch1)]!
            let ix2: Int = stoi[String(ch2)]!
            N[ix1][ix2] += 1
        }
    }
    
    return (N, chars)
}


struct Makemore: View {

    @State var data: [[Int]] = [[]]
    @State var chars: [String] = []
    
    var body: some View {
        Grid {
            ForEach(0 ..< data.count, id: \.self) { rowIndex in 
                let row = data[rowIndex]
                GridRow() {
                    ForEach(0 ..< row.count, id: \.self) { colIndex in
                        let count = row[colIndex]
                        VStack() {
                            let text = chars[rowIndex] + " ➝ " + chars[colIndex] + "\n" + String(count)
                            Text(text)
                            .minimumScaleFactor(0.05)
                        }
                        .background(Color(white: Double(count) / 10000.0))
                        
                    }
                }
            }
        }.onAppear(perform: {
            (data, chars) = makemore()
        })
    }
}

struct Makemore_Previews: PreviewProvider {
    static var previews: some View {
        Makemore()
    }
}
