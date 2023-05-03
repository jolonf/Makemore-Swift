import SwiftUI

struct ContentView: View {
    
    @ObservedObject var namesMLP = NamesMLP()
    @State var initializing = false
    @State var initialized = false
    
    var body: some View {
        VStack {

            Text("NamesMLP").padding()
            Divider()
            VStack() {  
                VStack(alignment: .leading) {
                    Button(action: {
                        initializing = true
                        initialized = false
                        Task {
                            await namesMLP.initialize()
                            initializing = false
                            initialized = true
                        }
                        
                    }) {
                        Label("1. Initialize", systemImage: "square.and.arrow.down").lineLimit(1)
                    }.padding()
                    
                    Button(action: {
                        Task {
                            await namesMLP.train()
                        }
                    }) {
                        Label("2. Train", systemImage: "brain.head.profile")
                    }.disabled(!initialized).padding()
                    
                    Button(action: {
                        Task {
                            await namesMLP.generate()
                        }
                    }) {
                        Label("3. Generate", systemImage: "text.word.spacing").lineLimit(1)
                    }.disabled(!initialized).padding()
                }
                
                VStack(alignment: .leading) {
                    Text("Initializing...").opacity(initializing ? 1 : 0)
                    Text("Loss: \(namesMLP.lossValue)")
                    Text("Iterations: \(namesMLP.iterations)")
                    Text("Training time: \(String(describing: namesMLP.trainingTime))")
                }
                
                Divider()
                
                VStack(alignment: .leading) {
                    ForEach( namesMLP.generatedWords, id: \.0 ) { (pair:(String, Float)) in
                        let (word, confidence) = pair
                      Text(word).foregroundColor(Color(red: Double( 1.0), green: Double(confidence), blue: Double(confidence)))
                    }
                    
                }.padding()
            }
            Spacer()
        }
    }
}
