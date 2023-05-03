import SwiftUI

struct ContentView: View {
    
    @ObservedObject var model = BigramGraph(fileName: "shakespeare")
    @State var initializing = false
    @State var initialized = false
    
    var body: some View {
        VStack {

            Text("Shakespeare Bigram").padding()
            
            Divider()
            
            VStack() {  
                VStack(alignment: .leading) {
                    Button(action: {
                        initializing = true
                        initialized = false
                        Task {
                            await model.initialize()
                            initializing = false
                            initialized = true
                        }
                        
                    }) {
                        Label("1. Initialize", systemImage: "square.and.arrow.down").lineLimit(1)
                    }.padding()
                    
                    Button(action: {
                        Task {
                            await model.train()
                        }
                    }) {
                        Label("2. Train", systemImage: "brain.head.profile")
                    }.disabled(!initialized).padding()
                    
                    Button(action: {
                        Task {
                            await model.generate()
                        }
                    }) {
                        Label("3. Generate", systemImage: "text.word.spacing").lineLimit(1)
                    }.disabled(!initialized).padding()
                }
                
                VStack(alignment: .leading) {
                    Text("Initializing...").opacity(initializing ? 1 : 0)
                    Text("Loss: \(model.lossValue)")
                    Text("Iterations: \(model.iterations)")
                    Text("Training time: \(String(describing: model.trainingTime))")
                }
                
                Text(model.attributedText)
                    .padding()
                    .frame(maxWidth: .infinity, 
                           alignment: .leading)
                    .background(.quaternary)
            }
            Spacer()
        }
    }
}
