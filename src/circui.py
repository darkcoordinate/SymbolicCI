from nicegui import ui
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Operator
import numpy as np
from functools import reduce

# Function to generate the circuit plot
def update_circuit(angle):
    plt.close('all')  # Clear previous figures to save memory
    
    # 1. Create the Circuit
    qc = QuantumCircuit(5,3)
    qc.ry(angle, 1)  # RY gate controlled by the slider
    qc.x(1)
    qc.x(0)
    #qc.ccx(2,1, 0)
    qc.cx(1,0)
    qc.cx(2,1)
    qc.cx(2,0)

    sp = Operator(qc)
    #print(sp.data.real)
    qc.measure([0],[0])
    qc.measure([1],[1])
    qc.measure([2],[2])


    b0 = np.array([[1,0.0]])
    b0b0 = np.kron(b0,b0)
    b0b0b0b0b0 = reduce(np.kron, [b0,b0,b0,b0,b0])
    cc = sp@b0b0b0b0b0.T
    # Select the Aer simulator backend
    backend = Aer.get_backend('qasm_simulator')

    # Transpile and run the circuit
    tqc = transpile(qc, backend)
    counts = backend.run(tqc).result().get_counts()


    # # 2. Update the UI Plot Container
    # with plot_container:
    #     plot_container.clear()
    #     # Draw using 'mpl' (Matplotlib)
    #     fig = qc.draw(output='mpl')
    #     ui.pyplot(fig)
    
    # 3. Update the Text Output
    sll = str(cc.data.T.real)
    text_output.content = f'```\n\n{sll+"\n==================\n"+str(qc.draw(output="text")) + "\n\n"+str(counts)}\n```'

# --- UI Layout ---

ui.label('Quantum Circuit Controller').classes('text-h4 q-ma-md')

with ui.card().classes('w-full q-pa-md'):
    ui.label('Adjust RY Gate Angle (θ)')
    # Slider from 0 to 2π
    slider = ui.slider(min=0, max=6.28, step=0.01, value=1.57, on_change=lambda e: update_circuit(e.value))
    ui.label().bind_text_from(slider, 'value', backward=lambda v: f'Angle: {v:.2f} rad')

with ui.tabs().classes('w-full') as tabs:
    ui.tab('Text Output')

with ui.tab_panels(tabs, value='Text Output').classes('w-full'):
    with ui.tab_panel('Text Output'):
        text_output = ui.markdown()

# Initialize the display
update_circuit(1.57)

ui.run()
