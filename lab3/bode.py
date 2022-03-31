from scipy import signal

import matplotlib.pyplot as plt
L=100*10**-3
C=(470*10**-6)+(100*10**-9)

sys = signal.TransferFunction([1],[L*C,0,1])

w,mag,phase = signal.bode(sys)

plt.semilogx(w/(2*3.14),mag)

plt.show()

