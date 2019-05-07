import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = [str(i) for i in range(1, 769)]
y_pos = np.arange(len(objects))
score = np.loadtxt("car.np")

plt.bar(y_pos, score, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Score')
plt.title('Attention: car')

plt.show()