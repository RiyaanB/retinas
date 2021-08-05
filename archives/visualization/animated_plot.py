from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from pathlib import Path
import matplotlib.colors as colors
import numpy as np
source_file = Path("current_tracking.tmp")

# Generate colors for plot
color_list = []
MAX_NR_VALUES = 512
for i in range(MAX_NR_VALUES):
    rnd_hsv = np.random.rand(3)
    hsv_adjusted = rnd_hsv + np.array([0,0,0.5])
    # divide by largest value to ensure value between 0 and 1
    hsv = hsv_adjusted/np.amax(hsv_adjusted)
    color_list.append(colors.hsv_to_rgb(hsv)) 

fig = plt.figure()
ax1 = plt.axes(projection='3d')

def animate(i):
    if source_file.exists():
        pullData = open("current_tracking.tmp","r").read()
        dataArray = pullData.split('\n')
        tags = set()
        coordinates = {}
        for eachLine in dataArray:
            if len(eachLine)>1:
                tag,x,y,z = eachLine.split(',')
                tags.add(tag) # add tag to the set of tags detected
                if not tag in coordinates:
                    coordinates[tag] = [[],[],[]]
                coordinates[tag][0].append(int(x))
                coordinates[tag][1].append(int(y))
                coordinates[tag][2].append(int(z))
        ax1.clear()
        for i, cur_tag in enumerate(tags):
            #print("Plotting {} with {} datapoints".format(cur_tag, len(coordinates[cur_tag][0])))
            ax1.plot3D(coordinates[cur_tag][0], coordinates[cur_tag][1], coordinates[cur_tag][2], color = color_list[i%147])
            ax1.scatter3D(coordinates[cur_tag][0], coordinates[cur_tag][1], coordinates[cur_tag][2], color = color_list[i%147])
    else:
        print("Source File Missing")

if __name__ == "__main__":
  ani = animation.FuncAnimation(fig, animate, interval=100)
  plt.show()
