import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..
df = pd.read_csv('K:/DAT210x-master/Module3/Datasets/wheat.data')


fig = plt.figure()
#
# TODO: Create a new 3D subplot using fig. Then use the
# subplot to graph a 3D scatter plot using the area,
# perimeter and asymmetry features. Be sure to use the
# optional display parameter c='red', and also label your
# axes
# 
# .. your code here ..
f1 = fig.add_subplot(111, projection='3d')
f1.set_xlabel('area')
f1.set_ylabel('perimeter')
f1.set_zlabel('asymmetry')
f1.scatter(df.area, df.perimeter, df.asymmetry, c='red')

fig = plt.figure()
#
# TODO: Create a new 3D subplot using fig. Then use the
# subplot to graph a 3D scatter plot using the width,
# groove and length features. Be sure to use the
# optional display parameter c='green', and also label your
# axes
# 
# .. your code here ..
f2 = fig.add_subplot(111, projection='3d')
f2.set_xlabel('width')
f2.set_ylabel('groove')
f2.set_zlabel('length')
f2.scatter(df.width, df.groove, df.length, c='red')

plt.show()


