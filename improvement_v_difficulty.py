import pickle
import numpy as np
import os; os.environ['DISPLAY'] = "localhost:10.0"
import matplotlib; matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt
import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd

base_id = 'tlm5fd1x'
our_id  = '12k6kxjh'

def get_nlls(id):
    with open(f'checkpoints/{id}/per_token_nlls.pkl','rb') as f:
        return np.array(pickle.load(f))

base_nlls = get_nlls(base_id)
our_nlls = get_nlls(our_id)

assert len(base_nlls) == len(our_nlls)
improvements = (base_nlls - our_nlls)


# # Density plot: improvements on x axis, base nlls on y axis
# sns.kdeplot(x=improvements, y=base_nlls,fill=True,)
# # add red vertical line at x=0
# plt.axvline(x=0, color='red')
# # label x axis with "Improvement" and y axis with "Base NLL"
# plt.show()

# Create a dataframe from your data
df = pd.DataFrame({'x': improvements, 'y': base_nlls})

# x_range and y_range: such that it includes 99.9% of the data
x_range = (df['x'].quantile(0.0005), df['x'].quantile(0.9995))
y_range = (df['y'].quantile(0.0005), df['y'].quantile(0.9995))
# Create a canvas
canvas = ds.Canvas(plot_width=800, plot_height=800, x_range=x_range, y_range=y_range)

# Aggregate data
agg = canvas.points(df, 'x', 'y')

# Create the image
img = tf.shade(agg)

# Convert the image to a format that can be displayed with matplotlib
img = tf.spread(img, px=1).to_pil()

# Display the image
plt.imshow(img, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), aspect='auto')
plt.xlabel('Improvements')
plt.ylabel('Base NLLs')
plt.title('Datashader Density Plot')
plt.show()

x = improvements  # your data for x-axis
y = base_nlls     # your data for y-axis

plt.hexbin(x, y, cmap='Blues', bins='log', gridsize=100)
plt.colorbar(label='Density')
plt.xlabel('Improvements')
plt.ylabel('Base NLLs')
plt.title('Hexbin Density Plot')
# vertical line for average improvement
plt.axvline(x=np.mean(improvements), color='red')
plt.show()