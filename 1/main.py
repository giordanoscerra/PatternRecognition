import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt

# Define CWT function
def plot_cwt(ax, data, wavelet='morl', title=''):
    scales = np.arange(1, 100)
    cwtmatr, freqs = pywt.cwt(data, scales, wavelet)
    cax = ax.imshow(cwtmatr, aspect='auto', cmap='PiYG', extent=[0, len(data), freqs[-1], freqs[0]])
    ax.set_title(title)
    return cax  # Return the QuadMesh object for colorbar

# Choose a participant
participant = 3
# Load data
participant_data = pd.read_csv(f'1/data/{participant}.csv')  # replace with actual file path
# Add column names to the dataframe
participant_data.columns = ['id', 'x', 'y', 'z', 'label']
# Should be from 1 to 7, but to be sure let's read the csv. Last element is always 0, so let's ignore it.
activities = participant_data['label'].unique()[:-1]
# the three axis
channels = ['x', 'y', 'z']

# Utility dictionary to map activity number to description
description_dict = {1: 'Working at Computer', 
                    2: 'Standing Up, Walking and Going up\down stairs', 
                    3: 'Standing', 
                    4: 'Walking', 
                    5: 'Going Up\Down Stairs', 
                    6: 'Walking and Talking with Someone', 
                    7: 'Talking while Standing'}

for activity in activities:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True, constrained_layout=True)
    fig.suptitle(f'Activity {activity}/7: {description_dict[activity]}')
    
    caxes = []  # To collect the QuadMesh objects for colorbar
    for i, channel in enumerate(channels):
        activity_data = participant_data[participant_data['label'] == activity][channel]
        cax = plot_cwt(axs[i], activity_data, title=f'Channel: {channel}')
        caxes.append(cax)
        
    print(f"Activity {activity}/7: {description_dict[activity]}")
    
    # Create a colorbar with a reference to the QuadMesh objects
    fig.colorbar(caxes[0], ax=axs, orientation='vertical', fraction=0.025, pad=0.05)
    
    plt.show()


