# # # import matplotlib.pyplot as plt

# # # # GCN+Norm data
# # # gcn_norm_train_loss = [1.7003, 1.6208, 1.6087, 1.5794, 1.5635, 1.5357, 1.5406, 1.5527, 1.5291, 1.5290]
# # # gcn_norm_valid_loss = [1.7341, 1.7257, 1.7019, 1.6838, 1.6326, 1.6272, 1.6579, 1.6233, 1.6426, 1.6317]

# # # # GCN data
# # # gcn_train_loss = [1.8609, 1.7732, 1.7438, 1.7036, 1.6919, 1.6705, 1.6663, 1.6508, 1.6346, 1.6312]
# # # gcn_valid_loss = [1.8529, 1.7918, 1.7551, 1.7336, 1.6916, 1.6696, 1.6984, 1.6401, 1.6584, 1.6419]

# # # # GraphSage data
# # # graphsage_train_loss = [1.7016, 1.6333, 1.6402, 1.6285, 1.5571, 1.5295, 1.5382, 1.5427, 1.5344, 1.5041]
# # # graphsage_valid_loss = [1.7306, 1.7251, 1.7144, 1.6887, 1.6440, 1.6060, 1.6724, 1.6257, 1.6463, 1.6403]

# # # epochs = range(5, 55, 5)

# # # # Plot GCN+Norm
# # # plt.figure(figsize=(6, 4))
# # # plt.plot(epochs, gcn_norm_train_loss, 'b-o', label='GCN+Norm Train Loss')
# # # plt.plot(epochs, gcn_norm_valid_loss, 'b--o', label='GCN+Norm Valid Loss')
# # # plt.xlabel('Epochs')
# # # plt.ylabel('Loss')
# # # plt.title('GCN+Norm')
# # # plt.legend()
# # # plt.grid(True)
# # # plt.show()

# # # # Plot GCN
# # # plt.figure(figsize=(6, 4))
# # # plt.plot(epochs, gcn_train_loss, 'g-o', label='GCN Train Loss')
# # # plt.plot(epochs, gcn_valid_loss, 'g--o', label='GCN Valid Loss')
# # # plt.xlabel('Epochs')
# # # plt.ylabel('Loss')
# # # plt.title('GCN')
# # # plt.legend()
# # # plt.grid(True)
# # # plt.show()

# # # # Plot GraphSage
# # # plt.figure(figsize=(6, 4))
# # # plt.plot(epochs, graphsage_train_loss, 'r-o', label='GraphSage Train Loss')
# # # plt.plot(epochs, graphsage_valid_loss, 'r--o', label='GraphSage Valid Loss')
# # # plt.xlabel('Epochs')
# # # plt.ylabel('Loss')
# # # plt.title('GraphSage')
# # # plt.legend()
# # # plt.grid(True)
# # # plt.show()

# # import matplotlib.pyplot as plt

# # # GCN+Norm data
# # # GCN+Norm data
# # gcn_norm_epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# # gcn_norm_train_loss = [1.7436, 1.6805, 1.6614, 1.6530, 1.6435, 1.6397, 1.6408, 1.6267, 1.6336, 1.6276, 1.6340, 1.6207, 1.6228, 1.6223, 1.6136, 1.6084, 1.5334, 1.5294, 1.5231, 1.5236]
# # gcn_norm_valid_loss = [1.7635, 1.7016, 1.6557, 1.6424, 1.6399, 1.6474, 1.6551, 1.6462, 1.6378, 1.6555, 1.6445, 1.6393, 1.6342, 1.6304, 1.6463, 1.6407, 1.6068, 1.5819, 1.5651, 1.5585]
# # gcn_norm_accuracy = [0.4676, 0.5521, 0.6130, 0.6169, 0.6346, 0.6189, 0.6267, 0.6385, 0.6346, 0.6031, 0.6149, 0.6385, 0.6385, 0.6464, 0.6326, 0.6542, 0.8409, 0.8409, 0.8487, 0.8585]

# # # GCN data
# # gcn_epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# # gcn_train_loss = [1.9367, 1.9288, 1.9206, 1.9143, 1.8143, 1.7785, 1.7776, 1.7657, 1.7652, 1.7593, 1.7661, 1.7539, 1.7570, 1.7176, 1.6573, 1.6393, 1.6347, 1.6327, 1.6103, 1.5869]
# # gcn_valid_loss = [1.9499, 1.9483, 1.9451, 1.9429, 1.8099, 1.7950, 1.7947, 1.8009, 1.7959, 1.7948, 1.7922, 1.7909, 1.7933, 1.7489, 1.6616, 1.6523, 1.6497, 1.6586, 1.6125, 1.6179]
# # gcn_accuracy = [0.2554, 0.2574, 0.2574, 0.2593, 0.4283, 0.4381, 0.4361, 0.4361, 0.4361, 0.4401, 0.4381, 0.4381, 0.4361, 0.5737, 0.7446, 0.7603, 0.7446, 0.7505, 0.7937, 0.8173]

# # # GraphSage data
# # graphsage_epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# # graphsage_train_loss = [1.8852, 1.7841, 1.7653, 1.7567, 1.6976, 1.6349, 1.6222, 1.6129, 1.5815, 1.5627, 1.5583, 1.5540, 1.5528, 1.5494, 1.5533, 1.5457, 1.5506, 1.5365, 1.5265, 1.5209]
# # graphsage_valid_loss = [1.8342, 1.7901, 1.7870, 1.7826, 1.7230, 1.6441, 1.6288, 1.6252, 1.5762, 1.5677, 1.5569, 1.5580, 1.5527, 1.5524, 1.5512, 1.5503, 1.5490, 1.5443, 1.5435, 1.5400]
# # graphsage_accuracy = [0.4244, 0.4401, 0.4420, 0.4420, 0.6424, 0.7760, 0.7859, 0.7976, 0.8310, 0.8389, 0.8409, 0.8409, 0.8487, 0.8487, 0.8507, 0.8566, 0.8566, 0.8625, 0.8684, 0.8880]

# # epochs = range(5, 105, 5)

# # # Create subplots for each model in a single row
# # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# # # GCN+Norm subplot
# # axes[0].plot(epochs, gcn_norm_train_loss, 'b-o', label='Train Loss')
# # axes[0].plot(epochs, gcn_norm_valid_loss, 'b--o', label='Valid Loss')
# # axes[0].set_xlabel('Epochs')
# # axes[0].set_ylabel('Loss')
# # axes[0].set_title('GCN+Norm')
# # axes[0].legend()
# # axes[0].grid(True)

# # # GCN subplot
# # axes[1].plot(epochs, gcn_train_loss, 'g-o', label='Train Loss')
# # axes[1].plot(epochs, gcn_valid_loss, 'g--o', label='Valid Loss')
# # axes[1].set_xlabel('Epochs')
# # axes[1].set_ylabel('Loss')
# # axes[1].set_title('GCN')
# # axes[1].legend()
# # axes[1].grid(True)

# # # GraphSage subplot
# # axes[2].plot(epochs, graphsage_train_loss, 'r-o', label='Train Loss')
# # axes[2].plot(epochs, graphsage_valid_loss, 'r--o', label='Valid Loss')
# # axes[2].set_xlabel('Epochs')
# # axes[2].set_ylabel('Loss')
# # axes[2].set_title('GraphSage')
# # axes[2].legend()
# # axes[2].grid(True)

# # # Adjust spacing between subplots
# # plt.tight_layout()

# # # Show the combined plot
# # plt.show()

# import matplotlib.pyplot as plt

# # GCN data
# gcn_epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# gcn_train_loss = [2.0212, 1.9595, 1.9530, 1.9385, 1.9302, 1.9265, 1.9292, 1.9237, 1.9215, 1.9149, 1.9201, 1.9162, 1.9139, 1.9100, 1.9112, 1.9102, 1.9098, 1.9077, 1.9126, 1.9096]
# gcn_valid_loss = [2.0027, 1.9702, 1.9629, 1.9554, 1.9509, 1.9482, 1.9482, 1.9478, 1.9490, 1.9521, 1.9479, 1.9486, 1.9440, 1.9447, 1.9453, 1.9477, 1.9437, 1.9435, 1.9457, 1.9458]
# gcn_accuracy = [0.2456, 0.2534, 0.2534, 0.2554, 0.2574, 0.2574, 0.2574, 0.2574, 0.2574, 0.2554, 0.2554, 0.2593, 0.2574, 0.2574, 0.2574, 0.2593, 0.2593, 0.2593, 0.2593, 0.2574]

# # GCN+Norm data
# gcn_norm_epochs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# gcn_norm_train_loss = [1.8486, 1.7950, 1.7815, 1.7484, 1.7197, 1.7090, 1.7121, 1.7068, 1.7015, 1.6948, 1.6965, 1.6977, 1.6942, 1.6884, 1.6883, 1.6908, 1.6908, 1.6877, 1.6890, 1.6942]
# gcn_norm_valid_loss = [1.8592, 1.8201, 1.8081, 1.7817, 1.7583, 1.7564, 1.7551, 1.7508, 1.7562, 1.7502, 1.7541, 1.7481, 1.7517, 1.7480, 1.7476, 1.7560, 1.7441, 1.7469, 1.7525, 1.7435]
# gcn_norm_accuracy = [0.5658, 0.5776, 0.5914, 0.6090, 0.6149, 0.6090, 0.6110, 0.6071, 0.6189, 0.6169, 0.6130, 0.5972, 0.6189, 0.6149, 0.6169, 0.6071, 0.6169, 0.6189, 0.6189, 0.6228]

# # # Create a figure and axis
# # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# # # GCN plot
# # axes[0].plot(gcn_epochs, gcn_train_loss, label='Train loss')
# # axes[0].plot(gcn_epochs, gcn_valid_loss, label='Valid loss')
# # axes[0].set_xlabel('Epochs')
# # axes[0].set_ylabel('Loss')
# # axes[0].set_title('GCN Training and Validation Loss')
# # axes[0].legend()
# # axes[0].grid(True)

# # # GCN+Norm plot
# # axes[1].plot(gcn_norm_epochs, gcn_norm_train_loss, label='Train loss')
# # axes[1].plot(gcn_norm_epochs, gcn_norm_valid_loss, label='Valid loss')
# # axes[1].set_xlabel('Epochs')
# # axes[1].set_ylabel('Loss')
# # axes[1].set_title('GCN+Norm Training and Validation Loss')
# # axes[1].legend()
# # axes[1].grid(True)

# # # GCN accuracy plot
# # axes[2].plot(gcn_epochs, gcn_norm_accuracy, label='GCN+Norm accuracy')
# # axes[2].plot(gcn_epochs, gcn_accuracy, label='GCN accuracy')
# # axes[2].set_xlabel('Epochs')
# # axes[2].set_ylabel('Accuracy')
# # axes[2].set_title('GCN Accuracy')
# # axes[2].legend()
# # axes[2].grid(True)

# # # Adjust the spacing between subplots
# # plt.tight_layout()

# #plot train loss for the two models
# plt.figure(figsize=(8, 4))
# plt.plot(gcn_norm_epochs, gcn_norm_train_loss, 'b-o', label='GCN+Norm')
# plt.plot(gcn_epochs, gcn_train_loss, 'g-o', label='GCN')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)

# # Display the plot
# plt.show()

import matplotlib.pyplot as plt

# Training time data
gcn_kitti_time = 99.88
gcn_modelnet10_time = 108.84
graphsage_kitti_time = 77.73
graphsage_modelnet10_time = 48.03

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Set the bar positions and heights
bar_positions = [0, 1, 2, 3]
bar_heights = [gcn_kitti_time, graphsage_kitti_time, gcn_modelnet10_time, graphsage_modelnet10_time]

# Set the labels for the bars
bar_labels = ['GCN (KITTI)', 'GraphSage (KITTI)', 'GCN (ModelNet10)', 'GraphSage (ModelNet10)']

# Set the colors for the bars
bar_colors = ['lightsteelblue', 'lightcoral', 'lightsteelblue', 'lightcoral']

# Create the bars
bars = ax.bar(bar_positions, bar_heights, tick_label=bar_labels, color=bar_colors, width=0.75)

# Set the x-axis label
ax.set_xlabel('Model and Dataset')

# Set the y-axis label
ax.set_ylabel('Training Time (seconds)')

# Set the plot title

# Show the gridlines
ax.grid(True, axis='y')

# Adjust the spacing between x-axis labels
#plt.xticks(rotation=45)

# Attach a text label above each bar displaying the precise time
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height,
            f'{height:.2f}', ha='center', va='bottom')


# Display the plot
plt.show()
