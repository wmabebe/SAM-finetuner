import re
import matplotlib.pyplot as plt

# Initialize an empty list to store train_loss values
largest, smallest, medium = [], [], []

# Open the file
with open('logs/2024-05-21--21:32:35.081980_dataset[sa1b]_trainable[em]_epochs[100]_lr[1e-05]_local_bs[24]/experiment_log-2024-05-21-21:32-35-results.log', 'r') as file:
    # Read each line in the file
    for line in file:
        # Check if the line contains the substring 'Smallest'
        if 'Largest' in line:
            # Extract train_loss value using regular expression
            train_loss = re.search(r'train_loss\':\s+([\d.]+)', line)
            if train_loss:
                # Append the train_loss value to the list
                largest.append(float(train_loss.group(1)))
        elif 'Smallest' in line:
            # Extract train_loss value using regular expression
            train_loss = re.search(r'train_loss\':\s+([\d.]+)', line)
            if train_loss:
                # Append the train_loss value to the list
                smallest.append(float(train_loss.group(1)))
        elif 'Medium' in line:
            # Extract train_loss value using regular expression
            train_loss = re.search(r'train_loss\':\s+([\d.]+)', line)
            if train_loss:
                # Append the train_loss value to the list
                medium.append(float(train_loss.group(1)))

# Plot the collected train_loss values
plt.plot(largest,label='Largest')
plt.plot(smallest,label='Smallest')
plt.plot(medium,label='Random')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Train Loss')
plt.title('Train Loss over Iterations')
plt.grid(True)
plt.show()
plt.savefig('loss.jpg')