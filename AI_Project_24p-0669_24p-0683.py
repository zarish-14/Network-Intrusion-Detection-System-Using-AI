#!/usr/bin/env python
# coding: utf-8

# # TASK 1 – DATA EXPLORATION

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("network_traffic.csv")

print("Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# In[2]:


print("\nStatistics:")
print(df.describe())


# In[3]:


plt.figure(figsize=(6,4))
df["label"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()


# In[4]:


# Histogram 1
plt.figure(figsize=(8,5))

plt.hist(
    df[df["label"]==0]["src_bytes"],
    bins=30,
    alpha=0.5,
    label="Normal"
)
plt.hist(
    df[df["label"]==1]["src_bytes"],
    bins=30,
    alpha=0.5,
    label="Attack"
)
plt.title("src_bytes Distribution")
plt.xlabel("src_bytes")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# In[5]:


# Histogram 2
plt.figure(figsize=(8,5))
plt.hist(
    df[df["label"]==0]["count"],
    bins=30,
    alpha=0.5,
    label="Normal"
)
plt.hist(
    df[df["label"]==1]["count"],
    bins=30,
    alpha=0.5,
    label="Attack"
)


# # TASK 2 – REFLEX AGENT

# In[7]:


import pandas as pd
df = pd.read_csv("network_traffic.csv")

def reflex_agent(row):
    if row["serror_rate"] > 0.5:
        return 1
    elif row["same_srv_rate"] < 0.5:
        return 1
    else:
        return 0

correct = 0

for i in range(len(df)):
    prediction = reflex_agent(df.iloc[i])

    if prediction == df.iloc[i]["label"]:
        correct += 1

accuracy = correct / len(df)

print("Accuracy =", accuracy)


# # TASK 3A – KNN

# In[8]:


import pandas as pd
import math

df = pd.read_csv("network_traffic.csv")
X = df.drop("label", axis=1).values.tolist()
y = df["label"].tolist()

training_data = []

for i in range(len(X)):
    row = X[i][:]
    row.append(y[i])
    training_data.append(row)

def euclidean_distance(point1, point2):
    total = 0
    for i in range(len(point1)):
        diff = point1[i] - point2[i]
        total += diff * diff
    return math.sqrt(total)

def get_neighbors(training_data, new_point, k):
    distances = []
    for row in training_data:
        features = row[:-1]
        label = row[-1]
        dist = euclidean_distance(new_point, features)
        distances.append([dist, label])
    distances.sort(key=lambda x: x[0])
    return distances[:k]

def classify(training_data, new_point, k):
    neighbors = get_neighbors(training_data, new_point, k)
    votes = {}
    for neighbor in neighbors:
        label = neighbor[1]
        if label in votes:
            votes[label] += 1
        else:
            votes[label] = 1
    best_label = max(votes, key=votes.get)
    return best_label

new_point = X[0]
result = classify(training_data, new_point, 3)

print("Prediction =", result)
print("Actual =", y[0])


# # TASK 3B – NAIVE BAYES

# In[9]:


import pandas as pd
df = pd.read_csv("network_traffic.csv")
data = df.values.tolist()
total_samples = len(data)

class_counts = {0:0, 1:0}
for row in data:
    class_counts[row[-1]] += 1

prior_normal = class_counts[0] / total_samples
prior_attack = class_counts[1] / total_samples

print("Prior Normal =", prior_normal)
print("Prior Attack =", prior_attack)

feature_counts_normal = {}
feature_counts_attack = {}

for row in data:
    label = row[-1]
    for i in range(len(row)-1):
        feature = row[i]
        if label == 0:
            if feature not in feature_counts_normal:
                feature_counts_normal[feature] = 0
            feature_counts_normal[feature] += 1
        else:
            if feature not in feature_counts_attack:
                feature_counts_attack[feature] = 0
            feature_counts_attack[feature] += 1

print("Feature Counting Complete")


# # TASK 3C – LOGISTIC REGRESSION

# In[10]:


import pandas as pd
import numpy as np

df = pd.read_csv("network_traffic.csv")
X = df.drop("label", axis=1).values
y = df["label"].values

def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std

X = standardize(X)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

n, num_features = X.shape
w = np.zeros(num_features)
b = 0
lr = 0.1
epochs = 1000

for epoch in range(epochs):
    z = X @ w + b
    y_hat = sigmoid(z)
    error = y_hat - y
    dw = (X.T @ error) / n
    db = np.mean(error)
    w = w - lr * dw
    b = b - lr * db

print("Training Complete")
print("Weights:")
print(w)
print("Bias:")
print(b)


# # TASK 4 – K-MEANS

# In[ ]:


import pandas as pd
import math

df = pd.read_csv("network_traffic.csv")
dataset = df.drop("label", axis=1).values.tolist()

def euclidean_distance(point1, point2):
    total = 0
    for i in range(len(point1)):
        diff = point1[i] - point2[i]
        total += diff * diff
    return math.sqrt(total)

def get_closest_centroid(point, centroids):
    min_dist = float("inf")
    closest = -1
    for i in range(len(centroids)):
        dist = euclidean_distance(point, centroids[i])
        if dist < min_dist:
            min_dist = dist
            closest = i
    return closest

def update_centroids(clusters):
    centroids = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        centroid = []
        for dim in range(len(cluster[0])):
            total = 0
            for point in cluster:
                total += point[dim]
            centroid.append(total / len(cluster))
        centroids.append(centroid)
    return centroids

k = 2
centroids = dataset[:k]

for iteration in range(10):
    clusters = [[] for _ in range(k)]
    for point in dataset:
        idx = get_closest_centroid(point, centroids)
        clusters[idx].append(point)
    new_centroids = update_centroids(clusters)
    if new_centroids == centroids:
        break
    centroids = new_centroids

print("Final Centroids:")
print(centroids)


# # TASK 5 – GENETIC ALGORITHM

# In[12]:


import random

CHROM_LEN = 15
POP_SIZE = 20
GENERATIONS = 50
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.8

def create_random_chromosome():
    return [random.randint(0,1) for _ in range(CHROM_LEN)]

def fitness(chromosome):
    return sum(chromosome)

def select_parent(population, scores):
    total = sum(scores)
    r = random.uniform(0,total)
    cumsum = 0
    for i,f in enumerate(scores):
        cumsum += f
        if r <= cumsum:
            return population[i]
    return population[-1]

def crossover(p1,p2):
    pt = random.randint(1,CHROM_LEN-1)
    c1 = p1[:pt] + p2[pt:]
    c2 = p2[:pt] + p1[pt:]
    return c1,c2

def mutate(chromosome):
    return [1-g if random.random()<MUTATION_RATE else g for g in chromosome]

population = [create_random_chromosome() for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    scores = [fitness(c) for c in population]
    best_idx = max(range(len(scores)), key=lambda i:scores[i])
    best_fit = scores[best_idx]

    print("Generation", gen, "Best =", best_fit)

    if best_fit == CHROM_LEN:
        print("Goal Reached")
        break

    new_pop = []

    while len(new_pop) < POP_SIZE:
        p1 = select_parent(population, scores)
        p2 = select_parent(population, scores)

        if random.random() < CROSSOVER_RATE:
            c1,c2 = crossover(p1,p2)
        else:
            c1 = p1[:]
            c2 = p2[:]

        c1 = mutate(c1)
        c2 = mutate(c2)

        new_pop.extend([c1,c2])

    population = new_pop[:POP_SIZE]


# In[ ]:




