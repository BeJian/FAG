import heapq
import warnings
from sklearn.manifold import MDS
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')


def data_normalize(rawdata):
    rdata = rawdata
    scaler = MinMaxScaler()
    nor_data = pd.DataFrame(scaler.fit_transform(rdata), columns=rdata.columns)
    return nor_data


def read_raw_data(dataset='RP1043_all'):
    data_path = str(dataset) + '.csv'
    print(data_path)
    if not os.path.exists(data_path):
        print(f"file is not exist: {data_path}")
        return None
    Data = pd.read_csv(data_path)
    X = Data.iloc[:, Data.columns != "Y"]
    Y = Data.iloc[:, Data.columns == "Y"]
    return X, Y


def feature_association_graph(datax, datay, dataset):
    stime = time.time()
    data = datax  # 输入数据
    print(type(data), data.shape)
    labels = datay  # 输入标签

    data = data.loc[:, data.std().round(10) != 0]
    corr = data.corr()
    dist = 1 - corr.abs()

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=11)
    embedding = mds.fit_transform(dist)
    normalized_data = data_normalize(data)
    embedding_min, embedding_max = embedding.min(axis=0), embedding.max(axis=0)
    embedding = (embedding - embedding_min) / (embedding_max - embedding_min)

    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def find_nearest_available(x, y, used_coordinates, grid_size):
        if (x, y) not in used_coordinates:
            return x, y

        radius = 1
        max_radius = min(grid_size // 2, grid_size - max(x, y))
        while radius <= max_radius:
            neighbors = [(manhattan_distance((x, y), (x + dx, y + dy)), x + dx, y + dy)
                         for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
                         if (dx, dy) != (0, 0) and 0 <= x + dx < grid_size and 0 <= y + dy < grid_size
                         and (x + dx, y + dy) not in used_coordinates]

            if neighbors:
                heapq.heapify(neighbors)
                _, cx, cy = heapq.heappop(neighbors)
                return cx, cy
            radius += 1
        return None

    max_attempts = 100
    initial_grid_size = int(np.ceil(np.sqrt(data.shape[1])))
    print('initial_grid_size:', initial_grid_size)
    grid_size = initial_grid_size
    coordinates = (embedding * (grid_size - 1)).astype(int)
    success = False

    for attempt in range(max_attempts):
        grid_size = initial_grid_size + attempt
        coordinates = (embedding * (grid_size - 1)).astype(int)
        used_coordinates = set()
        conflict = False

        for i, (x, y) in enumerate(coordinates):
            if (x, y) in used_coordinates:
                result = find_nearest_available(x, y, used_coordinates, grid_size)
                if result is None:
                    conflict = True
                    break
                x, y = result
            used_coordinates.add((x, y))
            coordinates[i] = [x, y]

        if not conflict:
            success = True
            break

    if not success:
        raise ValueError("No available grid_size")
    print('final grid_size:', grid_size)

    output_dir = 'dataset/FAP_images_'+str(dataset)
    os.makedirs(output_dir, exist_ok=True)

    # save image
    for idx in range(normalized_data.shape[0]):

        percent = (idx + 1) * 100 // normalized_data.shape[0]
        print('\r', end='')
        print(f'Progress：{percent}%', end='')

        label = labels.iloc[idx, 0]
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        image = np.full((grid_size, grid_size), np.nan)

        for i, (x, y) in enumerate(coordinates):
            image[x, y] = normalized_data.iloc[idx, i]

        fig, ax = plt.subplots(figsize=(grid_size, grid_size))

        for i, (x, y) in enumerate(coordinates):
            value = normalized_data.iloc[idx, i]
            rect = plt.Rectangle((y, x), 1, 1, color=plt.cm.Greys(value))
            ax.add_patch(rect)

        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        plt.axis('off')

        file_path = os.path.join(label_dir, f'{label}_{idx}.png')
        plt.savefig(file_path, dpi=25, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    etime = time.time()
    all_time = etime - stime
    print('\n time:', all_time)
    print("Done!")


X, Y = read_raw_data('dataset/AHU_all')
feature_association_graph(X, Y, 'AHU_111')
