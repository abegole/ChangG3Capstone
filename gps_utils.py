import socket
import numpy as np
import math
import matplotlib.pyplot as plt

# === Configuration ===
PORT = 5487
NUM_NODES = 4
Z_TAG = 1.0  # Tag height in meters
M_TO_FEET = 3.28084  # Conversion factor

# === Anchor positions in BLOCK UNITS (converted to feet later) ===
anchor_blocks = [(0, 2), (2, 10), (8, 10), (8, 4)]
anchor_positions = [(by * 2, bx * 2) for bx, by in anchor_blocks]  # Transposed and converted to feet
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def trilaterate_2D(distances, anchors=anchor_positions, z_tag=Z_TAG):
    anchors = np.array([[x / M_TO_FEET, y / M_TO_FEET, 0] for x, y in anchors])
    x = anchors[:, 0]
    y = anchors[:, 1]
    z = anchors[:, 2]
    kv = x**2 + y**2

    A = np.zeros((NUM_NODES - 1, 2))
    b = np.zeros(NUM_NODES - 1)
    for i in range(1, NUM_NODES):
        A[i-1, 0] = x[i] - x[0]
        A[i-1, 1] = y[i] - y[0]
        b[i-1] = distances[0]**2 - distances[i]**2 + kv[i] - kv[0]

    try:
        Ainv = np.linalg.inv(A.T @ A) @ A.T
        pos = 0.5 * (Ainv @ b)
    except np.linalg.LinAlgError:
        print("[ERROR] Singular matrix, cannot invert.")
        return None, None

    errors = []
    for i in range(NUM_NODES):
        dx = pos[0] - x[i]
        dy = pos[1] - y[i]
        dz = z_tag - z[i]
        estimated_dist = math.sqrt(dx**2 + dy**2 + dz**2)
        errors.append((distances[i] - estimated_dist) ** 2)
    rmse = math.sqrt(sum(errors) / NUM_NODES)

    # Convert to feet
    return pos * M_TO_FEET, rmse * M_TO_FEET

def gps_initialization():
    sock.bind(('', PORT))
    print(f"[INFO] Listening for UDP packets on port {PORT}...")

