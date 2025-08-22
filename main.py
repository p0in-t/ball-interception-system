import cv2
import os, shutil
import numpy as np
import sys, pygame
import matplotlib.pyplot as plt
import random
from sklearn.cluster import DBSCAN
from utils import euclidean_norm

M = 0.5
G = 9.81
K = 0.01
PIXELS_TO_METERS = 0.01
WIDTH, HEIGHT, FPS = 0, 0, 0
RADIUS = 0

class ConvergenceException(Exception):
    pass

def pixels_to_meters(pixels):
    return pixels * PIXELS_TO_METERS

def meters_to_pixels(meters):
    return meters / PIXELS_TO_METERS

def ball_motion_ode(vx, vy, m, g, k):
    dx = vx
    dy = vy
    dvx = (-k / m) * vx * np.sqrt(vx ** 2 + vy ** 2)
    dvy = -(-g - (k / m) * vy * np.sqrt(vx ** 2 + vy ** 2))
    return dx, dy, dvx, dvy

def clear_dir(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def interpolate_edges(contours, spacing=1):
    points = []

    for i in range(len(contours) - 1):
        pt1 = contours[i][0]
        pt2 = contours[i + 1][0]

        points.append(pt1)

        dist = euclidean_norm(pt1, pt2)
        num_points = max(2, int(dist / spacing))

        for j in range(num_points):
            offset = j / (num_points - 1)
            point = pt1 + offset * (pt2 - pt1)
            points.append(point.astype(int))

        points.append(pt2)

    return np.array(points)

def get_frames(path):
    global WIDTH, HEIGHT, FPS, RADIUS

    cap = cv2.VideoCapture(path)
    count = 0
    saved_frames = []
    contour_arr = []
    radius_arr = []

    WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS))

    backSub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=False)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        fgMask = backSub.apply(frame)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)

        masked_frame = cv2.bitwise_and(frame, frame, mask=fgMask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        kernel_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        copy = np.array(frame)

        blurred = cv2.GaussianBlur(masked_frame, (9, 9), 0)
        sobelxy = cv2.Sobel(src=blurred, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

        canny = cv2.Canny(image=cv2.convertScaleAbs(sobelxy), threshold1=120, threshold2=255)

        cleaned = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel_small, iterations=3)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_mid, iterations=3)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_small, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=4)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 200
        filtered = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        cv2.drawContours(copy, filtered, -1, (0, 255, 0), 2)

        cv2.imshow('su', masked_frame)
        cv2.imshow('processed', cleaned)
        cv2.imshow('contours', copy)

        for cnt in filtered:
            area = cv2.contourArea(cnt)
            radius = np.sqrt(area / np.pi)
            radius_arr.append(radius)

        if count % 1 == 0:  # save every frame
            contour_arr.append(filtered)
            saved_frames.append(np.array(frame))

        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if radius_arr:
        RADIUS = int(np.mean(radius_arr))

    return saved_frames, contour_arr

def center_of_mass(cluster):
    return np.mean(cluster, axis=0).astype(int)

def recognize_clusters(points, labels):
    centers = []
    unique_labels = set(labels)

    for label in unique_labels:
        if label == -1:
            continue

        cluster_points = points[labels == label]
        centers.append(center_of_mass(cluster_points))

    return centers

def get_positions(path):
    global FPS

    frames, contours = get_frames(path)

    dbscan = DBSCAN(eps=14, min_samples=25)
    previous_centers = None
    positions = []
    times = []
    wait_time = 10

    for i in range(len(frames)):
        all_points = []

        for contour in contours[i]:
            interpolated = interpolate_edges(contour, spacing=2)
            all_points.extend(interpolated)

        points = np.array(all_points)

        if points.size == 0:
            cv2.putText(frames[i], f"Number of objects: {0}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 2)
            cv2.imshow('Object detection', frames[i])
            cv2.waitKey(wait_time)
            continue

        labels = dbscan.fit_predict(points)
        centers = recognize_clusters(points, labels)
        unique_labels = set(labels)
        colors = [(255, 0, 0), (0, 255, 0)]

        cv2.putText(frames[i], f"Number of objects: {len(unique_labels)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for label in unique_labels:
            if label == -1:
                cv2.imshow('Object detection', frames[i])
                cv2.waitKey(wait_time)
                continue

            color = colors[label % len(colors)]
            cluster_points = points[labels == label]

            for point in cluster_points:
                cv2.circle(frames[i], tuple(point), 2, color, -1)

        if previous_centers is not None and len(centers) == len(previous_centers):
            for j, center in enumerate(centers):
                positions.append(center)
                times.append(i / FPS)

                # speed = euclidean_norm(center, previous_centers[j])
                cv2.circle(frames[i], tuple(center), 5, (0, 255, 255), -1)
                # cv2.putText(frames[i], f"Speed: {speed:.1f}", (center[0], center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        previous_centers = centers
        cv2.imshow('Object detection', frames[i])
        cv2.waitKey(wait_time)

    cv2.destroyAllWindows()
    return np.array(positions), np.array(times)

def rk4(state, delta_time, m, g, k):
    vx, vy = state[2], state[3]
    k1x, k1y, k1vx, k1vy = ball_motion_ode(vx, vy, m, g, k)
    k2x, k2y, k2vx, k2vy = ball_motion_ode(vx + k1vx * delta_time / 2, vy + k1vy * delta_time / 2, m, g, k)
    k3x, k3y, k3vx, k3vy = ball_motion_ode(vx + k2vx * delta_time / 2, vy + k2vy * delta_time / 2, m, g, k)
    k4x, k4y, k4vx, k4vy = ball_motion_ode(vx + k3vx * delta_time, vy + k3vy * delta_time, m, g, k)

    dx = (k1x + 2 * k2x + 2 * k3x + k4x) / 6
    dy = (k1y + 2 * k2y + 2 * k3y + k4y) / 6
    dvx = (k1vx + 2 * k2vx + 2 * k3vx + k4vx) / 6
    dvy = (k1vy + 2 * k2vy + 2 * k3vy + k4vy) / 6

    return dx * delta_time, dy * delta_time, dvx * delta_time, dvy * delta_time

def forward_euler(state, delta_time, m, g, k):
    vx, vy = state[2], state[3]
    dx, dy, dvx, dvy = ball_motion_ode(vx, vy, m, g, k)

    return dx * delta_time, dy * delta_time, dvx * delta_time, dvy * delta_time

def trapezoidal(state, delta_time, m, g, k, max_iterations=200, tolerance=1e-6):
    x, y, vx, vy = state
    curr_dx, curr_dy, curr_dvx, curr_dvy = ball_motion_ode(vx, vy, m, g, k)

    next_x = x + curr_dx * delta_time
    next_y = y + curr_dy * delta_time
    next_vx = vx + curr_dvx * delta_time
    next_vy = vy + curr_dvy * delta_time

    for _ in range(max_iterations):
        prev_x, prev_y = next_x, next_y
        prev_vx, prev_vy = next_vx, next_vy

        next_dx, next_dy, next_dvx, next_dvy = ball_motion_ode(next_vx, next_vy, m, g, k)

        next_x = x + 0.5 * delta_time * (curr_dx + next_dx)
        next_y = y + 0.5 * delta_time * (curr_dy + next_dy)
        next_vx = vx + 0.5 * delta_time * (curr_dvx + next_dvx)
        next_vy = vy + 0.5 * delta_time * (curr_dvy + next_dvy)

        x_diff = abs(next_x - prev_x)
        y_diff = abs(next_y - prev_y)
        vx_diff = abs(next_vx - prev_vx)
        vy_diff = abs(next_vy - prev_vy)

        if x_diff < tolerance and y_diff < tolerance and vx_diff < tolerance and vy_diff < tolerance:
            break

    dx = next_x - x
    dy = next_y - y
    dvx = next_vx - vx
    dvy = next_vy - vy

    return dx, dy, dvx, dvy

def normalize_pos(pos, r, w, h):
    if pos[0] <= r:
        pos[0] = r
    elif pos[0] > w-r:
        pos[0] = w-r

    if pos[1] <= r:
        pass
        # pos[1] = r
    elif pos[1] > h-r:
        pos[1] = h-r

def residual(simulated_pos, simulated_target_pos):
    simulated_x, simulated_y = simulated_pos
    simulated_target_x, simulated_target_y = simulated_target_pos

    simulated_end = np.array([simulated_x[-1], simulated_y[-1]])
    simulated_target_end = np.array([simulated_target_x[-1], simulated_target_y[-1]])

    residuals = np.zeros(2)
    residuals[0] = simulated_end[0] - simulated_target_end[0]
    residuals[1] = simulated_end[1] - simulated_target_end[1]

    return residuals

def residual_fit(simulated_pos, observed_pos):
    simulated_x, simulated_y = simulated_pos
    residuals = np.zeros(2 * len(observed_pos))

    residuals[0::2] = simulated_x - observed_pos[:, 0]
    residuals[1::2] = simulated_y - observed_pos[:, 1]

    return residuals

def estimate_max_iterations(window_size, velocity_bounds, delta_time):
    width_m, height_m = window_size

    max_vx = max(abs(velocity_bounds[0][0]), abs(velocity_bounds[0][1]))
    max_vy = max(abs(velocity_bounds[1][0]), abs(velocity_bounds[1][1]))

    diagonal_distance = np.sqrt(width_m ** 2 + height_m ** 2)
    max_velocity = np.sqrt(max_vx ** 2 + max_vy ** 2)

    min_time = (diagonal_distance / max_velocity) * 1.5

    max_iters = int(min_time / delta_time)

    return max_iters

def simulate_trajectory(initial_state, params, delta_time, window_size, velocity_bounds):
    M, G, K = params

    x = [initial_state[0]]
    y = [initial_state[1]]
    vx = initial_state[2]
    vy = initial_state[3]

    max_iters = estimate_max_iterations(window_size, velocity_bounds, delta_time)

    for i in range(max_iters // 2, max_iters):
        dx, dy, dvx, dvy = trapezoidal([x[-1], y[-1], vx, vy], delta_time, M, G, K)

        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
        vx += dvx
        vy += dvy

    return np.array(x), np.array(y)

def simulate_trajectory_fit(initial_state, params, times):
    M, G, K = params

    x = [initial_state[0]]
    y = [initial_state[1]]
    vx = initial_state[2]
    vy = initial_state[3]

    for i in range(1, len(times)):
        delta_time = times[i] - times[i - 1]
        dx, dy, dvx, dvy = trapezoidal([x[-1], y[-1], vx, vy], delta_time, M, G, K)

        x.append(x[-1] + dx)
        y.append(y[-1] + dy)
        vx += dvx
        vy += dvy

    return np.array(x), np.array(y), vx, vy

def numerical_jacobian(func, x, epsilon=1e-8):
    f = np.array(func(x))

    n = len(x)
    m = len(f)
    J = np.zeros((m, n))

    for i in range(n):
        h = np.zeros(n)
        h[i] = epsilon

        f_plus = func(x + h)
        f_minus = func(x - h)
        J[:, i] = (f_plus - f_minus) / (2 * epsilon)

    return f, J

def newton_shooting_method(position, initial_guess, params, mass, target_pos, target_velocity, target_mass, times, max_iters=10, tol=1e-8):
    velocity = initial_guess.copy()
    l0, l1 = position
    x0, y0 = target_pos
    vx0, vy0 = target_velocity
    G, K = params

    target_params = target_mass, G, K
    local_params = mass, G, K

    bounds = np.array([
        [-20.0, 20.0],
        [-20.0, 20.0]
    ])

    # positions = np.array([start_pos, target_pos])

    for iter in range(max_iters):
        def res(p):
            vx0_test, vy0_test = p
            target_state = np.array([x0, y0, vx0, vy0])
            local_state = np.array([l0, l1, vx0_test, vy0_test])
            simulated_target_x, simulated_target_y, _, _ = simulate_trajectory_fit(target_state, target_params, times)
            simulated_local_x, simulated_local_y, _, _ = simulate_trajectory_fit(local_state, local_params, times)

            return residual((simulated_local_x, simulated_local_y), (simulated_target_x, simulated_target_y))

        error, J = numerical_jacobian(res, velocity)

        lambda_reg = 1e-6
        delta = np.linalg.solve(J.T @ J + lambda_reg * np.eye(len(params)), -J.T @ error)

        new_velocity = velocity + delta
        new_velocity = np.clip(new_velocity, bounds[:, 0], bounds[:, 1])

        residual_norm = euclidean_norm(error, np.zeros(error.shape))

        if residual_norm < tol:
            return velocity, error

        velocity = new_velocity

    raise ConvergenceException("Newton's method did not converge!")

def newton_shooting_method_fit(initial_params, positions, times, max_iters=10, tol=1e-8):
    x0, y0 = positions[0]
    vx_end, vy_end = 0, 0
    params = initial_params.copy()

    bounds = np.array([
        [-20.0, 20.0],
        [-20.0, 20.0],
        [0.1, 10.0],
        [0, G*2],
        [1e-7, 1.0]
    ])

    for _ in range(max_iters):
        def res(p):
            nonlocal vx_end, vy_end
            vx0_test, vy0_test, m_test, g_test, k_test = p
            initial_state = np.array([x0, y0, vx0_test, vy0_test])
            simulated_x, simulated_y, vx_end, vy_end = simulate_trajectory_fit(initial_state, [m_test, g_test, k_test], times)
            return residual_fit((simulated_x, simulated_y), positions)

        error, J = numerical_jacobian(res, params)

        lambda_reg = 1e-6
        delta = np.linalg.solve(J.T @ J + lambda_reg * np.eye(len(params)), -J.T @ error)

        new_params = params + delta
        new_params = np.clip(new_params, bounds[:, 0], bounds[:, 1])

        residual_norm = euclidean_norm(error, np.zeros(error.shape))

        # print(residual_norm)

        if residual_norm < tol:
            return params, vx_end, vy_end

        params = new_params

    raise ConvergenceException("Newton's method did not converge!")

def analyze_sample(positions, times, analysis_ratio):
    global G, K

    cut_idx = int(len(positions) * analysis_ratio)
    positions = np.array([[pixels_to_meters(x[0]), pixels_to_meters(x[1])] for x in positions])
    positions_sample = positions[:cut_idx]
    positions_rest = positions[cut_idx:]
    times_sample = times[:cut_idx]
    times_rest = times[cut_idx:]

    initial_params = np.array([
        0.0,
        0.0,
        0.5,
        G,
        K
    ])

    optimal_params, VXN, VYN = newton_shooting_method_fit(initial_params, positions_sample, times_sample, tol=1)
    VX0, VY0, M, G, K = optimal_params

    print(f"Optimized parameters:")
    print(f"Initial X velocity: {VX0:.3f}")
    print(f"Initial Y velocity: {VY0:.3f}")
    print(f"Mass: {M:.3f}")
    print(f"Gravity: {G:.3f}")
    print(f"Drag coefficient: {K:.6f}")

    # simulated_x, simulated_y, _, _ = simulate_trajectory_fit(initial_state, [M, G, K], times_rest)
    #
    # mse_x = np.mean((simulated_x - positions_rest[:, 0]) ** 2)
    # mse_y = np.mean((simulated_y - positions_rest[:, 1]) ** 2)
    # total_mse = mse_x + mse_y
    #
    # print(f"\nError metrics:")
    # print(f"MSE X: {mse_x:.4f}")
    # print(f"MSE Y: {mse_y:.4f}")
    # print(f"Total MSE: {total_mse:.4f}\n-----------------------------\n")
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(positions[:, 0], positions[:, 1], 'b.', label='Observed')
    # plt.plot(simulated_x, simulated_y, 'r-', label='Simulated')
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return [M, G, K], positions_rest[0], [VXN, VYN], times_rest

def analyze_sample_alt(positions, times):
    global G, K

    average_step = np.diff(times).mean()
    times_extended = np.arange(times[0], times[-1] + average_step * len(times), average_step)
    positions = np.array([[pixels_to_meters(x[0]), pixels_to_meters(x[1])] for x in positions])
    initial_params = np.array([
        0.0,
        0.0,
        0.5,
        G,
        K
    ])

    optimal_params, VXN, VYN = newton_shooting_method_fit(initial_params, positions, times, tol=1)
    VX0, VY0, M, G, K = optimal_params

    print(f"Optimized parameters:")
    print(f"Initial X velocity: {VX0:.3f}")
    print(f"Initial Y velocity: {VY0:.3f}")
    print(f"Mass: {M:.3f}")
    print(f"Gravity: {G:.3f}")
    print(f"Drag coefficient: {K:.6f}")

    # simulated_x, simulated_y, _, _ = simulate_trajectory_fit(initial_state, [M, G, K], times_extended)
    # simulated_positions = np.column_stack((simulated_x, simulated_y))

    # simulated_x, simulated_y, _, _ = simulate_trajectory_fit(initial_state, [M, G, K], times_rest)
    #
    # mse_x = np.mean((simulated_x - positions_rest[:, 0]) ** 2)
    # mse_y = np.mean((simulated_y - positions_rest[:, 1]) ** 2)
    # total_mse = mse_x + mse_y
    #
    # print(f"\nError metrics:")
    # print(f"MSE X: {mse_x:.4f}")
    # print(f"MSE Y: {mse_y:.4f}")
    # print(f"Total MSE: {total_mse:.4f}\n-----------------------------\n")
    #
    # plt.figure(figsize=(10, 6))
    # plt.plot(positions[:, 0], positions[:, 1], 'b.', label='Observed')
    # plt.plot(simulated_x, simulated_y, 'r-', label='Simulated')
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return [M, G, K], positions[0], [VX0, VY0], times, times_extended

def find_velocity(position, params, mass, target_pos, target_velocity, target_mass, times):
    initial_velocity = np.array([
        0.0,
        0.0
    ])

    for i in range(int(len(times) * 0), len(times)):
        try:
            velocity, error = newton_shooting_method(position, initial_velocity, params, mass, target_pos, target_velocity, target_mass, times[:i])
            print(f"Found optimal velocity: {velocity[0]:.4f}, {velocity[1]:.4f}")
            print(f"Estimated error: {np.mean(error**2):.4f}")
            return velocity
        except ConvergenceException as e:
            continue

    raise ConvergenceException("Newton's shooting method could not converge throughout the time period!")

def find_velocity_alt(position, params, mass, target_pos, target_velocity, target_mass, times, times_ext):
    initial_velocity = np.array([
        0.0,
        0.0
    ])

    hit_time_mult = (len(times_ext) - len(times)) * 0.0 # modify to hit ball later

    for i in range(int((len(times)) + hit_time_mult), len(times_ext)):
        try:
            velocity, error = newton_shooting_method(position, initial_velocity, params, mass, target_pos, target_velocity, target_mass, times_ext[:i])
            print(f"Found optimal velocity: {velocity[0]:.4f}, {velocity[1]:.4f}")
            print(f"Estimated error: {np.mean(error**2):.4f}")
            return velocity
        except ConvergenceException as e:
            continue

    raise ConvergenceException("Newton's shooting method could not converge throughout the time period!")

def circle_collision(pos1, pos2, vel1, vel2, radius, damping):
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    distance = np.sqrt(dx * dx + dy * dy)

    if distance < radius * 2:
        overlap = (radius * 2 - distance) / 2
        nx = dx / distance
        ny = dy / distance

        pos1[0] += nx * overlap
        pos1[1] += ny * overlap
        pos2[0] -= nx * overlap
        pos2[1] -= ny * overlap

        temp_vel = vel1.copy()
        vel1_new = vel2 * damping
        vel2_new = temp_vel * damping

        return vel1_new, vel2_new

    return vel1, vel2

def ball_sim(positions, params, masses, fixed_params):
    global WIDTH, HEIGHT, RADIUS

    size = width, height = WIDTH, HEIGHT
    G, K = fixed_params

    pygame.init()
    pygame.font.init()
    my_font = pygame.font.SysFont('Tahoma', 30, True)
    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()

    p_positions = np.array([[meters_to_pixels(x[0]), meters_to_pixels(x[1])] for x in positions])
    p_params = np.array([[meters_to_pixels(x[0]), meters_to_pixels(x[1])] for x in params])


    circle_pos = np.array([pos for pos in p_positions], dtype=np.float64)
    circle_speed = np.array([param for param in p_params], dtype=np.float64)
    circle_radius = RADIUS
    circle_color = (233, 22, 27)
    black = 255, 255, 255
    damping = 0.7

    m_positions = np.array([[pixels_to_meters(x[0]), pixels_to_meters(x[1])] for x in circle_pos])
    m_velocities = np.array([[pixels_to_meters(x[0]), pixels_to_meters(x[1])] for x in circle_speed])
    trajectories = []

    for i in range(len(circle_pos)):
        initial_state = [m_positions[i][0], m_positions[i][1], m_velocities[i][0], m_velocities[i][1]]
        traj_x, traj_y = simulate_trajectory(initial_state, [masses[i], G, K], 1 / 60,
                                             [pixels_to_meters(WIDTH), pixels_to_meters(HEIGHT)],
                                             [[-10, 10], [-10, 10]])
        pixel_trajectory = np.column_stack((meters_to_pixels(traj_x), meters_to_pixels(traj_y)))
        trajectories.append(pixel_trajectory)

    debug = False
    reset = False

    while True:
        delta_time = clock.tick(120) / 1000
        fps = clock.get_fps()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    debug = not debug

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    reset = not reset

        if reset:
            circle_pos = np.array([pos for pos in p_positions], dtype=np.float64)
            circle_speed = np.array([param for param in p_params], dtype=np.float64)
            reset = False

            continue

        for i in range(len(circle_pos)):
            tx, ty = np.array([pixels_to_meters(circle_pos[i][0]), pixels_to_meters(circle_pos[i][1])])
            tvx, tvy = np.array([pixels_to_meters(circle_speed[i][0]), pixels_to_meters(circle_speed[i][1])])

            dx, dy, dvx, dvy = trapezoidal([tx, ty, tvx, tvy], delta_time, masses[i], G, K)
            circle_pos[i][0] += meters_to_pixels(dx)
            circle_pos[i][1] += meters_to_pixels(dy)
            circle_speed[i][0] += meters_to_pixels(dvx)
            circle_speed[i][1] += meters_to_pixels(dvy)

            circle_bottom = circle_pos[i][1] + circle_radius
            circle_left = circle_pos[i][0] - circle_radius
            circle_right = circle_pos[i][0] + circle_radius
            circle_top = circle_pos[i][1] - circle_radius

            if circle_bottom >= height:
                circle_speed[i][1] = -circle_speed[i][1] * damping
                circle_pos[i][1] = height - circle_radius

            if circle_left < 0 or circle_right > width:
                circle_speed[i][0] = -circle_speed[i][0] * damping

            # if circle_top < 0 or circle_top > height:
            #     circle_speed[i][1] = -circle_speed[i][1] * damping

            if circle_top > height:
                circle_speed[i][1] = -circle_speed[i][1] * damping

            normalize_pos(circle_pos[i], circle_radius, width, height)

        for i in range(len(circle_pos)):
            for j in range(i + 1, len(circle_pos)):
                circle_speed[i], circle_speed[j] = circle_collision(circle_pos[i], circle_pos[j], circle_speed[i], circle_speed[j], circle_radius, damping)

        fps_surface = my_font.render(f'FPS : {fps}', True, (0, 0, 0))

        screen.fill(black)

        for trajectory in trajectories:
            if len(trajectory) > 1:
                pygame.draw.lines(screen, (150, 150, 150), False, trajectory, 1)

        for i in range(len(circle_pos)):
            pygame.draw.circle(screen, circle_color, circle_pos[i], circle_radius)

        if debug:
            screen.blit(fps_surface, (0, 0))

        pygame.display.flip()

def shoot(positions, times, local_pos, mass):
    optimal_params, target_pos, target_velocity, times_rest = analyze_sample(positions, times, 1 / 3)
    target_mass, G, K = optimal_params

    velocity = find_velocity(local_pos, [G, K], mass, target_pos, target_velocity, target_mass, times_rest)

    positions = [local_pos, target_pos]
    params = [velocity, target_velocity]
    masses = [mass, target_mass]

    return positions, params, masses, [G, K]

def shoot_alt(positions, times, local_pos, mass):
    optimal_params, target_pos, target_velocity, times, times_ext = analyze_sample_alt(positions, times)
    target_mass, G, K = optimal_params

    velocity = find_velocity_alt(local_pos, [G, K], mass, target_pos, target_velocity, target_mass, times, times_ext)

    positions = [local_pos, target_pos]
    params = [velocity, target_velocity]
    masses = [mass, target_mass]

    return positions, params, masses, [G, K]

def run_sim(path):
    global WIDTH, HEIGHT, RADIUS

    mass = 0.5
    positions, times = get_positions(path)

    while True:
        local_pos = [pixels_to_meters(random.randint(30 + RADIUS, WIDTH - 30 - RADIUS)),
                     pixels_to_meters(random.randint(30 + RADIUS, HEIGHT - 30 - RADIUS))]
        sim_positions, sim_params, masses, [G, K] = shoot(positions, times, local_pos, mass)
        ball_sim(sim_positions, sim_params, masses, [G, K])


def compare_methods_on_video(video_path):
    positions, times = get_positions(video_path)
    positions_m = np.array([[pixels_to_meters(x[0]), pixels_to_meters(x[1])] for x in positions])

    [M, G, K], positions[0], [VX0, VY0], times, times_extended = analyze_sample_alt(positions, times)

    initial_state = [positions_m[0][0], positions_m[0][1], VX0, VY0]

    methods = {
        'RK4': {'x': [], 'y': []},
        'Euler': {'x': [], 'y': []},
        'Trapezoidal': {'x': [], 'y': []}
    }

    for method_name in methods:
        x, y = initial_state[0], initial_state[1]
        vx, vy = initial_state[2], initial_state[3]

        methods[method_name]['x'].append(x)
        methods[method_name]['y'].append(y)

        for i in range(1, len(times)):
            delta_time = times[i] - times[i - 1]

            if method_name == 'RK4':
                dx, dy, dvx, dvy = rk4([x, y, vx, vy], delta_time, M, G, K)
            elif method_name == 'Euler':
                dx, dy, dvx, dvy = forward_euler([x, y, vx, vy], delta_time, M, G, K)
            else:
                dx, dy, dvx, dvy = trapezoidal([x, y, vx, vy], delta_time, M, G, K)

            x += dx
            y += dy
            vx += dvx
            vy += dvy

            methods[method_name]['x'].append(x)
            methods[method_name]['y'].append(y)

    print("\nErrors compared to actual trajectory:")
    for method_name, data in methods.items():
        method_positions = np.column_stack((data['x'], data['y']))
        mse = np.mean(np.sum((method_positions - positions_m) ** 2, axis=1))
        max_error = np.max(np.sqrt(np.sum((method_positions - positions_m) ** 2, axis=1)))

        print(f"\n{method_name}:")
        print(f"Mean Square Error: {mse:.6f} m²")
        print(f"Maximum Error: {max_error:.6f} m")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(positions_m[:, 0], positions_m[:, 1], 'k.', label='Actual', alpha=0.5)

    for method_name, data in methods.items():
        plt.plot(data['x'], data['y'], '-', label=method_name, alpha=0.7)

    plt.title('Trajectories')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)

    for method_name, data in methods.items():
        method_positions = np.column_stack((data['x'], data['y']))
        errors = np.sqrt(np.sum((method_positions - positions_m) ** 2, axis=1))
        plt.plot(times, errors, label=f'{method_name} Error', alpha=0.7)

    plt.title('Position Error Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    # compare_methods_on_video("./data/cut2.mp4")
    run_sim("./data/cut2.mp4")

if __name__ == '__main__':
    main()