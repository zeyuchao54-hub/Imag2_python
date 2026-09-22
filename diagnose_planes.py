"""诊断 scan 与对齐后 CAD 之间的平面/边对应关系。"""

import numpy as np
import open3d as o3d


def detect_planes(pcd, n_planes=6, dist_thresh=1.0, min_points=1000):
    planes = []
    remaining = pcd
    for _ in range(n_planes):
        if len(remaining.points) < min_points:
            break
        model, inliers = remaining.segment_plane(
            distance_threshold=dist_thresh, ransac_n=3, num_iterations=1000
        )
        if len(inliers) < min_points:
            break
        inlier_cloud = remaining.select_by_index(inliers)
        model = np.array(model, dtype=np.float64)
        model = model / np.linalg.norm(model[:3])
        planes.append(
            {
                "model": model,
                "normal": model[:3],
                "d": model[3],
                "center": np.asarray(inlier_cloud.points).mean(axis=0),
                "n_points": len(inliers),
                "cloud": inlier_cloud,
            }
        )
        remaining = remaining.select_by_index(inliers, invert=True)
    return planes


def plane_angle(n1, n2):
    dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return np.degrees(np.arccos(abs(dot)))


def main():
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs_dim_consistency2"
    print(f"Loading aligned CAD and scan from {output_dir}...")
    cad = o3d.io.read_point_cloud(f"{output_dir}/aligned_cad.ply")
    scan = o3d.io.read_point_cloud(f"{output_dir}/fused.ply")
    print(f"CAD: {len(cad.points)}, Scan: {len(scan.points)}")

    print("\nDetecting planes in CAD...")
    cad_planes = detect_planes(cad, n_planes=6, dist_thresh=1.0)
    print(f"Found {len(cad_planes)} CAD planes")

    print("\nDetecting planes in Scan...")
    scan_planes = detect_planes(scan, n_planes=6, dist_thresh=1.0)
    print(f"Found {len(scan_planes)} scan planes")

    print("\n=== Plane matching (Scan -> CAD) ===")
    matches = []
    for i, sp in enumerate(scan_planes):
        best_j = -1
        best_score = float("inf")
        for j, cp in enumerate(cad_planes):
            angle = plane_angle(sp["normal"], cp["normal"])
            center_dist = abs(np.dot(cp["center"] - sp["center"], cp["normal"]))
            d_diff = abs(sp["d"] - cp["d"]) if np.dot(sp["normal"], cp["normal"]) > 0 else abs(sp["d"] + cp["d"])
            score = angle + center_dist + d_diff
            if score < best_score:
                best_score = score
                best_j = j

        cp = cad_planes[best_j]
        angle = plane_angle(sp["normal"], cp["normal"])
        center_dist = abs(np.dot(cp["center"] - sp["center"], cp["normal"]))
        d_diff = abs(sp["d"] - cp["d"]) if np.dot(sp["normal"], cp["normal"]) > 0 else abs(sp["d"] + cp["d"])

        status = "POOR"
        if angle < 5 and center_dist < 2 and d_diff < 2:
            status = "GOOD"
        elif angle < 10 and center_dist < 5:
            status = "APPROX"

        status_symbol = "[OK]" if status == "GOOD" else "[~]" if status == "APPROX" else "[X]"

        matches.append((i, best_j, angle, center_dist, d_diff, status))
        print(f"Scan P{i}: pts={sp['n_points']:5d}, center=({sp['center'][0]:7.2f},{sp['center'][1]:7.2f},{sp['center'][2]:7.2f})")
        print(f"  -> CAD P{best_j}: angle={angle:5.2f} deg, center_dist={center_dist:6.3f}mm, d_diff={d_diff:6.3f}mm  {status_symbol} {status}")

    print("\n=== Summary ===")
    good = sum(1 for m in matches if m[5] == "GOOD")
    approx = sum(1 for m in matches if m[5] == "APPROX")
    poor = sum(1 for m in matches if m[5] == "POOR")
    print(f"Good matches: {good}, Approximate: {approx}, Poor: {poor}")

    print("\n=== Parallel plane pair distances (for dimensional check) ===")
    print("CAD parallel pairs:")
    for i in range(len(cad_planes)):
        for j in range(i + 1, len(cad_planes)):
            ni, nj = cad_planes[i]["normal"], cad_planes[j]["normal"]
            if np.dot(ni, nj) > 0.9:
                dist = abs(cad_planes[i]["d"] - cad_planes[j]["d"])
                print(f"  P{i}-P{j}: {dist:.3f} mm")

    print("Scan parallel pairs:")
    for i in range(len(scan_planes)):
        for j in range(i + 1, len(scan_planes)):
            ni, nj = scan_planes[i]["normal"], scan_planes[j]["normal"]
            if np.dot(ni, nj) > 0.9:
                dist = abs(scan_planes[i]["d"] - scan_planes[j]["d"])
                print(f"  P{i}-P{j}: {dist:.3f} mm")

    print("\n=== Edge correspondence (intersection lines of matched plane pairs) ===")
    # Use the good matches to compute intersection lines
    good_matches = [m for m in matches if m[5] == "GOOD"]
    if len(good_matches) >= 2:
        for idx1 in range(len(good_matches)):
            for idx2 in range(idx1 + 1, len(good_matches)):
                si1, ci1 = good_matches[idx1][0], good_matches[idx1][1]
                si2, ci2 = good_matches[idx2][0], good_matches[idx2][1]
                sp1, sp2 = scan_planes[si1], scan_planes[si2]
                cp1, cp2 = cad_planes[ci1], cad_planes[ci2]

                # Skip if planes are parallel (no intersection line)
                if abs(np.dot(sp1["normal"], sp2["normal"])) > 0.9:
                    continue
                if abs(np.dot(cp1["normal"], cp2["normal"])) > 0.9:
                    continue

                # Compute intersection line for scan
                n1, d1 = sp1["normal"], sp1["d"]
                n2, d2 = sp2["normal"], sp2["d"]
                d_scan = np.cross(n1, n2)
                d_scan = d_scan / np.linalg.norm(d_scan)
                A = np.vstack([n1, n2, d_scan])
                try:
                    p_scan = np.linalg.solve(A, [-d1, -d2, 0])
                except np.linalg.LinAlgError:
                    continue

                # Compute intersection line for CAD
                n1, d1 = cp1["normal"], cp1["d"]
                n2, d2 = cp2["normal"], cp2["d"]
                d_cad = np.cross(n1, n2)
                d_cad = d_cad / np.linalg.norm(d_cad)
                A = np.vstack([n1, n2, d_cad])
                try:
                    p_cad = np.linalg.solve(A, [-d1, -d2, 0])
                except np.linalg.LinAlgError:
                    continue

                # Distance between the two lines
                diff = p_scan - p_cad
                cross = np.cross(d_scan, d_cad)
                cross_norm = np.linalg.norm(cross)
                if cross_norm < 1e-6:
                    # Parallel lines
                    dist = np.linalg.norm(diff - np.dot(diff, d_cad) * d_cad)
                    angle = 0.0
                else:
                    dist = abs(np.dot(diff, cross / cross_norm))
                    angle = np.degrees(np.arccos(np.clip(abs(np.dot(d_scan, d_cad)), -1, 1)))

                print(f"Edge from Scan P{si1}-P{si2} / CAD P{ci1}-P{ci2}:")
                print(f"  line distance={dist:.3f} mm, direction angle={angle:.2f}°")
                if dist < 1 and angle < 2:
                    print(f"  [OK] Edge matches well")
                elif dist < 5 and angle < 5:
                    print(f"  [~] Edge approximately matches")
                else:
                    print(f"  [X] Edge does not match")
    else:
        print("Not enough good plane matches to analyze edges.")


if __name__ == "__main__":
    main()
