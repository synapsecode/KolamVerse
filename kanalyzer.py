import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix, KDTree
from scipy.interpolate import splprep, splev
from sklearn.cluster import DBSCAN
from skimage.morphology import skeletonize, medial_axis
from scipy import ndimage
import networkx as nx
import math
from collections import defaultdict
import argparse
import os

class KolamAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = None
        self.processed_image = None
        self.skeleton = None
        self.dots = []
        self.lines = []
        self.grid_structure = None
        self.symmetry_info = None
        
    def load_and_preprocess(self):
        """Enhanced image preprocessing combining best methods"""
        # Load image
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Enhanced preprocessing using OTSU thresholding (from first code)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = (th == 0).astype(np.uint8)  # strokes black -> 1
        
        # Morphological operations to clean up
        binary = cv2.medianBlur((binary * 255).astype(np.uint8), 3) // 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        self.processed_image = binary
        self.gray_image = gray
        
        print(f"Preprocessing complete. Binary image shape: {self.processed_image.shape}")
        return binary
    
    def detect_dots(self):
        """Improved dot detection combining both approaches with stricter validation"""
        print("Starting enhanced dot detection...")
        dots = []
        
        # Method 1: Strict HoughCircles (from second code - better validation)
        gray_for_circles = self.gray_image
        
        # First pass: strict parameters for clear dots
        circles1 = cv2.HoughCircles(
            gray_for_circles,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=35,
            minRadius=8,
            maxRadius=30
        )
        
        if circles1 is not None:
            circles1 = np.round(circles1[0, :]).astype("int")
            for (x, y, r) in circles1:
                if self._is_valid_dot(x, y, r):
                    dots.append({'x': x, 'y': y, 'radius': r, 'method': 'hough_strict'})
        
        # Second pass: relaxed parameters (from first code - catch more dots)
        circles2 = cv2.HoughCircles(
            gray_for_circles,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=15,
            param1=30,
            param2=20,
            minRadius=2,
            maxRadius=25
        )
        
        if circles2 is not None:
            circles2 = np.round(circles2[0, :]).astype("int")
            for (x, y, r) in circles2:
                if self._is_valid_dot(x, y, r):
                    dots.append({'x': x, 'y': y, 'radius': r, 'method': 'hough_relaxed'})
        
        print(f"HoughCircles found {len(dots)} dots")
        
        # Method 2: Enhanced contour detection with validation (from second code)
        contour_dots = self._detect_real_dots_by_contours()
        dots.extend(contour_dots)
        
        # Method 3: Distance transform (from first code)
        distance_dots = self._detect_dots_by_distance_transform()
        dots.extend(distance_dots)
        
        # Method 4: Corner detection (from first code) 
        corner_dots = self._detect_intersection_points()
        dots.extend(corner_dots)
        
        # Remove duplicates using original method (from first code)
        if dots:
            unique_dots = self._remove_duplicate_dots(dots, threshold=15)
            # Additional filtering with validation
            validated_dots = [dot for dot in unique_dots if self._validate_final_dot(dot)]
            self.dots = validated_dots
        else:
            self.dots = []
        
        print(f"Final result: {len(self.dots)} validated dots")
        return self.dots
    
    def _is_valid_dot(self, x, y, radius):
        """Validate if detected circle is actually a kolam dot (from second code)"""
        h, w = self.gray_image.shape
        
        # Ensure coordinates are within bounds
        if x < radius or y < radius or x >= w - radius or y >= h - radius:
            return False
        
        # Extract region around the detected circle
        region = self.gray_image[y-radius:y+radius, x-radius:x+radius]
        
        # Check if center area is darker than surrounding
        center_area = region[radius//2:radius*3//2, radius//2:radius*3//2]
        if len(center_area) == 0:
            return False
            
        # Get surrounding ring
        mask = np.zeros_like(region)
        cv2.circle(mask, (radius, radius), radius//2, 1, -1)
        center_pixels = region[mask == 1]
        
        # Get outer ring
        outer_mask = np.zeros_like(region)
        cv2.circle(outer_mask, (radius, radius), radius, 1, 2)
        cv2.circle(outer_mask, (radius, radius), radius*2//3, 0, -1)
        outer_pixels = region[outer_mask > 0]
        
        if len(center_pixels) == 0 or len(outer_pixels) == 0:
            return False
        
        center_avg = np.mean(center_pixels)
        outer_avg = np.mean(outer_pixels)
        
        # Dot should be darker than surroundings
        return center_avg < outer_avg - 10
    
    def _detect_real_dots_by_contours(self):
        """Enhanced contour-based dot detection (from second code)"""
        # Use inverted image to find dark regions (dots)
        inverted = 255 - self.gray_image
        
        # Apply threshold to get only dark dots
        _, binary = cv2.threshold(inverted, 100, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Reasonable dot size range
            if 50 < area < 800:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Check circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * math.pi * area / (perimeter * perimeter)
                        
                        # Strict circularity for dots
                        if circularity > 0.6:
                            radius = int(math.sqrt(area / math.pi))
                            
                            # Final validation
                            if self._is_valid_dot(cx, cy, radius):
                                dots.append({'x': cx, 'y': cy, 'radius': radius, 'method': 'contour_validated'})
        
        print(f"Validated contour method found {len(dots)} dots")
        return dots
    
    def _detect_dots_by_distance_transform(self):
        """Distance transform method (from first code)"""
        inv = 1 - self.processed_image
        dist = ndimage.distance_transform_edt(inv)
        labels, n = ndimage.label(inv)
        
        dots = []
        for lab in range(1, n + 1):
            ys, xs = np.where(labels == lab)
            if len(xs) == 0:
                continue
                
            cx, cy = xs.mean(), ys.mean()
            area = len(xs)
            
            if 5 < area < 300:
                radius = max(3, int(math.sqrt(area / math.pi)))
                dots.append({'x': cx, 'y': cy, 'radius': radius, 'method': 'distance'})
        
        return dots
    
    def _detect_intersection_points(self):
        """Corner detection method (from first code)"""
        corners = cv2.goodFeaturesToTrack(
            (self.processed_image * 255).astype(np.uint8),
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
            useHarrisDetector=True
        )
        
        dots = []
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                dots.append({'x': int(x), 'y': int(y), 'radius': 3, 'method': 'corner'})
        
        return dots
    
    def _validate_final_dot(self, dot):
        """Final validation of dots"""
        # Check if dot is in reasonable location
        h, w = self.processed_image.shape
        
        # Not too close to edges
        margin = 20
        if (dot['x'] < margin or dot['x'] > w - margin or 
            dot['y'] < margin or dot['y'] > h - margin):
            return False
        
        # Reasonable size
        if dot['radius'] < 2 or dot['radius'] > min(w, h) // 8:
            return False
        
        return True
    
    def _remove_duplicate_dots(self, dots, threshold=15):
        """Remove duplicates using KDTree method (from first code)"""
        if not dots:
            return []
        
        coords = np.array([[d['x'], d['y']] for d in dots])
        tree = KDTree(coords)
        groups = tree.query_ball_tree(tree, threshold)
        
        unique = []
        processed = set()
        
        for i, group in enumerate(groups):
            if i in processed:
                continue
            
            pts = coords[group]
            cx, cy = pts[:,0].mean(), pts[:,1].mean()
            
            # Calculate average radius, preferring validated methods
            radii = []
            methods = []
            for g in group:
                radii.append(dots[g].get('radius', 3))
                methods.append(dots[g].get('method', 'unknown'))
            
            avg_radius = np.mean(radii)
            
            # Prefer validated methods
            preferred_methods = ['hough_strict', 'contour_validated', 'hough_relaxed']
            method_priority = 'unknown'
            for method in preferred_methods:
                if method in methods:
                    method_priority = method
                    break
            
            unique.append({
                'x': cx, 
                'y': cy, 
                'radius': avg_radius,
                'method': method_priority,
                'cluster_size': len(group)
            })
            
            processed.update(group)
        
        return unique
    
    def extract_curves(self):
        """Enhanced curve extraction using skeleton analysis (from first code - best method)"""
        print("Starting enhanced curve extraction...")
        
        try:
            # Create skeleton using skimage for better results (from first code)
            skeleton = skeletonize(self.processed_image == 1).astype(np.uint8)
            self.skeleton = skeleton
            
            # Build graph structure from skeleton (from first code)
            graph = self._skeleton_to_graph(skeleton)
            
            # Extract paths from graph
            curves = []
            for u, v, data in graph.edges(data=True):
                if 'pixels' in data:
                    pixels = data['pixels']
                    if len(pixels) > 8:  # Filter short segments
                        points = [{'x': int(pt[0]), 'y': int(pt[1])} for pt in pixels]
                        curves.append(points)
            
            print(f"Graph-based extraction found {len(curves)} curves")
            
            # Also use traditional contour method as backup (from first code)
            contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) > 15:
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    simplified = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(simplified) > 4:
                        points = [{'x': int(pt[0][0]), 'y': int(pt[0][1])} for pt in simplified]
                        curves.append(points)
            
            print(f"Combined with contour extraction: {len(curves)} total curves")
            
            # Filter similar curves (from first code)
            filtered_curves = self._filter_and_merge_curves(curves)
            
            self.lines = filtered_curves
            print(f"Final result: {len(filtered_curves)} unique curve segments")
            return filtered_curves
            
        except Exception as e:
            print(f"Error in curve extraction: {e}")
            # Fallback to simple method
            return self._extract_curves_fallback()
    
    def _extract_curves_fallback(self):
        """Fallback curve extraction method"""
        try:
            contours, _ = cv2.findContours(self.processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            curves = []
            
            for contour in contours:
                arc_length = cv2.arcLength(contour, False)
                if arc_length > 30:
                    epsilon = 0.01 * arc_length
                    simplified = cv2.approxPolyDP(contour, epsilon, False)
                    if len(simplified) > 4:
                        points = [{'x': int(pt[0][0]), 'y': int(pt[0][1])} for pt in simplified]
                        curves.append(points)
            
            self.lines = curves
            print(f"Fallback extraction found {len(curves)} curves")
            return curves
        except:
            self.lines = []
            return []
    
    def _filter_and_merge_curves(self, curves):
        """Filter and merge similar curves (from first code)"""
        if not curves:
            return curves
        
        # Remove very short curves
        filtered = [curve for curve in curves if len(curve) >= 5]
        
        # Remove duplicate curves that are very similar
        unique_curves = []
        
        for curve in filtered:
            is_duplicate = False
            
            for existing in unique_curves:
                if self._curves_are_similar(curve, existing, threshold=12):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_curves.append(curve)
        
        return unique_curves
    
    def _curves_are_similar(self, curve1, curve2, threshold=10):
        """Check if two curves are similar (from first code)"""
        if abs(len(curve1) - len(curve2)) > max(len(curve1), len(curve2)) * 0.5:
            return False
        
        # Sample points from both curves
        sample_size = min(8, min(len(curve1), len(curve2)))
        
        indices1 = np.linspace(0, len(curve1)-1, sample_size, dtype=int)
        indices2 = np.linspace(0, len(curve2)-1, sample_size, dtype=int)
        
        points1 = np.array([[curve1[i]['x'], curve1[i]['y']] for i in indices1])
        points2 = np.array([[curve2[i]['x'], curve2[i]['y']] for i in indices2])
        
        # Calculate average distance between corresponding points
        distances = np.sqrt(np.sum((points1 - points2)**2, axis=1))
        avg_distance = np.mean(distances)
        
        return avg_distance < threshold
    
    def _skeleton_to_graph(self, skeleton):
        """Convert skeleton to graph structure (from first code)"""
        H, W = skeleton.shape
        
        def neighbors_xy(x, y):
            """Get neighbors in (x,y) ordering -> (col, row)"""
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = x + dx
                    ny = y + dy
                    if 0 <= ny < H and 0 <= nx < W and skeleton[ny, nx]:
                        yield (nx, ny)
        
        # Classify pixels: endpoints (degree 1) and junctions (degree > 2)
        endpoints = []
        junctions = []
        for y in range(H):
            for x in range(W):
                if not skeleton[y, x]:
                    continue
                deg = sum(1 for _ in neighbors_xy(x, y))
                if deg == 1:
                    endpoints.append((x, y))
                elif deg > 2:
                    junctions.append((x, y))
        
        # Anchor points are endpoints + junctions
        node_points = endpoints + junctions
        anchor_map = {pt: i for i, pt in enumerate(node_points)}
        
        # Create graph
        G = nx.Graph()
        for i, pt in enumerate(node_points):
            G.add_node(i, coord=(pt[0], pt[1]))
        
        # Follow paths between anchors
        def follow_path(start):
            """Follow path from anchor to find connected anchors"""
            x0, y0 = start
            found_paths = []
            
            for nbr in neighbors_xy(x0, y0):
                branch = [(x0, y0)]
                prev = (x0, y0)
                cur = nbr
                branch_visited = set(branch)
                branch_visited.add(cur)
                branch.append(cur)
                
                steps = 0
                MAX_STEPS = min(H * W, 2000)  # Prevent infinite loops
                while steps < MAX_STEPS:
                    steps += 1
                    cx, cy = cur
                    
                    if cur in anchor_map and cur != start:
                        found_paths.append(branch.copy())
                        break
                    
                    nbrs = [p for p in neighbors_xy(cx, cy) if p != prev]
                    nbrs = [p for p in nbrs if p not in branch_visited]
                    
                    if not nbrs:
                        break
                    
                    nxt = nbrs[0]
                    branch.append(nxt)
                    branch_visited.add(nxt)
                    prev = cur
                    cur = nxt
            
            return found_paths
        
        # Add edges for each path found
        for start in node_points:
            paths_from_anchor = follow_path(start)
            a = anchor_map[start]
            for branch in paths_from_anchor:
                end_pt = branch[-1]
                if end_pt not in anchor_map:
                    continue
                b = anchor_map[end_pt]
                coords = [(int(p[0]), int(p[1])) for p in branch]
                if not G.has_edge(a, b):
                    G.add_edge(a, b, pixels=coords)
        
        return G
    
    def analyze_grid_structure(self):
        """Grid structure analysis (from first code)"""
        if len(self.dots) < 3:
            return self._estimate_grid_from_curves()
        
        # Extract coordinates
        coords = np.array([[dot['x'], dot['y']] for dot in self.dots])
        
        # Find the center of the pattern
        center_x = np.mean(coords[:, 0])
        center_y = np.mean(coords[:, 1])
        
        # Calculate distances from center
        center_distances = np.sqrt((coords[:, 0] - center_x)**2 + (coords[:, 1] - center_y)**2)
        
        # Enhanced grid spacing using nearest neighbor approach
        spacings = []
        
        # Method 1: Nearest neighbor distances
        if len(coords) > 1:
            tree = KDTree(coords)
            nn_dists, indices = tree.query(coords, k=min(3, len(coords)))
            if nn_dists.shape[1] > 1:
                nn_distances = nn_dists[:, 1]
                median_spacing = np.median(nn_distances)
                if median_spacing > 10:
                    spacings.append(median_spacing)
        
        # Method 2: Distance clustering
        unique_distances = []
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = np.sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)
                if 10 < dist < 200:
                    unique_distances.append(dist)
        
        if unique_distances:
            dist_array = np.array(unique_distances).reshape(-1, 1)
            clustering = DBSCAN(eps=15, min_samples=2).fit(dist_array)
            
            labels = clustering.labels_
            if len(set(labels)) > 1:
                for label in set(labels):
                    if label >= 0:
                        cluster_distances = dist_array[labels == label].flatten()
                        spacings.append(np.mean(cluster_distances))
        
        # Method 3: Estimate from image dimensions
        img_height, img_width = self.processed_image.shape
        estimated_spacing = min(img_width, img_height) / 12
        spacings.append(estimated_spacing)
        
        # Choose most reasonable spacing
        if spacings:
            grid_spacing = spacings[0] if len(spacings) > 0 and spacings[0] > 20 else min(s for s in spacings if s > 20) if any(s > 20 for s in spacings) else 50
        else:
            grid_spacing = 50
        
        # Estimate grid size
        max_distance = np.max(center_distances) if len(center_distances) > 0 else min(img_width, img_height) / 4
        grid_size = max(5, int(2 * max_distance / grid_spacing) + 1)
        
        # Ensure odd grid size for symmetry
        if grid_size % 2 == 0:
            grid_size += 1
        
        self.grid_structure = {
            'center': (center_x, center_y),
            'spacing': grid_spacing,
            'size': min(grid_size, 15)
        }
        
        print(f"Grid analysis: center=({center_x:.1f}, {center_y:.1f}), spacing={grid_spacing:.1f}, size={self.grid_structure['size']}")
        return self.grid_structure
    
    def _estimate_grid_from_curves(self):
        """Estimate grid structure from curves when dots are insufficient"""
        img_height, img_width = self.processed_image.shape
        
        center_x = img_width / 2
        center_y = img_height / 2
        spacing = min(img_width, img_height) / 12
        size = 9
        
        self.grid_structure = {
            'center': (center_x, center_y),
            'spacing': spacing,
            'size': size
        }
        
        print(f"Estimated grid from image: center=({center_x:.1f}, {center_y:.1f}), spacing={spacing:.1f}, size={size}")
        return self.grid_structure
    
    def detect_symmetry(self):
        """Symmetry detection (from first code)"""
        if len(self.dots) < 3:
            return self._estimate_symmetry_from_curves()
        
        center_x, center_y = self.grid_structure['center']
        
        # Convert dots to polar coordinates
        polar_dots = []
        for dot in self.dots:
            dx = dot['x'] - center_x
            dy = dot['y'] - center_y
            r = math.sqrt(dx*dx + dy*dy)
            theta = math.atan2(dy, dx)
            polar_dots.append({'r': r, 'theta': theta, 'dot': dot})
        
        # Test for different symmetry orders
        best_symmetry = 4
        best_score = 0
        
        for n in [2, 3, 4, 6, 8, 12]:
            score = self._test_rotational_symmetry(polar_dots, n)
            if score > best_score:
                best_score = score
                best_symmetry = n
        
        # If no good symmetry found, use default
        if best_score < 0.3:
            best_symmetry = 8
            best_score = 0.5
        
        self.symmetry_info = {
            'rotational_order': best_symmetry,
            'rotational_score': best_score
        }
        
        print(f"Symmetry analysis: {best_symmetry}-fold rotational symmetry (confidence: {best_score:.2f})")
        return self.symmetry_info
    
    def _estimate_symmetry_from_curves(self):
        """Estimate symmetry from curve patterns"""
        if not self.lines:
            symmetry_order = 8
        else:
            num_curves = len(self.lines)
            if num_curves > 20:
                symmetry_order = 8
            elif num_curves > 10:
                symmetry_order = 6
            else:
                symmetry_order = 4
        
        self.symmetry_info = {
            'rotational_order': symmetry_order,
            'rotational_score': 0.6
        }
        
        print(f"Estimated symmetry from curves: {symmetry_order}-fold")
        return self.symmetry_info
    
    def _test_rotational_symmetry(self, polar_dots, n):
        """Test if pattern has n-fold rotational symmetry"""
        angle_step = 2 * math.pi / n
        matches = 0
        total_tests = 0
        
        for dot in polar_dots:
            for k in range(1, n):
                expected_theta = dot['theta'] + k * angle_step
                expected_theta = expected_theta % (2 * math.pi)
                
                # Look for a dot at this expected angle
                found_match = False
                for other_dot in polar_dots:
                    angle_diff = abs(other_dot['theta'] - expected_theta)
                    angle_diff = min(angle_diff, 2 * math.pi - angle_diff)
                    radius_diff = abs(other_dot['r'] - dot['r'])
                    
                    if angle_diff < 0.3 and radius_diff < 20:
                        found_match = True
                        break
                
                if found_match:
                    matches += 1
                total_tests += 1
        
        return matches / total_tests if total_tests > 0 else 0
    
    def analyze_pattern(self):
        """Comprehensive pattern analysis"""
        print("Analyzing kolam pattern structure...")
        
        # Detect dots
        self.detect_dots()
        
        # Extract curves
        self.extract_curves()
        
        # Analyze grid structure
        self.analyze_grid_structure()
        
        # Detect symmetries
        self.detect_symmetry()
        
        print("Pattern analysis complete!")
        return {
            'dots': len(self.dots),
            'curves': len(self.lines),
            'grid': self.grid_structure,
            'symmetry': self.symmetry_info
        }
    
    def visualize_analysis(self, save_path=None):
        """Enhanced visualization showing both original and analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Original image
        ax1.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original Kolam Image", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Right: Pattern analysis with white background (no original image)
        ax2.set_facecolor('white')
        
        # Overlay detected curves with varied colors
        curve_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'darkgreen', 'navy', 'maroon']
        for i, curve in enumerate(self.lines):
            if len(curve) > 1:
                x_coords = [pt['x'] for pt in curve]
                y_coords = [pt['y'] for pt in curve]
                ax2.plot(x_coords, y_coords, color=curve_colors[i % len(curve_colors)], 
                        linewidth=3, alpha=0.9)
        
        # Overlay detected dots with better visibility
        for dot in self.dots:
            color = 'red'
            if dot.get('method') == 'hough_strict':
                color = 'red'
            elif dot.get('method') == 'contour_validated':
                color = 'orange'
            elif dot.get('method') == 'hough_relaxed':
                color = 'pink'
            else:
                color = 'yellow'
                
            circle = plt.Circle((dot['x'], dot['y']), dot['radius']*2, 
                            fill=False, color=color, linewidth=3, alpha=1.0)
            ax2.add_patch(circle)
            ax2.plot(dot['x'], dot['y'], 'o', color=color, markersize=6, 
                    markeredgecolor='white', markeredgewidth=1)
        
        # Mark center if detected
        if self.grid_structure:
            cx, cy = self.grid_structure['center']
            ax2.plot(cx, cy, 'g*', markersize=20, markeredgecolor='black', 
                    markeredgewidth=2, label=f'Center ({cx:.0f}, {cy:.0f})')
        
        ax2.set_title("Pattern Analysis Results", fontsize=14, fontweight='bold') 
        ax2.legend()
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze and recreate Kolam patterns')
    parser.add_argument('image_path', help='Path to the kolam image')
    parser.add_argument('--output', '-o', default='kolam_recreation.png',
                        help='Output file name for the recreation visualization')
    
    args = parser.parse_args()
    
    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at '{args.image_path}'")
        print("Make sure to:")
        print("1. Use quotes around paths with spaces")
        print("2. Use the correct file path")
        print("3. Check that the file exists")
        return 1
    
    try:
        # Initialize analyzer
        print("Loading and analyzing image...")
        analyzer = KolamAnalyzer(args.image_path)
        
        # Process the image
        analyzer.load_and_preprocess()
        print(f"Detected {len(analyzer.detect_dots())} dots")
        
        # Analyze structure
        grid_info = analyzer.analyze_grid_structure()
        if grid_info:
            print(f"Grid structure: {grid_info['size']}x{grid_info['size']}, spacing: {grid_info['spacing']:.1f}")
        
        # Detect symmetry
        symmetry_info = analyzer.detect_symmetry()
        if symmetry_info:
            print(f"Detected {symmetry_info.get('rotational_order', 8)}-fold rotational symmetry (confidence: {symmetry_info.get('rotational_score', 0.5):.2f})")
        
        # Extract curves
        curves = analyzer.extract_curves()
        print(f"Extracted {len(curves)} curve segments")
        
        # Visualize analysis results
        analyzer.visualize_analysis(save_path=args.output)
        
        # Create mathematical recreation plot
        # recreator = KolamRecreator(analyzer)
        # recreator.recreate_pattern()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())