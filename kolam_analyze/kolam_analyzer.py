import cv2
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import math
from collections import Counter

class KolamAnalyzer:
    """Analyzes kolam images to extract descriptive features for natural language generation."""
    
    def __init__(self):
        self.features = {}
    
    def analyze_image(self, image_path):
        """Main analysis function that extracts all features from a kolam image."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not load image")
        
        # Preprocess image
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Extract features
        self.features = {
            'grid_info': self._analyze_dot_grid(binary),
            'symmetry': self._analyze_symmetry(binary),
            'curves': self._analyze_curves(binary),
            'complexity': self._analyze_complexity(binary),
            'patterns': self._analyze_patterns(binary),
            'traditional_elements': self._detect_traditional_elements(binary)
        }
        
        return self.features
    
    def _analyze_dot_grid(self, binary):
        """Detect and analyze the underlying dot grid structure."""
        # Find contours for potential dots/circular regions
        contours, _ = cv2.findContours(~binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for circular/dot-like regions
        dots = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area > 50 and perimeter > 0:  # Filter small noise
                circularity = 4 * math.pi * area / (perimeter * perimeter)
                if circularity > 0.3:  # Reasonably circular
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        dots.append((cx, cy))
        
        if len(dots) < 4:
            return {'type': 'no_clear_grid', 'count': len(dots)}
        
        # Try to determine grid structure
        dots = np.array(dots)
        
        # Cluster dots by rows and columns
        y_coords = dots[:, 1]
        x_coords = dots[:, 0]
        
        # Simple grid detection based on coordinate clustering
        y_clusters = self._cluster_coordinates(y_coords)
        x_clusters = self._cluster_coordinates(x_coords)
        
        rows = len(y_clusters)
        cols = len(x_clusters)
        
        # Determine grid type
        if rows == cols:
            grid_type = f"{rows}x{cols} square grid"
        else:
            grid_type = f"{rows}x{cols} rectangular grid"
        
        return {
            'type': grid_type,
            'rows': rows,
            'cols': cols,
            'total_dots': len(dots),
            'spacing': self._calculate_average_spacing(dots)
        }
    
    def _cluster_coordinates(self, coords, tolerance=30):
        """Cluster coordinates that are close together."""
        if len(coords) == 0:
            return []
        
        coords_sorted = sorted(coords)
        clusters = []
        current_cluster = [coords_sorted[0]]
        
        for coord in coords_sorted[1:]:
            if coord - current_cluster[-1] <= tolerance:
                current_cluster.append(coord)
            else:
                clusters.append(current_cluster)
                current_cluster = [coord]
        
        clusters.append(current_cluster)
        return clusters
    
    def _calculate_average_spacing(self, dots):
        """Calculate average spacing between dots."""
        if len(dots) < 2:
            return "unknown"
        
        distances = cdist(dots, dots)
        # Remove zero distances (self-distances)
        distances = distances[distances > 0]
        
        if len(distances) == 0:
            return "unknown"
        
        avg_distance = np.mean(distances)
        
        if avg_distance < 50:
            return "tight"
        elif avg_distance < 100:
            return "medium"
        else:
            return "wide"
    
    def _analyze_symmetry(self, binary):
        """Analyze symmetry properties of the kolam."""
        h, w = binary.shape
        
        # Test vertical symmetry
        left_half = binary[:, :w//2]
        right_half = np.fliplr(binary[:, w//2:])
        
        # Resize to match if needed
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        vertical_symmetry = np.mean(left_half == right_half) > 0.85
        
        # Test horizontal symmetry
        top_half = binary[:h//2, :]
        bottom_half = np.flipud(binary[h//2:, :])
        
        min_height = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[:min_height, :]
        bottom_half = bottom_half[:min_height, :]
        
        horizontal_symmetry = np.mean(top_half == bottom_half) > 0.85
        
        # Test rotational symmetry (approximate)
        rotational_symmetry = self._test_rotational_symmetry(binary)
        
        symmetry_type = []
        if vertical_symmetry:
            symmetry_type.append("vertical")
        if horizontal_symmetry:
            symmetry_type.append("horizontal")
        if rotational_symmetry:
            symmetry_type.append("rotational")
        
        if not symmetry_type:
            symmetry_type = ["asymmetric"]
        
        return {
            'types': symmetry_type,
            'vertical': vertical_symmetry,
            'horizontal': horizontal_symmetry,
            'rotational': rotational_symmetry
        }
    
    def _test_rotational_symmetry(self, binary):
        """Test for 90, 180, or other rotational symmetries."""
        h, w = binary.shape
        center = (h//2, w//2)
        
        # Test 180-degree rotation
        rotated_180 = np.rot90(binary, 2)
        similarity_180 = np.mean(binary == rotated_180)
        
        return similarity_180 > 0.8
    
    def _analyze_curves(self, binary):
        """Analyze the curve characteristics in the kolam."""
        # Find edges
        edges = cv2.Canny(binary, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'type': 'no_clear_curves', 'count': 0}
        
        curve_types = []
        total_length = 0
        
        for contour in contours:
            length = cv2.arcLength(contour, False)
            total_length += length
            
            # Analyze curvature
            if len(contour) > 10:
                # Simple curvature analysis based on contour shape
                area = cv2.contourArea(contour)
                if area > 0:
                    compactness = (length * length) / area
                    if compactness < 20:
                        curve_types.append("circular")
                    elif compactness < 50:
                        curve_types.append("curved")
                    else:
                        curve_types.append("angular")
        
        curve_counter = Counter(curve_types)
        dominant_curve = curve_counter.most_common(1)[0][0] if curve_counter else "mixed"
        
        return {
            'dominant_type': dominant_curve,
            'total_length': int(total_length),
            'curve_count': len(contours),
            'types_distribution': dict(curve_counter)
        }
    
    def _analyze_complexity(self, binary):
        """Analyze the overall complexity of the kolam."""
        # Count connected components
        num_labels, labels = cv2.connectedComponents(~binary)
        
        # Calculate fractal dimension (box-counting method, simplified)
        fractal_dim = self._estimate_fractal_dimension(binary)
        
        # Count intersections/junctions
        junctions = self._count_junctions(binary)
        
        complexity_level = "simple"
        if junctions > 20 or fractal_dim > 1.3:
            complexity_level = "complex"
        elif junctions > 10 or fractal_dim > 1.2:
            complexity_level = "moderate"
        
        return {
            'level': complexity_level,
            'connected_components': num_labels - 1,  # -1 for background
            'estimated_fractal_dim': fractal_dim,
            'junction_count': junctions
        }
    
    def _estimate_fractal_dimension(self, binary):
        """Estimate fractal dimension using simplified box-counting."""
        # Very basic implementation
        non_zero_pixels = np.sum(binary == 0)  # Count black pixels
        total_pixels = binary.size
        
        if non_zero_pixels == 0:
            return 1.0
        
        # Simple approximation based on pixel density
        density = non_zero_pixels / total_pixels
        return 1.0 + density * 0.5  # Simplified mapping
    
    def _count_junctions(self, binary):
        """Count line junctions/intersections."""
        # Apply thinning to get skeleton
        kernel = np.ones((3,3), np.uint8)
        thinned = cv2.morphologyEx(~binary, cv2.MORPH_CLOSE, kernel)
        
        # Count pixels with more than 2 neighbors (junctions)
        junctions = 0
        h, w = thinned.shape
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                if thinned[i, j] > 0:
                    neighbors = np.sum(thinned[i-1:i+2, j-1:j+2] > 0) - 1
                    if neighbors > 2:
                        junctions += 1
        
        return junctions
    
    def _analyze_patterns(self, binary):
        """Analyze repeating patterns and motifs."""
        # Find contours
        contours, _ = cv2.findContours(~binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 2:
            return {'repetition': 'none', 'motif_count': len(contours)}
        
        # Analyze size distribution of contours
        areas = [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 100]
        
        if not areas:
            return {'repetition': 'none', 'motif_count': 0}
        
        # Check for similar-sized elements (indicating repetition)
        area_std = np.std(areas)
        area_mean = np.mean(areas)
        
        if area_std / area_mean < 0.3:  # Low variation indicates repetition
            repetition_type = "high_repetition"
        elif area_std / area_mean < 0.6:
            repetition_type = "moderate_repetition"
        else:
            repetition_type = "varied_elements"
        
        return {
            'repetition': repetition_type,
            'motif_count': len(areas),
            'size_variation': area_std / area_mean if area_mean > 0 else 0
        }
    
    def _detect_traditional_elements(self, binary):
        """Detect traditional kolam elements and motifs."""
        # This is a simplified version - in practice, you'd use template matching
        # or trained models to detect specific traditional patterns
        
        contours, _ = cv2.findContours(~binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        traditional_elements = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
                
            # Simple shape analysis
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            
            if len(approx) == 3:
                traditional_elements.append("triangular_motif")
            elif len(approx) == 4:
                traditional_elements.append("diamond_motif")
            elif len(approx) > 8:
                # Check for circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * math.pi * area / (perimeter * perimeter)
                if circularity > 0.7:
                    traditional_elements.append("circular_motif")
                else:
                    traditional_elements.append("curved_motif")
        
        element_counts = Counter(traditional_elements)
        
        return {
            'detected_elements': list(element_counts.keys()),
            'element_counts': dict(element_counts),
            'dominant_element': element_counts.most_common(1)[0][0] if element_counts else None
        }
    
    def generate_description(self, features=None):
        """Generate natural language description from extracted features."""
        if features is None:
            features = self.features
        
        if not features:
            return "Unable to analyze the kolam image."
        
        description_parts = []
        
        # Grid description
        grid_info = features.get('grid_info', {})
        if grid_info.get('type') != 'no_clear_grid':
            description_parts.append(f"This kolam is built on a {grid_info.get('type', 'dot grid')} with {grid_info.get('total_dots', 'several')} dots arranged in {grid_info.get('spacing', 'regular')} spacing.")
        
        # Symmetry description
        symmetry = features.get('symmetry', {})
        if symmetry.get('types'):
            if len(symmetry['types']) == 1:
                description_parts.append(f"The pattern displays {symmetry['types'][0]} symmetry.")
            else:
                types_str = ", ".join(symmetry['types'][:-1]) + " and " + symmetry['types'][-1]
                description_parts.append(f"The design features {types_str} symmetries.")
        
        # Curve description
        curves = features.get('curves', {})
        if curves.get('dominant_type'):
            if curves['dominant_type'] == 'circular':
                description_parts.append("The drawing consists primarily of smooth, flowing circular curves.")
            elif curves['dominant_type'] == 'curved':
                description_parts.append("The pattern features graceful curved lines that flow around the dots.")
            elif curves['dominant_type'] == 'angular':
                description_parts.append("The design incorporates more angular, geometric line segments.")
        
        # Complexity description
        complexity = features.get('complexity', {})
        if complexity.get('level'):
            if complexity['level'] == 'simple':
                description_parts.append("This is a relatively simple kolam with clean, uncluttered lines.")
            elif complexity['level'] == 'moderate':
                description_parts.append("The kolam shows moderate complexity with several interconnected elements.")
            elif complexity['level'] == 'complex':
                description_parts.append("This is an intricate kolam with many interconnected patterns and detailed elements.")
        
        # Pattern repetition
        patterns = features.get('patterns', {})
        if patterns.get('repetition') == 'high_repetition':
            description_parts.append(f"The design shows strong repetitive elements with {patterns.get('motif_count', 'several')} similar motifs.")
        elif patterns.get('repetition') == 'moderate_repetition':
            description_parts.append("The pattern includes some repeating elements that create visual rhythm.")
        
        # Traditional elements
        traditional = features.get('traditional_elements', {})
        if traditional.get('detected_elements'):
            elements = traditional['detected_elements']
            if len(elements) == 1:
                description_parts.append(f"The kolam incorporates traditional {elements[0].replace('_', ' ')}s.")
            elif len(elements) > 1:
                description_parts.append(f"Traditional elements include {', '.join([e.replace('_', ' ') for e in elements])}.")
        
        # Final artistic description
        description_parts.append("The overall composition follows traditional kolam principles of continuous line drawing that creates enclosed spaces around the dots.")
        
        return " ".join(description_parts)


def analyze_kolam_image(image_path):
    """Convenience function to analyze an image and return description."""
    analyzer = KolamAnalyzer()
    features = analyzer.analyze_image(image_path)
    description = analyzer.generate_description(features)
    
    # Convert all numpy types and complex objects to JSON-serializable types
    def make_serializable(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    return {
        'description': description,
        'features': make_serializable(features)
    }