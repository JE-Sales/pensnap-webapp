import numpy as np

class Position:
    def __init__(self, line_threshold=20):
        self.line_threshold = line_threshold

    def calculate_polygon_center(self, polygon):
        """
        Calculates the center (average of y-coordinates) of a polygon.
        """
        y_coords = [polygon[i+1] for i in range(0, len(polygon), 2)]  # Extract y-coordinates (every 2nd value)
        return np.mean(y_coords)

    def group_words_by_line(self, polygons, scores):
        """
        Groups words into lines based on the vertical center of their polygons.
        """
        word_centers = [self.calculate_polygon_center(polygon) for polygon in polygons]

        # Sort polygons by their y-coordinate (vertical position)
        sorted_indices = np.argsort(word_centers)

        # Group words into lines
        lines = []
        current_line = []
        last_y_center = word_centers[sorted_indices[0]]

        for idx in sorted_indices:
            word_polygon = polygons[idx]
            word_score = scores[idx]
            y_center = word_centers[idx]

            if abs(y_center - last_y_center) < self.line_threshold:
                current_line.append((word_polygon, word_score, idx))
            else:
                lines.append(current_line)
                current_line = [(word_polygon, word_score, idx)]

            last_y_center = y_center

        if current_line:
            lines.append(current_line)

        return lines

    def sort_words_in_line(self, line):
        """
        Sorts words in a line based on the x-coordinate of their polygons (leftmost point).
        """
        return sorted(line, key=lambda word: min(word[0][i] for i in range(0, len(word[0]), 2)))  # Sort by x-coordinates

    def process_predictions(self, predictions):
        """
        Processes the predictions and returns the final text in the correct word order.
        """
        # Extract polygons, text, and scores from the predictions
        rec_texts = predictions['rec_texts']
        rec_scores = predictions['rec_scores']
        det_polygons = predictions['det_polygons']

        # Group words by line based on their polygon centers
        lines = self.group_words_by_line(det_polygons, rec_scores)

        final_text = []

        for line in lines:
            # Sort words in each line from left to right
            sorted_line = self.sort_words_in_line(line)

            # Collect the corresponding text for each word
            line_text = [rec_texts[idx] for _, _, idx in sorted_line]
            final_text.append(" ".join(line_text))

        # Join all lines with line breaks
        return "\n".join(final_text)
