"""
Integral Image Module for Viola-Jones Face Detector

This module implements the Integral Image (Summed-Area Table) representation
for efficient computation of rectangle features.

The integral image allows any Haar-like feature to be computed at any scale
or location in constant time.
"""

import numpy as np


class IntegralImage:
    """Compute and use integral image for efficient rectangle sum computation."""
    
    def __init__(self, image):
        """
        Initialize integral image from input image.
        
        Args:
            image: 2D numpy array (grayscale image)
        """
        self.original_image = image.astype(np.float64)
        self.integral_image = self.compute_integral_image(self.original_image)
    
    @staticmethod
    def compute_integral_image(image):
        """
        Compute integral image using recurrence relations.
        
        The integral image at location (x, y) contains the sum of all pixels
        above and to the left of (x, y), inclusive.
        
        Recurrence relations:
            s(x, y) = s(x, y-1) + i(x, y)
            ii(x, y) = ii(x-1, y) + s(x, y)
        where s(x, y) is cumulative row sum
        
        Args:
            image: 2D numpy array
            
        Returns:
            2D numpy array (integral image)
        """
        h, w = image.shape
        
        # Initialize integral image (with extra row and column of zeros for boundary)
        ii = np.zeros((h + 1, w + 1), dtype=np.float64)
        
        # Compute integral image using cumulative sums
        # This is equivalent to the recurrence relations but more efficient
        for y in range(h):
            for x in range(w):
                ii[y + 1, x + 1] = (image[y, x] + 
                                    ii[y, x + 1] + 
                                    ii[y + 1, x] - 
                                    ii[y, x])
        
        return ii
    
    def get_rectangle_sum(self, x, y, width, height):
        """
        Get sum of pixels in rectangle using 4 array references.
        
        Rectangle is defined by top-left corner (x, y) and dimensions.
        
        For a rectangle with corners at points 1, 2, 3, 4:
            Sum(D) = ii(4) + ii(1) - (ii(2) + ii(3))
        
        Args:
            x: Top-left x coordinate
            y: Top-left y coordinate
            width: Rectangle width
            height: Rectangle height
            
        Returns:
            Sum of pixels in rectangle
        """
        # Adjust coordinates for 1-indexed integral image
        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height
        
        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, self.integral_image.shape[1] - 1))
        y1 = max(0, min(y1, self.integral_image.shape[0] - 1))
        x2 = max(0, min(x2, self.integral_image.shape[1] - 1))
        y2 = max(0, min(y2, self.integral_image.shape[0] - 1))
        
        # Calculate sum using 4 array references
        total = (self.integral_image[y2, x2] + 
                self.integral_image[y1, x1] - 
                self.integral_image[y1, x2] - 
                self.integral_image[y2, x1])
        
        return total
    
    def get_two_rectangle_feature(self, x, y, width, height, orientation='horizontal'):
        """
        Compute two-rectangle feature (difference between adjacent rectangles).
        
        Requires 6 array references.
        
        Args:
            x: Top-left x coordinate
            y: Top-left y coordinate
            width: Total width (for horizontal) or individual width (for vertical)
            height: Individual height (for horizontal) or total height (for vertical)
            orientation: 'horizontal' or 'vertical'
            
        Returns:
            Difference between white and grey rectangles
        """
        if orientation == 'horizontal':
            # Left (white) - Right (grey)
            half_width = width // 2
            left_sum = self.get_rectangle_sum(x, y, half_width, height)
            right_sum = self.get_rectangle_sum(x + half_width, y, half_width, height)
            return left_sum - right_sum
        else:  # vertical
            # Top (white) - Bottom (grey)
            half_height = height // 2
            top_sum = self.get_rectangle_sum(x, y, width, half_height)
            bottom_sum = self.get_rectangle_sum(x, y + half_height, width, half_height)
            return top_sum - bottom_sum
    
    def get_three_rectangle_feature(self, x, y, width, height, orientation='horizontal'):
        """
        Compute three-rectangle feature (outside rectangles - center rectangle).
        
        Requires 8 array references.
        
        Args:
            x: Top-left x coordinate
            y: Top-left y coordinate
            width: Total width (for horizontal) or individual width (for vertical)
            height: Individual height (for horizontal) or total height (for vertical)
            orientation: 'horizontal' or 'vertical'
            
        Returns:
            Difference: (left + right - center) or (top + bottom - center)
        """
        if orientation == 'horizontal':
            # (Left + Right) - Center
            third_width = width // 3
            left_sum = self.get_rectangle_sum(x, y, third_width, height)
            center_sum = self.get_rectangle_sum(x + third_width, y, third_width, height)
            right_sum = self.get_rectangle_sum(x + 2 * third_width, y, third_width, height)
            return left_sum + right_sum - center_sum
        else:  # vertical
            # (Top + Bottom) - Center
            third_height = height // 3
            top_sum = self.get_rectangle_sum(x, y, width, third_height)
            center_sum = self.get_rectangle_sum(x, y + third_height, width, third_height)
            bottom_sum = self.get_rectangle_sum(x, y + 2 * third_height, width, third_height)
            return top_sum + bottom_sum - center_sum
    
    def get_four_rectangle_feature(self, x, y, width, height):
        """
        Compute four-rectangle feature (diagonal difference).
        
        Requires 9 array references.
        
        Args:
            x: Top-left x coordinate
            y: Top-left y coordinate
            width: Total width
            height: Total height
            
        Returns:
            Difference between diagonal pairs: (top-left + bottom-right) - (top-right + bottom-left)
        """
        half_width = width // 2
        half_height = height // 2
        
        # Top-left (white)
        top_left = self.get_rectangle_sum(x, y, half_width, half_height)
        
        # Top-right (grey)
        top_right = self.get_rectangle_sum(x + half_width, y, half_width, half_height)
        
        # Bottom-left (grey)
        bottom_left = self.get_rectangle_sum(x, y + half_height, half_width, half_height)
        
        # Bottom-right (white)
        bottom_right = self.get_rectangle_sum(x + half_width, y + half_height, half_width, half_height)
        
        return (top_left + bottom_right) - (top_right + bottom_left)


def compute_integral_image_fast(image):
    """
    Fast computation of integral image using numpy cumsum.
    
    This is a convenience function for quick integral image computation.
    
    Args:
        image: 2D numpy array
        
    Returns:
        IntegralImage object
    """
    return IntegralImage(image)


# Example usage and testing
if __name__ == "__main__":
    # Create a simple test image
    test_image = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=np.float64)
    
    print("Original Image:")
    print(test_image)
    
    # Compute integral image
    ii = IntegralImage(test_image)
    
    print("\nIntegral Image:")
    print(ii.integral_image)
    
    # Test rectangle sum
    rect_sum = ii.get_rectangle_sum(0, 0, 2, 2)
    print(f"\nSum of top-left 2x2 rectangle: {rect_sum}")
    print(f"Expected: {1 + 2 + 5 + 6} = 14")
    
    # Test two-rectangle feature
    two_rect = ii.get_two_rectangle_feature(0, 0, 4, 2, orientation='horizontal')
    print(f"\nTwo-rectangle feature (horizontal): {two_rect}")
    
    # Test three-rectangle feature
    three_rect = ii.get_three_rectangle_feature(0, 0, 3, 2, orientation='horizontal')
    print(f"\nThree-rectangle feature (horizontal): {three_rect}")
    
    # Test four-rectangle feature
    four_rect = ii.get_four_rectangle_feature(0, 0, 4, 4)
    print(f"\nFour-rectangle feature: {four_rect}")
