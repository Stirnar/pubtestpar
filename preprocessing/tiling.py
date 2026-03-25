import os
from pathlib import Path
import numpy as np
from skimage import io, color, img_as_ubyte
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk, remove_small_holes
from skimage.util import view_as_windows
from skimage.exposure import rescale_intensity


class MicroscopeImageProcessor:
    """
    Encapsulates the processing pipeline for a single microscope image to extract valid tiles.
    The pipeline involves:
    1. Finding the circular Field of View (FOV).
    2. Masking the image to keep only the FOV.
    3. Extracting smaller, overlapping tiles from the valid area.
    """

    def __init__(self, image_path: Path):
        """
        Initializes the processor with the path to an image.

        Args:
            image_path (Path): The path to the microscope image file.
        """
        self.image_path = image_path
        self.filename = image_path.name
        
        # Load image and convert to grayscale float for processing
        img = io.imread(image_path)
        self.original_image = img
        # Ensure image is 2D grayscale, normalized to [0, 1]
        self.gray_image = color.rgb2gray(img) if img.ndim == 3 else img.astype(np.float32) / np.max(img)
        
        # These will be computed by the processing steps
        self.fov_mask = None
        self.clean_image = None
        self.tiles = None
        self.tile_coords = None

    def _create_fov_mask(self, closing_disk_size: int = 15, min_hole_area: int = 1000, min_region_area: int = 50000):
        """
        Segments the circular field of view (FOV) from the background.
        This method assumes the FOV is a large, roughly circular, bright region.
        """
        # Step 1: Rough segmentation using Otsu's thresholding.
        # This works well if the area outside the FOV is darker than the inside.
        t = threshold_otsu(self.gray_image)
        mask = self.gray_image > t
        
        # Step 2: Clean up the mask using morphological operations to close gaps and remove noise.
        mask = binary_closing(mask, disk(closing_disk_size))
        mask = remove_small_holes(mask, area_threshold=min_hole_area)

        # Step 3: Find all distinct regions in the mask.
        lab = label(mask)
        regions = regionprops(lab)
        
        # Step 4: Filter out small, irrelevant regions.
        regions = [r for r in regions if r.area > min_region_area]
        if not regions:
            raise ValueError(f"Could not find a sufficiently large FOV region in {self.filename}")

        # Step 5: Identify the main FOV region by selecting the one with the highest circularity.
        # Circularity = 4*pi*area / perimeter^2. A perfect circle has circularity 1.
        def circularity(r):
            return 4 * np.pi * r.area / (r.perimeter**2 + 1e-8)
        
        fov_region = max(regions, key=circularity)
        
        # Step 6: Create a perfect circular mask based on the properties of the found region.
        # This provides a cleaner boundary than the initial segmentation.
        cy, cx = fov_region.centroid
        radius = 0.5 * fov_region.equivalent_diameter
        
        h, w = self.gray_image.shape
        Y, X = np.ogrid[:h, :w]
        
        self.fov_mask = (X - cx)**2 + (Y - cy)**2 <= radius**2
        
        # Create a "clean" version of the image where everything outside the FOV is black.
        self.clean_image = self.gray_image.copy()
        self.clean_image[~self.fov_mask] = 0

    def extract_tiles(self, patch_size: int = 256, stride: int = 128, min_fov_fraction: float = 0.9, require_center_inside: bool = True):
        """
        Extracts overlapping tiles (patches) from the image.

        Args:
            patch_size (int): The height and width of the square tiles.
            stride (int): The step size to move between tiles. Overlap = patch_size - stride.
            min_fov_fraction (float): The minimum fraction of a tile that must be inside the FOV to be kept.
            require_center_inside (bool): If True, only keep tiles whose center pixel is inside the FOV.
        
        Returns:
            A tuple of (tiles, coords):
            - tiles: (N, patch_size, patch_size) numpy array of image data.
            - coords: (N, 2) numpy array of (row, col) top-left coordinates for each tile.
        """
        if self.fov_mask is None:
            self._create_fov_mask()

        # Use view_as_windows for an efficient, sliding-window view of the arrays.
        patches = view_as_windows(self.gray_image, (patch_size, patch_size), step=stride)
        mask_patches = view_as_windows(self.fov_mask.astype(np.uint8), (patch_size, patch_size), step=stride)

        n_rows, n_cols = patches.shape[:2]
        
        tiles = []
        coords = []
        center = patch_size // 2

        for i in range(n_rows):
            r0 = i * stride
            for j in range(n_cols):
                c0 = j * stride

                # Check if the tile meets our criteria
                mask_tile = mask_patches[i, j]
                fov_fraction = mask_tile.mean()

                if require_center_inside and (mask_tile[center, center] == 0):
                    continue

                if fov_fraction >= min_fov_fraction:
                    tiles.append(patches[i, j])
                    coords.append((r0, c0))
        
        self.tiles = np.stack(tiles) if tiles else np.empty((0, patch_size, patch_size))
        self.tile_coords = np.array(coords, dtype=int) if coords else np.empty((0, 2), dtype=int)
        
        return self.tiles, self.tile_coords

    def save_tiles(self, out_dir: Path, prefix: str, save_masked: bool = False):
        """
        Saves each extracted tile as a PNG image.

        Args:
            out_dir (Path): The directory where tiles will be saved.
            prefix (str): A prefix for the output filenames.
            save_masked (bool): If True, also save a version where the area outside the FOV is black.
        """
        if self.tiles is None or len(self.tiles) == 0:
            print(f"No valid tiles to save for {self.filename}.")
            return

        os.makedirs(out_dir, exist_ok=True)

        for k, (tile, (r0, c0)) in enumerate(zip(self.tiles, self.tile_coords)):
            # Normalize each tile's intensity for better visualization before saving.
            t_float = tile.astype(np.float32)
            t_norm = rescale_intensity(t_float, in_range="image", out_range=(0, 1))
            png = img_as_ubyte(t_norm)

            # Save the raw tile
            fname = f"{prefix}_{k:05d}_r{r0}_c{c0}.png"
            io.imsave(out_dir / fname, png)

            """ 
            # Optionally, save a version with the FOV mask applied
            if save_masked:
                h, w = tile.shape
                mask_tile = self.fov_mask[r0:r0+h, c0:c0+w]
                t_masked = t_norm.copy()
                t_masked[~mask_tile] = 0.0
                png_masked = img_as_ubyte(t_masked)
                fname_masked = f"{prefix}_{k:05d}_r{r0}_c{c0}_masked.png"
                io.imsave(out_dir / fname_masked, png_masked)
            """

        print(f"Saved {len(self.tiles)} tiles for {self.filename} to: {out_dir.resolve()}")


def process_images(
    source_dir: Path, 
    dest_dir: Path, 
    microscope_type: str = None, 
    parasite_type: str = None,
    patch_size: int = 256,
    stride: int = 128,
    min_fov_fraction: float = 0.9,
    save_masked: bool = True
):
    """
    Finds images in a source directory, processes them to extract tiles, and saves the tiles.

    Args:
        source_dir (Path): Directory containing the original microscope images.
        dest_dir (Path): Root directory where processed tiles will be saved.
        microscope_type (str, optional): If provided, only process images from this microscope.
        parasite_type (str, optional): If provided, only process images of this parasite.
        patch_size (int): The size of tiles to extract.
        stride (int): The step size between tiles.
        min_fov_fraction (float): The minimum fraction of a tile that must be inside the FOV.
        save_masked (bool): Whether to save masked versions of the tiles.
    """
    # Ensure the destination directory exists
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']

    # Iterate over all files in the source directory
    for image_path in source_dir.iterdir():
        # Skip non-image files
        if image_path.suffix.lower() not in image_extensions:
            continue

        # --- Filtering based on filename conventions ---
        # Assumes filename format: {microscope}__{parasite}__{...}.jpg
        parts = image_path.stem.split('__')
        if len(parts) < 2:
            print(f"Skipping {image_path.name}: does not match expected format for filtering.")
            continue
        
        img_microscope = parts[0]
        img_parasite = parts[1]

        # Apply filters if they are provided
        if microscope_type and microscope_type.lower() != img_microscope.lower():
            continue
        if parasite_type and parasite_type.lower() != img_parasite.lower():
            continue
            
        print(f"Processing: {image_path.name}")
        try:
            # --- Processing a single image ---
            # 1. Initialize the processor for the image
            processor = MicroscopeImageProcessor(image_path)
            
            # 2. Extract tiles using the specified parameters
            processor.extract_tiles(
                patch_size=patch_size, 
                stride=stride, 
                min_fov_fraction=min_fov_fraction
            )
            
            # 3. Define a specific output directory and filename prefix for this image's tiles
            # e.g., data/tiles/Ascaris_Lumbricoides/
            tile_output_dir = dest_dir / img_parasite
            file_prefix = image_path.stem # Use the original filename stem as a prefix

            # 4. Save the extracted tiles
            processor.save_tiles(
                out_dir=tile_output_dir,
                prefix=file_prefix,
                save_masked=save_masked
            )

        except (ValueError, IndexError) as e:
            print(f"Could not process {image_path.name}. Error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {image_path.name}: {e}")
