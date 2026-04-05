"""
Tile Coding for Function Approximation.

Based on Sutton & Barto's tile coding implementation.
Used by the True Online Sarsa(λ) agent for feature construction.
"""

import numpy as np
from typing import List, Tuple


class IHT:
    """Index Hash Table for tile coding.
    
    Maps tile indices to a fixed-size table using hashing.
    Handles collisions by maintaining a dictionary of seen index tuples.
    """
    
    def __init__(self, size: int):
        self.size = size
        self.overfull_count = 0
        self.dictionary = {}
    
    def count(self) -> int:
        return len(self.dictionary)
    
    def full(self) -> bool:
        return len(self.dictionary) >= self.size
    
    def get_index(self, obj, read_only: bool = False) -> int:
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfull_count == 0:
                print(f"IHT full, starting to allow collisions")
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count


def tiles(iht: IHT, num_tilings: int, floats: List[float], 
          ints: List[int] = None, read_only: bool = False) -> List[int]:
    """Return list of tile indices for the given input.
    
    Args:
        iht: Index hash table
        num_tilings: Number of tilings
        floats: List of continuous variables (will be scaled externally)
        ints: List of integer variables (e.g., action)
        read_only: If True, don't add new entries to IHT
        
    Returns:
        List of tile indices (one per tiling)
    """
    if ints is None:
        ints = []
    
    qfloats = [int(np.floor(f * num_tilings)) for f in floats]
    result = []
    
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tilingX2
        coords.extend(ints)
        result.append(iht.get_index(tuple(coords), read_only))
    
    return result


class TileCoder:
    """Tile coding wrapper for the Text Flappy Bird environment.
    
    Provides a convenient interface for converting (Δx, Δy, action) 
    into a sparse feature vector suitable for linear function approximation.
    """
    
    def __init__(self, num_tilings: int = 8, num_tiles: int = 8, 
                 iht_size: int = 4096,
                 obs_low: np.ndarray = None, obs_high: np.ndarray = None):
        """
        Args:
            num_tilings: Number of overlapping tilings
            num_tiles: Number of tiles per dimension per tiling
            iht_size: Size of the index hash table
            obs_low: Lower bounds of observation space [Δx_min, Δy_min]
            obs_high: Upper bounds of observation space [Δx_max, Δy_max]
        """
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.iht_size = iht_size
        self.iht = IHT(iht_size)
        
        # Default observation ranges for TextFlappyBird-v0
        if obs_low is None:
            obs_low = np.array([0.0, -12.0])
        if obs_high is None:
            obs_high = np.array([14.0, 12.0])
        
        self.obs_low = obs_low
        self.obs_high = obs_high
        self.obs_range = obs_high - obs_low
        
        # Scale factors: map observation to [0, num_tiles] range
        self.scale = self.num_tiles / self.obs_range
    
    def get_tiles(self, obs: np.ndarray, action: int) -> List[int]:
        """Get tile indices for a (state, action) pair.
        
        Args:
            obs: Observation array [Δx, Δy]
            action: Action (0 or 1)
            
        Returns:
            List of active tile indices
        """
        scaled_obs = (obs - self.obs_low) * self.scale
        return tiles(self.iht, self.num_tilings, 
                     scaled_obs.tolist(), [action])
    
    def get_feature_vector(self, obs: np.ndarray, action: int) -> np.ndarray:
        """Get sparse feature vector (binary) for (state, action).
        
        Args:
            obs: Observation array [Δx, Δy]
            action: Action (0 or 1)
            
        Returns:
            Binary feature vector of size iht_size
        """
        x = np.zeros(self.iht_size)
        active_tiles = self.get_tiles(obs, action)
        for idx in active_tiles:
            if idx is not None:
                x[idx] = 1.0
        return x
    
    @property
    def feature_size(self) -> int:
        """Size of the feature vector."""
        return self.iht_size
