import random


class BaseTransform:
    """
    Internal base class for all audio transforms in Soundmentations.
    
    .. warning::
        This class is for internal use only and should not be used directly
        by end users. Use the specific transform classes instead.
    
    This class provides common functionality for all transforms including
    probability handling and validation. All transform classes inherit
    from this base class to ensure consistent behavior across the library.
    
    Parameters
    ----------
    p : float, optional
        Probability of applying the transform, by default 1.0.
        Must be between 0.0 and 1.0.
    
    Notes
    -----
    This is an internal implementation detail. End users should use
    specific transform classes like Gain, Trim, Pad, etc.
    """
    
    def __init__(self, p: float = 1.0):
        """
        Initialize the base transform with probability parameter.
        
        Parameters
        ----------
        p : float, optional
            Probability of applying the transform, by default 1.0.
            Must be between 0.0 and 1.0.
        
        Raises
        ------
        TypeError
            If p is not a float or integer.
        ValueError
            If p is not between 0.0 and 1.0.
        """
        if not isinstance(p, (float, int)):
            raise TypeError("p must be a float or an integer")
        if not (0.0 <= p <= 1.0):
            raise ValueError("p must be between 0.0 and 1.0")
        self.p = p

    def should_apply(self) -> bool:
        """
        Determine whether the transform should be applied based on probability.
        
        Returns
        -------
        bool
            True if the transform should be applied, False otherwise.
            
        Notes
        -----
        Internal method used by transform implementations.
        """
        return random.random() < self.p

    def __call__(self, *args, **kwargs):
        """
        Apply the transform to input data.
        
        This is an abstract method that must be implemented by all subclasses.
        
        Raises
        ------
        NotImplementedError
            Always raised as this is an abstract method.
        """
        raise NotImplementedError("Subclasses must implement __call__.")
