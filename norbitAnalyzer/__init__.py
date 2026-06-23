import os

__version__ = "0.0.1"

# Fixed: Changed os.path_dirname to os.path.dirname
norbit_dir = os.path.dirname(__file__)
DATADIR = os.path.join(norbit_dir, "norbit_data_ex/")

# Check for PyCUDA availability
try: 
    import pycuda.driver as cuda 
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    cuda_ext = True
except ImportError: 
    cuda_ext = False


try: 
    from . import _kepler
    cext = True 
except ImportError: 
    cext = False  

