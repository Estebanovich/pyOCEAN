"""
Components of the 3-D core (N1).  Each module holds one role of ``core.model.contract.Role``: a small object built
from its parameters that calls pure array kernels, declares what it reads and writes, and reports configuration
problems through ``check``.  The physics is unchanged from ``SigmaBaroclinic``, from which the code was extracted.
"""
