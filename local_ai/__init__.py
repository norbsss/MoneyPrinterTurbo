"""Local image generation: hardware detection, model catalog, downloads and the GPU server.

Everything here except ``server`` and ``pipelines`` uses only the standard library,
so the main app can detect hardware and pick models before torch is installed.
"""
