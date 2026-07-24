from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension(
            "crosscorr.corrdesi",
            sources=[
                "csrc/corr_desi.c",
                "csrc/corr_desi_smu.c",
                "csrc/corr_desi_xyz.c",
            ],
            include_dirs=["include"],
            extra_compile_args=["-fopenmp"],
            extra_link_args=["-fopenmp"],
        )
    ]
)
