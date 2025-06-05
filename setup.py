from setuptools import setup, find_packages

setup(
    name="jax-fsm-mcmc",
    version="0.1.0",
    author="Hugh Dance",
    description="Vectorized FSM-MCMC in JAX",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "jax>=0.4.26,<=0.4.26",
        "jaxlib>=0.4.26,<=0.4.26",
    ],
    python_requires=">=3.9",
)