from setuptools import setup, find_packages

setup(
    name='oaisys-rag',
    version='0.1.0',
    author='Sergey Kurshev',
    author_email='sergey.kurshev@gmail.com',
    description='A brief description of your library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/oaisys-dev/oaisys-rag/',
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=[
        'numpy',  # Add your dependencies here
        # 'requests',
        # 'pandas',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11.9',  # Specify minimum Python version
    include_package_data=True,  # Include files from MANIFEST.in
)